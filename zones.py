# -*- coding: utf-8 -*-
"""
区域入侵报警模块：多边形区域管理 + 入侵检查线程 + 事件 + 服务端报警音。

设计要点：
  - 区域以归一化坐标 (0~1) 存储，与画面分辨率解耦；锁 + 原子替换
    （仿 state.camera_info 模式），持久化用临时文件 + os.replace 原子写；
  - 检查线程约 5Hz：仅对 ALERT_CLASSES（人员/动物）判定，取框底边中心点，
    乘 state.display_size 转像素后逐区域做点在多边形内判定；
  - 进入沿触发：维护 (目标指纹, 区域) 的区内状态集合，仅区外→区内触发，
    滞留区内不重复；目标无跟踪 ID 时用 bbox 指纹 + 短冷却兜底；
  - 同一 (区域, 类别) 有 alert_cooldown_s 冷却；
  - 报警音：numpy+wave 合成 880/660Hz 交替警笛 WAV 到临时路径（零新依赖），
    winsound 异步循环播放，Timer 到时停止；播放中再触发只重置计时器；
    非 Windows / 异常仅记日志，绝不影响主流程。

本模块导入无任何副作用（不加载文件、不发声、不起线程）。
"""

import itertools
import json
import logging
import os
import sys
import threading
import time
import traceback

import numpy as np
from flask import jsonify, request

import state
from config import settings
from config_classes import ALERT_CLASSES

logger = logging.getLogger(__name__)

# ============================================================
# 常量与模块级状态
# ============================================================
CHECK_INTERVAL = 0.2        # 检查线程周期（约 5Hz）
_FINGERPRINT_TTL = 1.0      # 无 ID 目标的 bbox 指纹区内状态存活（秒），兜底防抖
_SOUND_SAMPLE_RATE = 22050  # 合成报警音采样率

_zones = []                 # [{id, name, points:[[x,y],...]}]，归一化坐标
_zones_lock = threading.Lock()

_event_id_seq = itertools.count(1)   # 事件 ID 自增（GIL 下 next() 原子，无需额外锁）

# (目标指纹, 区域id) -> 最近确认在区内的时间（进入沿触发 + 无 ID 冷却兜底）
_inside = {}
_inside_lock = threading.Lock()

_last_alert = {}            # (区域id, 类别) -> 最近触发时间（冷却）
_last_alert_lock = threading.Lock()

# 报警音
_alarm_wav_path = None
_alarm_timer = None
_alarm_lock = threading.Lock()
_alarm_active = False


# ============================================================
# 纯函数：点在多边形内（射线法，支持凹多边形）
# ============================================================
def point_in_polygon(px, py, points):
    """射线法（ray casting）：从 (px,py) 发水平射线，与多边形边交点数为奇则在内部。
    对凸/凹多边形均适用；points 为 [(x, y), ...]（与待判定点同一坐标系）"""
    inside = False
    n = len(points)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = points[i]
        xj, yj = points[j]
        # 边 (j, i) 跨越射线高度，且交点在待判定点右侧
        if (yi > py) != (yj > py):
            x_cross = (xj - xi) * (py - yi) / (yj - yi) + xi
            if px < x_cross:
                inside = not inside
        j = i
    return inside


# ============================================================
# 区域存储（锁 + 原子替换）与持久化
# ============================================================
def get_zones():
    """读取侧：锁内拷贝一份返回，调用方可安全迭代/序列化"""
    with _zones_lock:
        return [dict(z) for z in _zones]


def _normalize_points(pts):
    """逐点数值化 + 范围检查（与 validate_regions 等价）；任一顶点非法返回 None。
    NaN/非数字/超范围均拒，避免损坏的 zones.json 让检查线程每 0.2s 抛错"""
    out = []
    for p in pts:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            return None
        try:
            x, y = float(p[0]), float(p[1])
        except (TypeError, ValueError):
            return None
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):   # NaN 比较为 False，一并拒收
            return None
        out.append([x, y])
    return out


def _load_zones_from_file():
    """从 zones.json 加载区域定义；文件不存在/损坏只记日志不报错（惰性调用）。
    非数字坐标/NaN 等非法区域跳过并记一次警告，绝不让检查线程周期性抛错"""
    global _zones
    if not os.path.isfile(settings.zones_file):
        return
    try:
        with open(settings.zones_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        regions = data.get("regions") if isinstance(data, dict) else None
        if not isinstance(regions, list):
            return
        valid = []
        skipped = 0
        for i, r in enumerate(regions):
            if not isinstance(r, dict):
                skipped += 1
                continue
            pts = r.get("points")
            if not isinstance(pts, list) or len(pts) < 3:
                skipped += 1
                continue
            norm_pts = _normalize_points(pts)
            if norm_pts is None:
                skipped += 1
                continue
            valid.append({
                "id": r.get("id", f"zone_{i + 1}"),
                "name": r.get("name") or f"区域 {i + 1}",
                "points": norm_pts,
            })
        with _zones_lock:
            _zones = valid
        if skipped:
            logger.warning("加载区域文件时跳过 %d 个非法区域（坐标非数字/NaN/超范围等）", skipped)
        logger.info("已从 %s 加载 %d 个警戒区域", settings.zones_file, len(valid))
    except Exception as e:
        logger.warning("加载区域文件失败（忽略，使用空区域）: %s", e)


def _save_zones_to_file(regions):
    """原子写：先写临时文件再 os.replace，避免半截 JSON 污染既有配置"""
    try:
        tmp_path = settings.zones_file + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump({"regions": regions}, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, settings.zones_file)
    except Exception as e:
        logger.warning("保存区域文件失败: %s", e)


def clear_zones():
    """清空区域（/set_orientation 调用：旋转后旧区域坐标失效，宁可让用户重画）。
    同时清空区内状态与报警冷却：否则冷却期内重画同 ID 区域时，
    首次入侵会被旧冷却吞掉不报警"""
    global _zones
    with _zones_lock:
        _zones = []
    with _inside_lock:
        _inside.clear()
    with _last_alert_lock:
        _last_alert.clear()
    _save_zones_to_file([])


def validate_regions(data):
    """校验 POST /zones 请求体。合法返回 (regions, None)，非法返回 (None, 错误信息)。
    约束：区域数 ≤10、每区域 ≥3 点、每点为 [x, y] 且坐标 ∈ [0,1]"""
    if not isinstance(data, dict) or not isinstance(data.get("regions"), list):
        return None, '请求体应包含 "regions" 数组'
    regions = data["regions"]
    if len(regions) > 10:
        return None, "区域数量过多（最多 10 个）"
    out = []
    for i, r in enumerate(regions):
        if not isinstance(r, dict):
            return None, f"区域 #{i + 1} 格式非法（应为对象）"
        pts = r.get("points")
        if not isinstance(pts, list) or len(pts) < 3:
            return None, f"区域 #{i + 1} 至少需要 3 个顶点"
        norm_pts = []
        for p in pts:
            if not isinstance(p, (list, tuple)) or len(p) != 2:
                return None, f"区域 #{i + 1} 的每个顶点应为 [x, y] 二元组"
            try:
                x, y = float(p[0]), float(p[1])
            except (TypeError, ValueError):
                return None, f"区域 #{i + 1} 的顶点坐标应为数字"
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                return None, "顶点坐标应为 0~1 之间的归一化值"
            norm_pts.append([x, y])
        name_raw = r.get("name")
        if name_raw is not None and not isinstance(name_raw, str):
            return None, f"区域 #{i + 1} 的名称应为字符串"
        name = (name_raw or "").strip() or f"区域 {i + 1}"
        if len(name) > 50:
            return None, f"区域 #{i + 1} 的名称过长（最多 50 个字符）"
        out.append({"id": f"zone_{i + 1}", "name": name, "points": norm_pts})
    return out, None


# ============================================================
# 事件
# ============================================================
def _record_event(zone_id, zone_name, cls_name, track_id, confidence):
    """写入事件环形缓冲并返回事件对象"""
    now = time.time()
    event = {
        "id": next(_event_id_seq),
        "ts": now,
        "iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
        "zone_id": zone_id,
        "zone_name": zone_name,
        "class": cls_name,
        "track_id": track_id,
        "confidence": confidence,
    }
    with state.events_lock:
        state.events.append(event)
    return event


def _cooldown_ok(zone_id, cls_name):
    """同一 (区域, 类别) 冷却判定：冷却内返回 False"""
    key = (zone_id, cls_name)
    now = time.monotonic()
    with _last_alert_lock:
        last = _last_alert.get(key)
        if last is not None and now - last < settings.alert_cooldown_s:
            return False
        _last_alert[key] = now
    return True


# ============================================================
# 服务端报警音（numpy+wave 合成，winsound 播放，零新依赖）
# ============================================================
def _ensure_alarm_wav():
    """惰性合成 10 秒内循环警笛（880/660Hz 交替）并写到临时路径；只执行一次"""
    global _alarm_wav_path
    if _alarm_wav_path and os.path.isfile(_alarm_wav_path):
        return _alarm_wav_path
    try:
        import tempfile
        import wave
        tone_dur = 0.25   # 每个音调 0.25 秒，880/660 交替即经典警笛
        n_total = _SOUND_SAMPLE_RATE * settings.alert_sound_s
        n_tone = _SOUND_SAMPLE_RATE // 4
        t = np.arange(n_tone) / _SOUND_SAMPLE_RATE
        hi = np.sin(2 * np.pi * 880 * t)
        lo = np.sin(2 * np.pi * 660 * t)
        segs = []
        for i in range(0, n_total, n_tone):
            segs.append(hi if (i // n_tone) % 2 == 0 else lo)
        sig = np.concatenate(segs)[:n_total]
        # 首尾 10ms 淡出，避免爆音
        fade = max(1, int(_SOUND_SAMPLE_RATE * 0.01))
        sig[-fade:] *= np.linspace(1.0, 0.0, fade)
        pcm = (sig * 0.5 * 32767).astype("<i2")
        # 文件名带进程 PID：多实例并存时互不覆盖对方的报警音文件，
        # 进程退出后残留文件也易于辨识归属（临时目录由系统自行清理）
        path = os.path.join(tempfile.gettempdir(), f"yolo_alert_siren_{os.getpid()}.wav")
        with wave.open(path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(_SOUND_SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())
        _alarm_wav_path = path
    except Exception:
        logger.warning("报警音合成失败（将无声报警）...")
        _alarm_wav_path = None
    return _alarm_wav_path


def _stop_alarm_sound():
    """停止 winsound 播放并把报警状态复位"""
    global _alarm_active
    with _alarm_lock:
        _alarm_active = False
    if sys.platform != "win32":
        return
    try:
        import winsound
        winsound.PlaySound(None, 0)
    except Exception:
        pass


def trigger_alarm():
    """触发一次报警音：异步循环播放 alert_sound_s 秒。
    播放中再次触发只重置停止计时器，不叠加播放。"""
    global _alarm_timer, _alarm_active
    if sys.platform != "win32":
        logger.info("（非 Windows 环境，报警音跳过，仅记录事件）")
        return
    try:
        wav = _ensure_alarm_wav()
        if wav:
            import winsound
            with _alarm_lock:
                if not _alarm_active:
                    winsound.PlaySound(wav,
                                       winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
                    _alarm_active = True
                # cancel 与重新赋值同在锁内：消除并发触发时计时器引用丢失/重复 cancel 的缺口
                if _alarm_timer is not None:
                    _alarm_timer.cancel()   # 重置计时器而不是叠加播放
                _alarm_timer = threading.Timer(settings.alert_sound_s, _stop_alarm_sound)
                _alarm_timer.daemon = True
                _alarm_timer.start()
    except Exception:
        logger.warning("播放报警音失败（已忽略）...")


# ============================================================
# 区域检查线程（约 5Hz）
# ============================================================
def zones_check_loop():
    """持续检查 ALERT_CLASSES 目标是否进入警戒区域（进入沿触发）。
    全程异常自保护，绝不外抛。"""
    global _zones
    _load_zones_from_file()   # 启动时加载一次（文件不存在不报错）
    logger.info("区域入侵检查线程已启动（约 5Hz）")

    while state.running:
        try:
            zones = get_zones()
            if zones:
                with state.detections_lock:
                    dets = list(state.latest_detections)
                w, _h = state.display_size
                if w > 0 and _h > 0:
                    now = time.monotonic()
                    with _inside_lock:
                        # 清掉过期指纹（无 ID 目标的兜底条目，避免集合无限膨胀）
                        expired = [k for k, ts in _inside.items()
                                   if now - ts > _FINGERPRINT_TTL * 6]
                        for k in expired:
                            del _inside[k]

                        for det in dets:
                            if det.get("class") not in ALERT_CLASSES:
                                continue
                            x1, _y1, x2, y2 = det["bbox"]
                            # 底边中心点：bbox 已是校正后画面的像素坐标（检测与绘制同源），
                            # 直接用于与区域像素坐标同系判定；w/h 仅用于确认画面已就绪
                            cx = (x1 + x2) / 2.0
                            cy = y2
                            track_id = det.get("id")
                            # 目标指纹：有跟踪 ID 用 ID；否则用粗量化 bbox（兜底）
                            if track_id is not None:
                                target_key = ("id", track_id)
                            else:
                                target_key = ("box", round(x1 / 24), round(_y1 / 24),
                                              round(x2 / 24), round(y2 / 24))

                            for zone in zones:
                                # 区域点为归一化坐标，乘当前画面尺寸转像素后同系判定
                                pix_pts = [(p[0] * w, p[1] * _h) for p in zone["points"]]
                                key = (target_key, zone["id"])
                                was_inside = key in _inside
                                is_inside = point_in_polygon(cx, cy, pix_pts)
                                if is_inside:
                                    _inside[key] = now
                                else:
                                    _inside.pop(key, None)
                                if not is_inside or was_inside:
                                    continue   # 未进入 / 已在区内（滞留不重复）
                                # --- 进入沿：触发报警 ---
                                if not _cooldown_ok(zone["id"], det["class"]):
                                    continue
                                _record_event(zone["id"], zone["name"],
                                              det["class"], track_id,
                                              det.get("confidence"))
                                logger.warning(
                                    "区域入侵: %s 进入「%s」（id=%s conf=%.2f）",
                                    det["class"], zone["name"], track_id,
                                    det.get("confidence") or 0.0)
                                trigger_alarm()
        except Exception:
            logger.error("区域检查线程异常（已隔离，继续运行）...\n%s",
                         traceback.format_exc())
        time.sleep(CHECK_INTERVAL)

    logger.info("区域检查线程已退出。")


# ============================================================
# 路由
# ============================================================
def register_zone_routes(app):
    @app.route('/zones', methods=['GET'])
    def zones_get():
        """当前区域列表"""
        return jsonify({"regions": get_zones()})

    @app.route('/zones', methods=['POST'])
    def zones_post():
        """整体替换区域定义（校验失败返回 400 + 中文错误）"""
        data = request.get_json(silent=True)
        regions, error = validate_regions(data)
        if error is not None:
            return jsonify({"ok": False, "error": error}), 400
        global _zones
        with _zones_lock:
            _zones = regions
        _save_zones_to_file(regions)
        # 区域变了：区内状态与冷却全部失效
        with _inside_lock:
            _inside.clear()
        with _last_alert_lock:
            _last_alert.clear()
        logger.info("警戒区域已更新（%d 个）", len(regions))
        return jsonify({"ok": True, "regions": get_zones()})

    @app.route('/events')
    def events_get():
        """入侵事件列表（最新在后）+ 报警音是否正在播放"""
        with state.events_lock:
            evs = list(state.events)
        with _alarm_lock:
            active = _alarm_active
        return jsonify({
            "events": evs,
            "latest_id": evs[-1]["id"] if evs else 0,
            "alarm_active": active,
        })

    @app.route('/alert/test', methods=['POST'])
    def alert_test():
        """手动触发一次报警（联调用）：写一条"测试"事件并响铃"""
        _record_event("test", "测试", "person", None, 1.0)
        trigger_alarm()
        return jsonify({"ok": True})
