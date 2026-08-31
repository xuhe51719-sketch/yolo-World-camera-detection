# -*- coding: utf-8 -*-
"""
滚动录制模块：持续录制检测画面（最多保留约 1 小时，自动删最旧段）+ 回放路由。

设计要点：
  - 帧队列 maxsize=2，只保留最新帧：录制是尽力而为的旁路功能，
    绝不阻塞检测主线程（队列满时丢最旧帧）；
  - 独立录制线程按 settings.record_fps 节流写段（mp4v/.mp4），
    每 settings.record_segment_s 秒关段开新段；
  - 三级降级：mp4v 打不开 → MJPG/.avi → 停录并记录原因；
  - 磁盘剩余 <2GB 时停录，避免写满系统盘；
  - 段文件名形如 seg_20260831_120000.mp4，按文件名排序即按时间排序，
    清理时删最旧段直至段数不超保留上限。

本模块导入无任何副作用（不创建目录、不启动线程）。
"""

import datetime
import logging
import os
import queue
import re
import shutil
import threading
import time
import traceback

import cv2
from flask import Response, jsonify, request

import state
from config import settings

logger = logging.getLogger(__name__)

# ============================================================
# 模块级状态
# ============================================================
# 帧队列只保留最新 2 帧：录制允许丢帧，绝不允许阻塞检测线程
frame_queue = queue.Queue(maxsize=2)

_recorder_thread = None
_recorder_lock = threading.Lock()

# 录制线程是否存活；停止原因（磁盘不足/编码器全失败时为中文说明，正常为 None）
recorder_active = False
stopped_reason = None

# 磁盘剩余低于该值（字节）时停录，避免写满磁盘
_MIN_DISK_FREE = 2 * 1024 * 1024 * 1024

# 回放路由的文件名白名单：只接受本模块生成的段文件名，防路径穿越；
# 同秒重开段时文件名可能带序号后缀（如 seg_20260831_120000_2.mp4）
_CLIP_NAME_RE = re.compile(r'^seg_\d{8}_\d{6}(_\d+)?\.(mp4|avi)$')
# 回放倍速白名单（上限 2 倍）
_ALLOWED_SPEEDS = (0.5, 1.0, 1.5, 2.0)

# 连续写帧失败达到该次数则重开段（编码器可能已失效）
_WRITE_FAIL_MAX = 30

# 已关闭段的时长缓存 {段文件名: 时长秒}：关段路径本就知道时长，
# /recordings/clips 优先读缓存，避免每次请求逐段打开 VideoCapture（60 段可达数秒）
_duration_cache = {}

# 正在写入的活动段文件名（无活动段时为 None）：clips 列表不列出，
# 与 docstring “正在写入的段不列出”行为一致（尾帧未落盘，时长未知）
_active_segment_name = None


# ============================================================
# 入队（检测线程调用）
# ============================================================
def enqueue(frame):
    """非阻塞入队：队列满时丢最旧帧。调用方入队副本（见 detector 调用处），
    避免录制线程写盘与主管线就地绘制竞争同一缓冲。
    录制关闭 / 已停录时直接跳过，开销仅为一次标志判断。"""
    if not settings.record_enabled or not recorder_active:
        return
    try:
        while True:
            try:
                frame_queue.put_nowait(frame)
                return
            except queue.Full:
                try:
                    frame_queue.get_nowait()   # 丢最旧帧，只留最新画面
                except queue.Empty:
                    pass
    except Exception:
        pass   # 录制是旁路功能：任何异常都不允许影响检测主线程


# ============================================================
# 段文件管理
# ============================================================
def _list_segments():
    """按文件名（即时间）排序列出录制目录中的段文件"""
    try:
        names = [n for n in os.listdir(settings.record_dir)
                 if _CLIP_NAME_RE.match(n)]
    except OSError:
        return []
    return sorted(names)


def _disk_free():
    """录制目录所在盘剩余空间（字节）；查询失败返回 None"""
    try:
        return shutil.disk_usage(settings.record_dir).free
    except OSError:
        return None


def _cleanup_old_segments():
    """删除最旧段，直至段数不超保留上限（约 1 小时滚动窗口）"""
    max_segments = max(1, int(settings.record_retention_min * 60
                              // max(1, settings.record_segment_s)))
    names = _list_segments()
    overflow = names[:-max_segments] if len(names) > max_segments else []
    for name in overflow:
        try:
            os.remove(os.path.join(settings.record_dir, name))
            _duration_cache.pop(name, None)   # 段已删除，同步丢弃缓存时长
            logger.info("滚动清理：已删除最旧录制段 %s", name)
        except OSError as e:
            logger.warning("删除旧录制段 %s 失败: %s", name, e)


def _remember_duration(path, seconds):
    """把已关闭段的时长写入缓存（关段路径调用，后续 clips 列表免开 VideoCapture）"""
    try:
        _duration_cache[os.path.basename(path)] = round(max(0.0, seconds), 1)
    except Exception:
        pass


def _unique_seg_path(ext):
    """生成不重名的段文件路径：秒级时间戳同秒内重开段时追加序号避免覆盖"""
    base = "seg_" + time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(settings.record_dir, base + ext)
    seq = 1
    while os.path.exists(path):
        seq += 1
        path = os.path.join(settings.record_dir, f"{base}_{seq}{ext}")
    return path


def _open_writer(size):
    """尝试打开段写入器，三级降级：mp4v/.mp4 → MJPG/.avi → None。
    返回 (writer, path) 或 (None, None)"""
    candidates = [
        (cv2.VideoWriter_fourcc(*"mp4v"), _unique_seg_path(".mp4")),
        (cv2.VideoWriter_fourcc(*"MJPG"), _unique_seg_path(".avi")),
    ]
    for fourcc, path in candidates:
        try:
            w = cv2.VideoWriter(path, fourcc, float(max(1, settings.record_fps)), size)
            if w.isOpened():
                return w, path
        except Exception:
            pass
        try:
            w.release()
        except Exception:
            pass
    return None, None


# ============================================================
# 录制线程
# ============================================================
def recorder_loop():
    """消费帧队列 → 按 record_fps 节流 → 分段写入。
    全程异常自保护：任何失败只记日志，绝不外抛。"""
    global recorder_active, stopped_reason, _active_segment_name

    recorder_active = True
    stopped_reason = None
    writer = None
    writer_path = None
    frame_size = None
    seg_start = 0.0      # 当前段开始时间（monotonic，节流基准）
    next_write = 0.0     # 下一帧允许写入时刻（monotonic）
    min_interval = 1.0 / max(1, settings.record_fps)
    write_fail_streak = 0

    try:
        os.makedirs(settings.record_dir, exist_ok=True)
        _cleanup_old_segments()   # 启动录制前先清理一次（避免残留超限段）
        logger.info("滚动录制已启动: %s（每段 %ds，保留约 %d 分钟）",
                    settings.record_dir, settings.record_segment_s,
                    settings.record_retention_min)
    except Exception:
        logger.error("录制目录创建失败，停止录制...\n%s", traceback.format_exc())
        stopped_reason = "录制目录创建失败"
        recorder_active = False
        return

    while state.running:
        try:
            try:
                frame = frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # 磁盘剩余 <2GB：停录，不冒险写满磁盘
            free = _disk_free()
            if free is not None and free < _MIN_DISK_FREE:
                stopped_reason = "磁盘剩余空间不足 2GB，已停止录制"
                logger.warning(stopped_reason)
                break

            # 帧尺寸变化（换源/旋转切换）→ 关段重开
            h, w = frame.shape[:2]
            if writer is None or frame_size != (w, h):
                if writer is not None:
                    try:
                        writer.release()
                    except Exception:
                        pass
                    _remember_duration(writer_path, time.monotonic() - seg_start)
                    _active_segment_name = None
                    _cleanup_old_segments()
                    writer = None
                writer, writer_path = _open_writer((w, h))
                if writer is None:
                    stopped_reason = "视频编码器打开失败（mp4v 与 MJPG 均不可用），已停止录制"
                    logger.error(stopped_reason)
                    break
                _active_segment_name = os.path.basename(writer_path)
                frame_size = (w, h)
                seg_start = time.monotonic()
                next_write = 0.0
                write_fail_streak = 0
                logger.info("录制段已打开: %s (%dx%d)", os.path.basename(writer_path), w, h)

            # 按 record_fps 节流：间隔不足的帧直接丢弃
            now = time.monotonic()
            if now < next_write:
                continue
            next_write = now + min_interval

            if not writer.write(frame):
                write_fail_streak += 1
                if write_fail_streak >= _WRITE_FAIL_MAX:
                    logger.warning("连续 %d 帧写入失败，重开录制段...", write_fail_streak)
                    try:
                        writer.release()
                    except Exception:
                        pass
                    _remember_duration(writer_path, now - seg_start)
                    _active_segment_name = None
                    writer = None   # 下轮循环重开段
                continue
            write_fail_streak = 0

            # 段时长到点：关段开新段（时长写入缓存，供 clips 列表免开 VideoCapture）
            if now - seg_start >= settings.record_segment_s:
                try:
                    writer.release()
                except Exception:
                    pass
                _remember_duration(writer_path, now - seg_start)
                _active_segment_name = None
                writer = None
                _cleanup_old_segments()

        except Exception:
            logger.error("录制线程异常（已隔离，继续运行）...\n%s", traceback.format_exc())
            time.sleep(0.5)

    # 收尾：排空队列 + 关闭当前段
    try:
        while True:
            frame_queue.get_nowait()
    except queue.Empty:
        pass
    except Exception:
        pass
    if writer is not None:
        try:
            writer.release()
        except Exception:
            pass
        _remember_duration(writer_path, time.monotonic() - seg_start)
        try:
            _cleanup_old_segments()
        except Exception:
            pass
    _active_segment_name = None
    recorder_active = False
    logger.info("录制线程已退出。")


# ============================================================
# 装配入口（app.py 调用）
# ============================================================
def start_recorder():
    """启动录制线程（幂等）；返回线程对象供 _shutdown join"""
    global _recorder_thread
    with _recorder_lock:
        if _recorder_thread is not None and _recorder_thread.is_alive():
            return _recorder_thread
        _recorder_thread = threading.Thread(target=recorder_loop, name="recorder", daemon=True)
        _recorder_thread.start()
        return _recorder_thread


def stop_recorder(timeout=5.0):
    """停止录制线程（置 state.running 后由 _shutdown 统一调用，此处仅 join）"""
    with _recorder_lock:
        t = _recorder_thread
    if t is not None and t.is_alive():
        t.join(timeout=timeout)
        if t.is_alive():
            logger.warning("录制线程未在 %.0fs 内退出", timeout)


# ============================================================
# 回放路由
# ============================================================
def register_recording_routes(app):
    @app.route('/recordings/status')
    def recordings_status():
        """录制状态：是否运行、段数、总大小、磁盘剩余、停止原因"""
        names = _list_segments()
        total = 0
        for n in names:
            try:
                total += os.path.getsize(os.path.join(settings.record_dir, n))
            except OSError:
                pass
        free = _disk_free()
        return jsonify({
            "enabled": settings.record_enabled,
            "segment_count": len(names),
            "total_size_mb": round(total / 1e6, 1),
            "disk_free_gb": round(free / 1e9, 2) if free is not None else None,
            "stopped_reason": stopped_reason,
        })

    @app.route('/recordings/clips')
    def recordings_clips():
        """已关闭段列表（正在写入的活动段不列出）。
        时长优先读关段时写入的缓存，仅缓存缺失才回退打开 VideoCapture；
        start_ts 为真实 Unix 秒（从文件名解析），绝不抛错。"""
        clips = []
        for name in _list_segments():
            if name == _active_segment_name:
                continue   # 正在写入的活动段尾帧未落盘，不列出（与 docstring 一致）
            path = os.path.join(settings.record_dir, name)
            duration = _duration_cache.get(name)
            if duration is None:
                duration = 0.0
                cap = None
                try:
                    cap = cv2.VideoCapture(path)
                    if cap.isOpened():
                        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        if n_frames > 0 and fps > 0:
                            duration = round(n_frames / fps, 1)
                except Exception:
                    pass
                finally:
                    if cap is not None:
                        try:
                            cap.release()
                        except Exception:
                            pass
            try:
                size_mb = round(os.path.getsize(path) / 1e6, 2)
            except OSError:
                size_mb = 0.0
            try:
                start_ts = int(datetime.datetime.strptime(
                    name[4:19], "%Y%m%d_%H%M%S").timestamp())   # 真实 Unix 秒
            except (ValueError, IndexError):
                start_ts = 0
            clips.append({
                "file": name,
                "start_ts": start_ts,
                "duration_s": duration,
                "size_mb": size_mb,
            })
        return jsonify({"clips": clips})

    @app.route('/recordings/stream')
    def recordings_stream():
        """回放指定段：服务端解码 → multipart MJPEG。
        speed 白名单 {0.5,1,1.5,2}（上限 2 倍），文件名白名单防路径穿越。"""
        clip = os.path.basename((request.args.get("clip") or "").strip())
        if not _CLIP_NAME_RE.match(clip):
            return jsonify({"ok": False,
                            "error": "非法的回放文件名（应为 seg_日期_时间.mp4/.avi）"}), 400

        speed_raw = request.args.get("speed", "1")
        try:
            speed = float(speed_raw)
        except (TypeError, ValueError):
            return jsonify({"ok": False,
                            "error": "回放倍速非法，仅支持 0.5 / 1 / 1.5 / 2"}), 400
        if speed not in _ALLOWED_SPEEDS:
            return jsonify({"ok": False,
                            "error": "回放倍速非法，仅支持 0.5 / 1 / 1.5 / 2"}), 400

        path = os.path.join(settings.record_dir, clip)
        if not os.path.isfile(path):
            return jsonify({"ok": False, "error": "录制段不存在"}), 404
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            return jsonify({"ok": False, "error": "录制段无法打开"}), 404

        try:
            seg_fps = cap.get(cv2.CAP_PROP_FPS)
        except Exception:
            seg_fps = 0.0
        if not seg_fps or seg_fps <= 0:
            seg_fps = float(max(1, settings.record_fps))
        delay = 1.0 / (seg_fps * speed)   # 倍速回放节流间隔

        def _playback():
            try:
                # 帧预算时刻制：每帧按 delay 推进目标时刻，编码耗时不计入间隔，
                # 2 倍速等倍速可真实达标（旧写法 sleep(delay) 会把编码耗时叠加进间隔）
                next_t = time.monotonic()
                while state.running:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                    ok, buffer = cv2.imencode('.jpg', frame,
                                              [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ok:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n'
                               + buffer.tobytes() + b'\r\n')
                    next_t += delay
                    time.sleep(max(0.0, next_t - time.monotonic()))
            except Exception:
                logger.warning("回放异常（已中断）...")
            finally:
                try:
                    cap.release()
                except Exception:
                    pass

        return Response(_playback(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
