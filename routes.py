# -*- coding: utf-8 -*-
"""
Flask 路由：对外行为契约保持不变 ——
8 个路由（/、/video_feed、/detections、/model_info、/source、/metrics、
/set_orientation、/set_source）的 URL、HTTP 方法、返回 JSON 字段名均与原实现一致；
另新增轻量 /health 看门狗端点（不影响既有契约）。
"""

import logging
import os
import re
import time
import urllib.parse

import numpy as np
from flask import Response, jsonify, render_template, request

import state
from config import settings
from detector import get_model
from metrics import metrics, metrics_lock

logger = logging.getLogger(__name__)

# /set_source 允许的推流协议（白名单）
ALLOWED_SOURCE_SCHEMES = ("http", "https", "rtsp")
# 主机名只允许字母/数字/点/连字符/下划线（手机推流 App 的设备名常含下划线，如 my_phone）；
# 连续点（如 'foo..bar'）在 normalize_and_validate 中另行显式拒绝。
# 防 'javascript:alert(1)' 被自动补全协议后伪装成地址（其主机名含括号/冒号，不匹配本正则）
_HOSTNAME_RE = re.compile(r'^[A-Za-z0-9._\-]+$')


# ============================================================
# 视频流生成器
# ============================================================
def generate_frames():
    """从后台线程取最新帧，推送给浏览器。
    终止条件：state.running 置 False（优雅退出）后循环退出，
    不再无限挂起客户端连接"""
    while state.running:
        with state.frame_lock:
            frame_bytes = state.latest_frame_bytes

        if frame_bytes is None:
            time.sleep(0.05)
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(1.0 / settings.target_fps)


# ============================================================
# 路由注册
# ============================================================
def register_routes(app):
    @app.route('/')
    def index():
        """主页"""
        return render_template('index.html')

    @app.route('/video_feed')
    def video_feed():
        """MJPEG 视频流端点"""
        return Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/detections')
    def detections():
        """获取最新检测结果（JSON）"""
        with state.detections_lock:
            data = list(state.latest_detections)
        return jsonify(data)

    @app.route('/model_info')
    def model_info():
        """获取模型信息"""
        model = get_model()
        try:
            device = str(next(model.model.parameters()).device)
        except Exception:
            device = "unknown"
        return jsonify({
            # 对外展示用文件名（避免下游页面显示冗长绝对路径）；内部加载仍用完整路径
            "model": os.path.basename(settings.model_path),
            "confidence_threshold": settings.confidence_threshold,
            "imgsz": settings.detect_imgsz,
            "open_vocab": settings.is_world_model,
            "total_classes": len(model.names),
            "class_names": model.names,
            "camera": state.get_camera_info(),
            "device": device
        })

    @app.route('/source')
    def source():
        """返回当前图像源信息"""
        url = state.phone_url_override if state.phone_url_override else settings.phone_camera_url
        return jsonify({
            "configured_url": settings.phone_camera_url,
            "active_url": state.phone_url_override,
            "resolved_url": url,
            "camera": state.get_camera_info(),
            "connected": state.camera_connected,
            "frame_mean": round(state.frame_mean, 1),
            "black_warning": state.black_warning,
            "orientation": {"rotate": state.phone_orientation["rotate"],
                            "mirror": state.phone_orientation["mirror"]}
        })

    @app.route('/metrics')
    def metrics_api():
        """量化指标接口：性能评估与调参用"""
        with metrics_lock:
            loop_arr = np.array(metrics["loop_times"]) if metrics["loop_times"] else None
            infer_arr = np.array(metrics["infer_times"]) if metrics["infer_times"] else None
            pipeline_fps = round(1.0 / float(loop_arr.mean()), 1) if loop_arr is not None and loop_arr.mean() > 0 else 0
            # infer_fps：隔帧推理口径的实际推理频率。infer_times 窗口只记录耗时、无时间戳，
            # 无法直接算"帧数/窗口时长"，用 pipeline_fps / detection_interval 近似。
            infer_fps = round(pipeline_fps / settings.detection_interval, 1) if pipeline_fps else 0
            data = {
                "uptime_s": round(time.time() - metrics["start_time"], 1),
                "pipeline_fps": pipeline_fps,
                "loop_ms_avg": round(float(loop_arr.mean()) * 1000, 1) if loop_arr is not None else 0,
                "infer_ms_avg": round(float(infer_arr.mean()) * 1000, 1) if infer_arr is not None else 0,
                "infer_ms_p95": round(float(np.percentile(infer_arr, 95)) * 1000, 1) if infer_arr is not None else 0,
                "id_switches_total": metrics["id_switches"],
                "detections_total": metrics["detections_total"],
                "top_classes": metrics["class_counts"].most_common(10),
                "connected": state.camera_connected,
                "model": os.path.basename(settings.model_path),
                "imgsz": settings.detect_imgsz,
                "detect_interval": settings.detection_interval,
                "infer_fps": infer_fps,
                "conf": settings.confidence_threshold,
                "half": settings.use_half,
            }
        # GPU 显存（尽力而为）
        try:
            import torch
            if torch.cuda.is_available():
                data["gpu_mem_mb"] = round(torch.cuda.memory_allocated() / 1e6, 1)
        except Exception:
            pass
        return jsonify(data)

    @app.route('/set_orientation', methods=['POST'])
    def set_orientation():
        """调整画面方向：rotate 取 0/90/180/270（顺时针），mirror 取 ""/"h"/"v"
        即时生效，无需重连"""
        data = request.get_json(silent=True) or {}
        if 'rotate' in data:
            try:
                rot = int(data['rotate'])
            except (TypeError, ValueError):
                rot = 0
            if rot in (0, 90, 180, 270):
                state.phone_orientation["rotate"] = rot
        if 'mirror' in data:
            mir = data['mirror']
            if mir in ('', 'h', 'v'):
                state.phone_orientation["mirror"] = mir
        logger.info("画面方向已更新: 旋转%s° 镜像=%s",
                    state.phone_orientation['rotate'],
                    state.phone_orientation['mirror'] or '无')
        return jsonify({"ok": True, "orientation": state.phone_orientation})

    @app.route('/health')
    def health():
        """轻量健康检查（运维看门狗用）：不触碰模型/摄像头，只读共享状态。
        frame_fresh：最近 3 秒内是否有新帧写入。"""
        now = time.time()
        last = state.last_frame_time
        return jsonify({
            "ok": True,
            "running": state.running,
            "frame_fresh": bool(last and now - last <= 3.0),
            "frame_age_s": round(now - last, 2) if last else None,
            "uptime_s": round(now - metrics["start_time"], 1),
        })

    @app.route('/set_source', methods=['POST'])
    def set_source():
        """切换图像源：提交手机流地址，或留空恢复本地摄像头"""
        data = request.get_json(silent=True) or {}
        raw = (data.get('url') or '').strip()
        url = normalize_and_validate(raw) if raw else ''
        if isinstance(url, tuple):   # 校验失败：(错误信息, 400)
            msg, code = url
            logger.warning("图像源地址被拒绝: %r -> %s", raw, msg)
            return jsonify({"ok": False, "error": msg}), code
        if url and url != raw:
            logger.info("地址已自动纠正: %r -> %r", raw, url)
        state.phone_url_override = url
        state.force_reconnect = True
        return jsonify({"ok": True, "url": url})


def normalize_and_validate(raw):
    """先做自动纠错（全角转半角、补协议头、候选 /video 逻辑在连接时处理），
    再按协议白名单校验：仅允许 http/https/rtsp 且必须有 hostname，
    否则返回 (中文错误信息, 400)。"""
    from camera import normalize_camera_url
    url = normalize_camera_url(raw)
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme.lower() not in ALLOWED_SOURCE_SCHEMES:
        return (f"不支持的推流协议 '{parsed.scheme or '无'}'，仅允许 "
                f"{'/'.join(ALLOWED_SOURCE_SCHEMES)}", 400)
    if not parsed.hostname:
        return "推流地址缺少主机名（形如 http://192.168.1.100:8080/video）", 400
    if not _HOSTNAME_RE.match(parsed.hostname):
        return (f"推流地址主机名非法：'{parsed.hostname}'"
                f"（仅允许字母、数字、点(.)、连字符(-)和下划线(_)）"), 400
    if '..' in parsed.hostname:
        return f"推流地址主机名非法：'{parsed.hostname}'（不允许连续的点）", 400
    try:
        parsed.port   # 端口非数字（如 'javascript:alert(1)' 里的 'alert(1)'）会招 ValueError
    except ValueError:
        return "推流地址端口非法（应为数字或省略）", 400
    return url
