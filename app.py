# -*- coding: utf-8 -*-
"""
YOLOv8 实时目标检测系统
支持两种图像源：
  1. 本地摄像头（默认，自动选择索引与后端）
  2. 手机 IP 摄像头（在网页上填推流地址，或设置 PHONE_CAMERA_URL / --url）
通过 Web 前端展示检测结果

模块结构：
  config.py    集中配置（环境变量 + BASE_DIR 路径锚定）
  state.py     共享运行时状态（帧 / 检测结果 / 摄像头信息）
  metrics.py   量化指标采集
  camera.py    摄像头打开 / 重连 / 地址纠错 / 画面方向
  detector.py  模型惰性单例 / 检测循环 / 绘制
  recorder.py  滚动录制（最多 1 小时）/ 回放路由（新增）
  tracks.py    按跟踪 ID 的轨迹绘制（新增）
  zones.py     区域入侵报警 / 事件 / 报警音（新增）
  routes.py    Flask 路由（对外契约不变）
  app.py       create_app() 工厂 + main() 入口

`import app` 无副作用：不加载模型、不占显存、不连摄像头、不触发联网下载。
"""

import argparse
import atexit
import logging
import os
import sys
import threading
import time

from flask import Flask

from config import BASE_DIR, settings

logger = logging.getLogger(__name__)


# ============================================================
# 日志（替代 print；保留 [INFO]/[WARN]/[ERROR] 前缀语义）
# ============================================================
def setup_logging():
    """控制台日志：级别由 settings.log_level 提供（环境变量 LOG_LEVEL 在 config.py 中统一读取，默认 INFO）"""
    logging.addLevelName(logging.WARNING, "WARN")   # 保留原 [WARN] 语义
    level = settings.log_level.upper()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    root = logging.getLogger()
    root.setLevel(getattr(logging, level, logging.INFO))
    if not root.handlers:   # 幂等：重复调用不叠加 handler
        root.addHandler(handler)


# ============================================================
# Flask 应用工厂
# ============================================================
def create_app():
    """创建 Flask 应用并注册全部路由（不启动任何后台线程）"""
    from recorder import register_recording_routes
    from routes import register_routes
    from zones import register_zone_routes

    app = Flask(__name__,
                template_folder=os.path.join(BASE_DIR, "templates"),
                static_folder=os.path.join(BASE_DIR, "static"))
    register_routes(app)
    register_recording_routes(app)   # 滚动录制状态/列表/回放（新增，不影响既有契约）
    register_zone_routes(app)        # 区域/事件/报警（新增，不影响既有契约）
    return app


# ============================================================
# 进程退出清理
# ============================================================
_shutdown_lock = threading.Lock()
_shutdown_done = False


def _shutdown(worker_thread=None, recorder_thread=None, zones_thread=None, timeout=8.0):
    """优雅退出：置运行标志为 False，等待检测/录制/区域检查线程收尾（带超时，
    避免卡死），摄像头由 detection_loop 退出时自行 release。
    atexit 与 KeyboardInterrupt 两条路径都可能触发，用标志保证幂等。"""
    global _shutdown_done
    with _shutdown_lock:
        if _shutdown_done:
            return
        _shutdown_done = True

    import state
    if not state.running:
        return
    logger.info("收到退出信号，正在停止采集/检测线程...")
    state.running = False   # 各循环与重连退避均会检查此标志并退出，同时释放摄像头
    if worker_thread is not None and worker_thread.is_alive():
        worker_thread.join(timeout=timeout)
        if worker_thread.is_alive():
            logger.warning("采集/检测线程未在 %.0fs 内退出，强制结束（资源由操作系统回收）", timeout)
        else:
            logger.info("采集/检测线程已正常退出，摄像头已释放。")
    # 录制与区域检查线程均为轻量旁路线程，短超时 join 即可
    for t, name in ((recorder_thread, "录制"), (zones_thread, "区域检查")):
        if t is not None and t.is_alive():
            t.join(timeout=3.0)
            if t.is_alive():
                logger.warning("%s线程未在 3s 内退出", name)
    logger.info("服务已停止。")


# ============================================================
# 启动
# ============================================================
def main():
    setup_logging()

    # 让终端输出立即刷新，方便看到启动进度
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    # 只显示 OpenCV 错误日志，屏蔽摄像头驱动的多余信息
    try:
        import cv2
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="YOLOv8 实时目标检测（支持手机摄像头）")
    parser.add_argument('--url', default=None,
                        help='手机 IP 摄像头流地址，如 http://192.168.1.100:8080/video')
    parser.add_argument('--port', type=int, default=None, help='Web 服务端口')
    parser.add_argument('--host', default=None,
                        help='监听地址。默认 127.0.0.1（仅本机）；'
                             '局域网访问（如手机浏览器查看）需显式传 0.0.0.0')
    args = parser.parse_args()

    # 命令行参数优先，其次环境变量（环境变量在 config.load_config 中已读取）
    from camera import normalize_camera_url
    if args.url:
        settings.phone_camera_url = normalize_camera_url(args.url)
    if args.port:
        settings.flask_port = args.port
    if args.host:
        settings.flask_host = args.host

    print("=" * 60)
    print("  YOLOv8 实时目标检测系统")
    print(f"  模型: {settings.model_path}")
    print(f"  置信度阈值: {settings.confidence_threshold}")
    print(f"  推理精度: {'FP16 半精度' if settings.use_half else 'FP32'}（环境变量 YOLO_HALF 可切换）")
    if settings.phone_camera_url:
        print(f"  图像源: 手机摄像头 {settings.phone_camera_url}")
    else:
        print("  图像源: 本地摄像头（自动选择）")
    if settings.record_enabled:
        print(f"  录像: 开启（保留约 {settings.record_retention_min} 分钟滚动，"
              f"每段 {settings.record_segment_s}s，RECORD_ENABLED=0 可关闭）")
    else:
        print("  录像: 关闭（RECORD_ENABLED=1 可开启）")
    print(f"  打开浏览器访问: http://{'localhost' if settings.flask_host == '127.0.0.1' else settings.flask_host}:{settings.flask_port}")
    if settings.flask_host != '127.0.0.1':
        logger.warning("已监听 %s，局域网内其他设备可访问本服务", settings.flask_host)
    print("=" * 60)

    # 把 uptime 锚定在"服务启动"（而非模块导入时刻），在线程启动前显式赋值
    from metrics import metrics as _metrics, metrics_lock as _metrics_lock
    with _metrics_lock:
        _metrics["start_time"] = time.time()

    # 启动后台采集/检测线程（退出时置 state.running=False 即可让其收尾，
    # daemon 仅为兜底：正常退出路径由 _shutdown 带超时 join）
    from detector import detection_loop
    t = threading.Thread(target=detection_loop, daemon=True)
    t.start()

    # 旁路线程：滚动录制（可通过 RECORD_ENABLED=0 关闭）与区域入侵检查（约 5Hz）
    recorder_thread = None
    if settings.record_enabled:
        from recorder import start_recorder
        recorder_thread = start_recorder()
    from zones import zones_check_loop
    zones_thread = threading.Thread(target=zones_check_loop, name="zones-check", daemon=True)
    zones_thread.start()

    atexit.register(_shutdown, worker_thread=t,
                    recorder_thread=recorder_thread, zones_thread=zones_thread)

    # 给后台线程一点时间连接图像源并完成第一帧检测
    import state
    logger.info("正在初始化图像源和模型，请稍候...")
    timeout = 30
    start = time.time()
    while state.latest_frame_bytes is None and time.time() - start < timeout:
        time.sleep(0.5)
    if state.latest_frame_bytes is not None:
        logger.info("初始化完成，开始服务。")
    else:
        logger.warning("初始化超时，服务仍会启动，但可能还没有视频画面。")

    app = create_app()
    try:
        app.run(host=settings.flask_host, port=settings.flask_port, threaded=True)
    except KeyboardInterrupt:
        # Ctrl+C：先打印再走清理（atexit 亦会兜底，_shutdown 幂等不会重复执行）
        logger.info("捕获到 Ctrl+C，开始清理...")
        _shutdown(worker_thread=t, recorder_thread=recorder_thread,
                  zones_thread=zones_thread)


if __name__ == '__main__':
    main()
