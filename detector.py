# -*- coding: utf-8 -*-
"""
检测模块：模型惰性加载、后台采集/检测循环、检测框绘制。
从原 app.py 拆分而来；本模块导入不加载模型、不占显存、不触发联网下载，
模型在首次调用 get_model() 时才加载（线程安全单例）。
"""

import logging
import threading
import time
import traceback

import cv2
import numpy as np

import state
from camera import apply_orientation, open_camera
from config import settings
from config_classes import DETECTION_CLASSES
from metrics import metrics, metrics_lock, reset_source_metrics

logger = logging.getLogger(__name__)

# ============================================================
# 模型惰性单例
# ============================================================
_model = None
_model_lock = threading.Lock()


def get_model():
    """首次调用才加载模型（惰性单例，线程安全）。
    保证 `import app` 不再加载模型、不占显存、不触发联网下载。"""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:   # double-checked locking
                from ultralytics import YOLO   # 延迟到真正需要时才引入
                m = YOLO(settings.model_path)
                if settings.is_world_model:
                    m.set_classes(DETECTION_CLASSES)
                    logger.info("开放词汇模型已加载: %s，注册 %d 个检测类别",
                                settings.model_path, len(DETECTION_CLASSES))
                else:
                    logger.info("常规模型已加载: %s（固定 %d 类）",
                                settings.model_path, len(m.names))
                _model = m
    return _model


# ============================================================
# 框计算与颜色映射
# ============================================================
def _box_iou(a, b):
    """计算两个 [x1,y1,x2,y2] 框的 IoU"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + \
            max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def get_color(idx):
    """根据索引生成 BGR 颜色"""
    np.random.seed(42)
    colors = np.random.randint(60, 255, size=(100, 3))
    return tuple(int(c) for c in colors[idx % 100])


def draw_detection_boxes(frame, detections):
    """在画面上绘制检测框与标签（新推理结果与复用的历史结果均可绘制）"""
    model = get_model()
    for det in detections:
        cls_name = det["class"]
        conf = det["confidence"]
        box_id = det.get("id")
        x1, y1, x2, y2 = det["bbox"]
        cls_id = next((i for i, n in model.names.items() if n == cls_name), 0)
        color = get_color(cls_id)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{cls_name} {conf:.2f}" + (f" #{box_id}" if box_id is not None else "")
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (int(x1), int(y1) - th - 8), (int(x1) + tw + 4, (int(y1))), color, -1)
        cv2.putText(frame, label, (int(x1) + 2, (int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


# ============================================================
# 后台采集与检测线程
# ============================================================
# 切换图像源时置位：下一次 track 用 persist=False 重建 ByteTrack 会话，
# 清掉旧源的跟踪状态；之后恢复 persist=True，跟踪能力不受影响
_tracker_reset_requested = False


def request_tracker_reset():
    """请求重建跟踪会话（切换图像源时调用）"""
    global _tracker_reset_requested
    _tracker_reset_requested = True


def detection_loop():
    """持续读取图像源、运行 YOLO 检测、更新最新帧和结果。
    整个循环体有异常保护：任何错误都会打印堆栈并重置连接后重试，
    采集线程不会静默死亡（否则一旦启动时手机未就绪，服务将永远无法恢复）。"""
    global _tracker_reset_requested

    # 模型加载（带退避重试）：权重缺失 / CUDA 异常等加载失败不能让异常穿透
    # 线程函数导致采集线程静默死亡；每 5 秒重试一次，直到成功或服务停止（此时 state.running 置 False，循环退出）
    model = None
    while state.running:
        try:
            model = get_model()
            break
        except Exception:
            logger.error("模型加载失败，5 秒后重试...\n%s", traceback.format_exc())
            time.sleep(5)
    if model is None:
        return   # 重试期间收到退出信号（state.running=False），直接结束线程

    cap = None
    fail_count = 0    # 连续失败计数：失败分支指数退避用，避免持续故障时重连风暴/日志风暴；成功读到帧即复位
    while state.running:
        t_loop_start = time.time()   # 本轮循环计时起点（量化指标用）
        try:
            if state.force_reconnect:
                if cap is not None:
                    cap.release()
                cap = None
                state.force_reconnect = False
                # 切换图像源：清零与旧源绑定的跟踪/检出指标，并请求重建
                # ByteTrack 会话（旧源的跟踪状态对新画面毫无意义，还会误报 ID 切换）
                reset_source_metrics()
                request_tracker_reset()
                with state.detections_lock:
                    state.latest_detections = []   # 清掉旧源残留的检测框
                logger.info("收到切换图像源请求，正在重新连接...")

            if cap is None or not cap.isOpened():
                cap = open_camera()
                if cap is None:
                    state.camera_connected = False
                    logger.error("没有可用的图像源，即将重试...")
                    time.sleep(min(0.3 * (2 ** min(fail_count, 4)), 5.0))
                    fail_count += 1
                    continue
                state.camera_connected = True

            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("读取帧失败，尝试重新连接图像源...")
                cap.release()
                cap = None
                state.camera_connected = False
                time.sleep(min(0.3 * (2 ** min(fail_count, 4)), 5.0))
                fail_count += 1
                continue
            fail_count = 0   # 成功读到帧，退避计数复位

            # 画面方向校正（手机流方向固定，在电脑端旋转/镜像；检测框画在校正后的图上）
            frame = apply_orientation(frame)

            # 黑帧检测：整帧平均亮度过低且持续，则标记黑屏
            state.frame_mean = float(frame.mean())
            if state.frame_mean >= settings.black_mean_threshold:
                state.last_nonblack_time = time.time()
            state.black_warning = (time.time() - state.last_nonblack_time) > 1.5

            state.frame_count += 1

            # 每隔 N 帧执行一次检测：推理是管线中最耗时环节，非检测帧复用上次结果，
            # 降低推理负载、提高消费帧率，避免读流端积压（延迟根因之一）
            if state.frame_count % settings.detection_interval == 0:
                # track 而非 predict：内置 ByteTrack 多目标跟踪，每个物体有稳定 ID，
                # 检测框不再逐帧闪烁。persist=True 使跟踪器状态跨跳帧保持，
                # 隔帧推理仍可维持 ID 连续性。
                # 必须显式指定 bytetrack：8.4.x 默认的 TRACKTRACK 依赖 ReID 特征，
                # 与 YOLO-World 结构不兼容（实测 0 跟踪结果）
                persist = not _tracker_reset_requested
                t_infer_start = time.time()
                results = model.track(frame, persist=persist, conf=settings.confidence_threshold,
                                      imgsz=settings.detect_imgsz, tracker="bytetrack.yaml",
                                      half=settings.use_half, verbose=False)
                _tracker_reset_requested = False   # persist=False 已重建跟踪会话，恢复持久跟踪
                infer_dt = time.time() - t_infer_start

                current_detections = []
                if results and results[0].boxes is not None:
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        cls_name = model.names[cls_id]
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        box_id = int(box.id[0]) if box.id is not None else None

                        current_detections.append({
                            "class": cls_name,
                            "confidence": round(conf, 4),
                            "id": box_id,
                            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
                        })

                # 更新全局检测结果（仅推理帧更新，跳帧沿用上次结果）
                with state.detections_lock:
                    state.latest_detections = current_detections

                # --- 量化指标：推理耗时、ID 切换、类别计数 ---
                with metrics_lock:
                    metrics["infer_times"].append(infer_dt)
                    curr_tracks = [(d["class"], d["bbox"], d["id"]) for d in current_detections
                                   if d["id"] is not None]
                    for cls_c, box_c, id_c in curr_tracks:
                        best_id, best_iou = None, 0.3
                        for cls_p, box_p, id_p in metrics["prev_tracks"]:
                            if cls_p != cls_c:
                                continue
                            v = _box_iou(box_c, box_p)
                            if v > best_iou:
                                best_iou, best_id = v, id_p
                        if best_id is not None and best_id != id_c:
                            metrics["id_switches"] += 1   # 同一物体 ID 变了 = 一次切换
                    metrics["prev_tracks"] = curr_tracks
                    metrics["detections_total"] += len(current_detections)
                    for d in current_detections:
                        metrics["class_counts"][d["class"]] += 1

            # 绘制检测框：推理帧用最新结果，跳帧复用上次结果（避免框逐帧闪烁）
            with state.detections_lock:
                draw_dets = state.latest_detections
            draw_detection_boxes(frame, draw_dets)

            # 编码为 JPEG 流并保存最新帧（同时刷新帧时间戳，/health 据此判断是否有新帧）
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                with state.frame_lock:
                    state.latest_frame_bytes = buffer.tobytes()
                    state.last_frame_time = time.time()

            # --- 量化指标：本轮循环耗时（含读流+推理+编码）---
            with metrics_lock:
                metrics["loop_times"].append(time.time() - t_loop_start)

        except Exception:
            logger.error("采集/检测出现异常，已安全重置，即将重试...\n%s", traceback.format_exc())
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass
            cap = None
            state.camera_connected = False
            time.sleep(min(0.3 * (2 ** min(fail_count, 4)), 5.0))
            fail_count += 1

    if cap is not None:
        cap.release()
