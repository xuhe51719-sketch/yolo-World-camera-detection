# -*- coding: utf-8 -*-
"""
集中配置模块：全项目所有环境变量读取与默认常量都收敛在这里。
环境变量名与原实现保持完全兼容（YOLO_MODEL / YOLO_IMGSZ / YOLO_HALF /
PHONE_ROTATE / PHONE_MIRROR / PHONE_CAMERA_URL / FLASK_HOST / FLASK_PORT /
DETECTION_INTERVAL / LOG_LEVEL / RECORD_ENABLED / RECORD_DIR / RECORD_FPS /
RECORD_SEGMENT_S / RECORD_RETENTION_MIN / ALERT_COOLDOWN_S / ALERT_SOUND_S /
ZONES_FILE）。

所有相对路径以项目根（BASE_DIR）锚定，避免依赖运行时工作目录。
本模块导入时无任何模型加载 / 摄像头连接副作用。
"""

import os
from dataclasses import dataclass

# 项目根目录：一切权重 / 数据 / 输出相对路径的锚点
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _resolve_path(path):
    """相对路径统一锚定到项目根；绝对路径原样保留"""
    return path if os.path.isabs(path) else os.path.join(BASE_DIR, path)


@dataclass
class AppConfig:
    """运行时配置（环境变量在 load_config() 中一次性读取）"""
    # --- 模型 ---
    # 模型选择：
    #   yolov8x-worldv2.pt（默认）— 开放词汇检测，可识别 DETECTION_CLASSES 指定的任意类别；
    #   yolov8m.pt / yolov8n.pt — 传统固定 80 类模型（速度快，CPU 环境应急用）。
    # 可用环境变量 YOLO_MODEL 覆盖。
    model_path: str
    confidence_threshold: float   # 置信度阈值（开放词汇模型适当调低可减少漏检）
    detect_imgsz: int             # 检测输入分辨率（640 兼顾精度与推理速度）
    use_half: bool                # FP16 半精度推理（CUDA 上默认开启，YOLO_HALF=0 回退 FP32）
    is_world_model: bool          # 是否开放词汇（YOLO-World）模型

    # --- 摄像头 / 管线 ---
    frame_width: int
    frame_height: int
    target_fps: int
    detection_interval: int       # 每 N 帧执行一次检测（非检测帧复用上次结果）
    frame_max_age: float          # 帧最大允许年龄（秒）：超过则等待新帧
    fallback_max_age: float       # 兜底帧返回上限（秒）：超过按断流重连
    black_mean_threshold: float   # 黑帧判定：整帧平均亮度低于该值即视为黑屏

    # --- 手机流 ---
    phone_camera_url: str         # 手机 IP 摄像头推流地址（留空用本地摄像头）
    phone_fallback_to_local: bool # 手机流失败时是否回退本地摄像头（建议 False）
    phone_rotate: int             # 画面旋转角度 0/90/180/270（顺时针）
    phone_mirror: str             # 镜像 ""（不镜像）/ "h"（左右）/ "v"（上下）

    # --- Web 服务 ---
    flask_host: str               # 默认 127.0.0.1（仅本机）；显式设 0.0.0.0 才开放局域网
    flask_port: int

    # --- 日志 ---
    log_level: str                # 日志级别（环境变量 LOG_LEVEL，默认 INFO）

    # --- 滚动录制 ---
    record_enabled: bool          # 是否启用滚动录制（RECORD_ENABLED=0 关闭）
    record_dir: str               # 录制段存放目录（默认项目根 recordings/）
    record_fps: int               # 录制节流帧率（写入段的最大帧率）
    record_segment_s: int         # 每段时长（秒），到点关段开新段
    record_retention_min: int     # 滚动保留时长（分钟），超出的最旧段自动删除（最多约 1 小时）

    # --- 区域入侵报警 ---
    alert_cooldown_s: int         # 同一 (区域, 类别) 报警冷却（秒），避免持续滞留重复触发
    alert_sound_s: int            # 报警音播放时长（秒）
    zones_file: str               # 区域定义持久化文件（默认项目根 zones.json）

    # --- 锚定到项目根的路径 ---
    dataset_dir: str
    dataset_yaml: str
    cam_test_dir: str
    benchmark_csv: str


def load_config():
    """从环境变量构建配置（保留原环境变量名兼容）"""
    import torch  # 仅探测 CUDA 可用性，不加载模型、不占显存

    model_path = os.environ.get("YOLO_MODEL", "yolov8x-worldv2.pt")

    return AppConfig(
        model_path=_resolve_path(model_path),
        confidence_threshold=0.2,
        # 下限收敛：YOLO_IMGSZ=0/负值会让推理每帧抛异常，陷入异常-重连风暴（32 为 YOLO 可接受的最小输入）
        detect_imgsz=max(32, _env_int("YOLO_IMGSZ", 640)),
        # FP16 半精度：CUDA 上默认开启（RTX 40 系可提速 1.5~2 倍，精度损失可忽略）
        use_half=torch.cuda.is_available() and os.environ.get("YOLO_HALF", "1") == "1",
        is_world_model="world" in model_path.lower(),
        frame_width=1280,
        frame_height=720,
        target_fps=30,
        # 下限收敛：0/负值会使 frame_count % interval 永远为 0（每帧都推理）甚至除零异常，最低 1 帧一次检测
        detection_interval=max(1, _env_int("DETECTION_INTERVAL", 2)),
        frame_max_age=0.5,
        fallback_max_age=1.0,
        black_mean_threshold=8.0,
        phone_camera_url=os.environ.get("PHONE_CAMERA_URL", ""),
        phone_fallback_to_local=False,
        phone_rotate=_env_int("PHONE_ROTATE", 0),
        phone_mirror=os.environ.get("PHONE_MIRROR", ""),
        flask_host=os.environ.get("FLASK_HOST", "127.0.0.1"),
        flask_port=_env_int("FLASK_PORT", 5000),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
        # --- 滚动录制（默认开启；段时长/保留时长配合实现"最多 1 小时滚动"）---
        record_enabled=os.environ.get("RECORD_ENABLED", "1") != "0",
        record_dir=_resolve_path(os.environ.get("RECORD_DIR", "recordings")),
        record_fps=_env_int("RECORD_FPS", 15),
        record_segment_s=_env_int("RECORD_SEGMENT_S", 60),
        record_retention_min=_env_int("RECORD_RETENTION_MIN", 60),
        # --- 区域入侵报警 ---
        alert_cooldown_s=_env_int("ALERT_COOLDOWN_S", 30),
        alert_sound_s=_env_int("ALERT_SOUND_S", 10),
        zones_file=_resolve_path(os.environ.get("ZONES_FILE", "zones.json")),
        dataset_dir=os.path.join(BASE_DIR, "dataset"),
        dataset_yaml=os.path.join(BASE_DIR, "dataset", "data.yaml"),
        cam_test_dir=os.path.join(BASE_DIR, "cam_test"),
        benchmark_csv=os.path.join(BASE_DIR, "benchmark_results.csv"),
    )


# 全局唯一配置实例；main() 中的命令行参数可对其字段做覆盖（命令行优先于环境变量）
settings = load_config()
