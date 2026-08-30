# -*- coding: utf-8 -*-
"""
量化指标采集（性能评估与调参用，通过 /metrics 接口查看）。
本模块导入无任何副作用。
"""

import threading
import time
from collections import deque, Counter

metrics_lock = threading.Lock()
metrics = {
    "start_time": time.time(),
    "loop_times": deque(maxlen=120),   # 每轮采集循环耗时（秒），滚动窗口
    "infer_times": deque(maxlen=120),  # 每次模型推理耗时（秒），滚动窗口
    "id_switches": 0,                  # 累计跟踪 ID 切换次数（越低越稳定）
    "detections_total": 0,             # 累计检出框数
    "class_counts": Counter(),         # 各类别累计检出次数
    "prev_tracks": [],                 # 上一帧跟踪结果，用于判定 ID 切换
}


def reset_source_metrics():
    """切换图像源时清零与"当前源"绑定的累计指标。

    跟踪统计（prev_tracks / id_switches）与检出累计（detections_total /
    class_counts）都依附于具体图像源：换源后新旧画面的统计混在一起
    没有意义，且旧源的 prev_tracks 还会对新源误判 ID 切换。
    耗时窗口（loop_times / infer_times）与运行时长保持不变。
    """
    with metrics_lock:
        metrics["prev_tracks"] = []
        metrics["id_switches"] = 0
        metrics["detections_total"] = 0
        metrics["class_counts"] = Counter()
