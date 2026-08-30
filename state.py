# -*- coding: utf-8 -*-
"""
共享运行时状态：采集线程与 Flask 路由之间的全部可变状态集中于此。
本模块导入无任何副作用。

camera_info 采用"原子替换 + 拷贝读取"策略（修复跨线程竞争）：
写入侧新建 dict 整体赋值，读取侧在锁内拷贝一份，
保证 /source、/model_info 序列化迭代时不会抛
"dictionary changed size during iteration"。
"""

import threading
import time

from config import settings

# ============================================================
# 摄像头信息（原子替换保护）
# ============================================================
_camera_info_lock = threading.Lock()
_camera_info = {"index": None, "backend_name": "unknown", "width": 0, "height": 0}


def set_camera_info(**fields):
    """写入侧：新建 dict 整体替换，避免就地修改引发迭代竞争"""
    global _camera_info
    with _camera_info_lock:
        new_info = dict(_camera_info)
        new_info.update(fields)
        _camera_info = new_info


def get_camera_info():
    """读取侧：锁内拷贝，调用方可安全迭代/序列化"""
    with _camera_info_lock:
        return dict(_camera_info)


# ============================================================
# 帧与检测结果
# ============================================================
latest_frame_bytes = None
frame_lock = threading.Lock()

latest_detections = []
detections_lock = threading.Lock()

# ============================================================
# 管线运行状态
# ============================================================
frame_count = 0
running = True
last_frame_time = 0.0          # 最近一次成功写入新帧的时间戳（/health 判新鲜度用）

phone_url_override = None    # 运行时通过网页设置的手机流地址
force_reconnect = False      # 触发切换图像源

camera_connected = False         # 当前图像源是否已成功连接
frame_mean = 0.0                 # 当前帧平均亮度（0~255）
black_warning = False            # 画面是否持续为黑
last_nonblack_time = time.time() # 最近一次出现非黑帧的时间

# 画面方向（运行时可调）：旋转 0/90/180/270（顺时针），镜像 ""/"h"/"v"
phone_orientation = {"rotate": settings.phone_rotate, "mirror": settings.phone_mirror}
