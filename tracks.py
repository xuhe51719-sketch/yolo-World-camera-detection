# -*- coding: utf-8 -*-
"""
轨迹绘制模块：按跟踪 ID 记录目标底边中心点并在画面上画尾迹。

设计要点：
  - TrajectoryStore 集中管理 {track_id: 点队列 + 最近出现时间}，锁保护；
  - 点队列用 deque(maxlen=trail_max_points)，旧点自动挤出；
  - 超过 3 秒未再出现的 ID 从存储中清除（换源/目标离场后不留残迹）；
  - 颜色复用 detector.get_color(id)，与检测框配色一致；
  - settings.trail_enabled=False 时整体为空操作。

本模块导入无任何副作用。
"""

import logging
import threading
import time
from collections import deque

import cv2
import numpy as np

from config import settings

logger = logging.getLogger(__name__)

# 超过该秒数未再出现的跟踪 ID 视为离场，其轨迹被清除
_STALE_SECONDS = 3.0


class TrajectoryStore:
    """{track_id: deque((x, y))} + last_seen 时间，整体由一把锁保护"""

    def __init__(self):
        self.lock = threading.Lock()
        self.points = {}      # track_id -> deque((x, y), maxlen)
        self.last_seen = {}   # track_id -> monotonic 秒


# 模块级单例（与项目其他模块的"模块级状态 + 专用锁"风格一致）
_store = TrajectoryStore()


def update_and_draw_trajectories(frame, detections):
    """更新轨迹并就地绘制尾迹。

    对每个有跟踪 ID 的目标取框底边中心 ((x1+x2)/2, y2) 作为轨迹点；
    与上一点完全相同则去重不追加。随后绘制所有存活轨迹。
    """
    if not settings.trail_enabled:
        return

    from detector import get_color   # 延迟导入，避免与 detector 形成导入环

    now = time.monotonic()
    with _store.lock:
        # --- 更新：追加新点（同点去重）---
        for det in detections:
            track_id = det.get("id")
            if track_id is None:
                continue
            x1, _y1, x2, y2 = det["bbox"]
            pt = (int((x1 + x2) / 2), int(y2))   # 框底边中心（脚底位置，轨迹更直观）
            dq = _store.points.get(track_id)
            if dq is None:
                dq = deque(maxlen=max(1, settings.trail_max_points))
                _store.points[track_id] = dq
            if not dq or dq[-1] != pt:
                dq.append(pt)
            _store.last_seen[track_id] = now

        # --- 清除：超过 3 秒未见的 ID（含其点队列）---
        stale = [tid for tid, ts in _store.last_seen.items()
                 if now - ts > _STALE_SECONDS]
        for tid in stale:
            _store.points.pop(tid, None)
            _store.last_seen.pop(tid, None)

        # --- 绘制：每条轨迹一条折线，颜色与检测框一致 ---
        for track_id, dq in _store.points.items():
            if len(dq) < 2:
                continue
            try:
                pts = np.array(dq, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, get_color(track_id), 2)
            except Exception:
                pass   # 绘制失败不影响检测主流程
