# -*- coding: utf-8 -*-
"""
pytest 全局配置与公共夹具。

约束（必须保证）：
  - 测试绝不真实加载 YOLO 模型（detector.get_model 默认被替换为"哨兵"，
    任何未显式提供 fake 的用例一旦触碰模型会立刻失败，而不是悄悄下载/加载权重）；
  - 测试绝不打开摄像头 / 不联网（摄像头打开函数按需在用例内 monkeypatch 为 fake）；
  - 测试路径锚定项目根，无论从哪个工作目录运行 `python -m pytest`。
"""

import os
import sys

import numpy as np
import pytest

# ------------------------------------------------------------
# 路径锚定：项目根 + tools 目录（eval_dataset.py 所在，非包模块）
# ------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ------------------------------------------------------------
# 模型隔离：默认禁止任何用例真实加载模型
# ------------------------------------------------------------
@pytest.fixture(autouse=True)
def _forbid_real_model(monkeypatch):
    """全局兜底：get_model 若未被用例显式替换为 fake，调用即报错。
    同时每个用例结束后清空模型单例，避免跨用例残留。"""
    import detector

    def _boom():
        raise RuntimeError("测试中禁止真实加载 YOLO 模型，请显式提供 fake 模型")

    monkeypatch.setattr(detector, "get_model", _boom)
    yield
    detector._model = None


# ------------------------------------------------------------
# 共享状态隔离：用例对 state 的修改在结束后还原；
# 同时覆盖新模块（zones / tracks / recorder）的模块级状态与
# zones.json 持久化文件，保证用例间完全不串。
# ------------------------------------------------------------
def _drain_recorder_queue():
    """排空录制帧队列（用例可能残留入队帧）"""
    import queue as _queue

    import recorder
    while True:
        try:
            recorder.frame_queue.get_nowait()
        except _queue.Empty:
            return


@pytest.fixture(autouse=True)
def _isolate_state():
    import recorder
    import state
    import tracks
    import zones
    from config import settings

    # zones.json 备份（/set_orientation 与 POST /zones 会真实写盘）
    zones_file_backup = None
    if os.path.isfile(settings.zones_file):
        with open(settings.zones_file, "rb") as f:
            zones_file_backup = f.read()

    snapshot = {
        "phone_orientation": dict(state.phone_orientation),
        "phone_url_override": state.phone_url_override,
        "force_reconnect": state.force_reconnect,
        "latest_detections": list(state.latest_detections),
        "camera_connected": state.camera_connected,
        "frame_mean": state.frame_mean,
        "black_warning": state.black_warning,
        # --- 新模块状态 ---
        "events": list(state.events),
        "display_size": state.display_size,
        "zones_list": zones.get_zones(),
        "zones_inside": dict(zones._inside),
        "zones_last_alert": dict(zones._last_alert),
        "zones_alarm_active": zones._alarm_active,
        "tracks_points": {k: list(v) for k, v in tracks._store.points.items()},
        "tracks_last_seen": dict(tracks._store.last_seen),
        "recorder_active": recorder.recorder_active,
        "stopped_reason": recorder.stopped_reason,
        "active_segment_name": recorder._active_segment_name,
        "duration_cache": dict(recorder._duration_cache),
    }
    _drain_recorder_queue()
    yield
    state.phone_orientation.update(snapshot["phone_orientation"])
    state.phone_url_override = snapshot["phone_url_override"]
    state.force_reconnect = snapshot["force_reconnect"]
    with state.detections_lock:
        state.latest_detections = snapshot["latest_detections"]
    state.camera_connected = snapshot["camera_connected"]
    state.frame_mean = snapshot["frame_mean"]
    state.black_warning = snapshot["black_warning"]
    # --- 新模块状态还原 ---
    with state.events_lock:
        state.events.clear()
        state.events.extend(snapshot["events"])
    state.display_size = snapshot["display_size"]
    with zones._zones_lock:
        zones._zones = snapshot["zones_list"]
    with zones._inside_lock:
        zones._inside.clear()
        zones._inside.update(snapshot["zones_inside"])
    with zones._last_alert_lock:
        zones._last_alert.clear()
        zones._last_alert.update(snapshot["zones_last_alert"])
    zones._alarm_active = snapshot["zones_alarm_active"]
    with tracks._store.lock:
        tracks._store.points.clear()
        tracks._store.last_seen.clear()
        tracks._store.last_seen.update(snapshot["tracks_last_seen"])
        for tid, pts in snapshot["tracks_points"].items():
            from collections import deque
            tracks._store.points[tid] = deque(
                pts, maxlen=max(1, settings.trail_max_points))
    recorder.recorder_active = snapshot["recorder_active"]
    recorder.stopped_reason = snapshot["stopped_reason"]
    recorder._active_segment_name = snapshot["active_segment_name"]
    recorder._duration_cache.clear()
    recorder._duration_cache.update(snapshot["duration_cache"])
    _drain_recorder_queue()
    # zones.json 还原（被用例写过的话）
    try:
        if zones_file_backup is not None:
            with open(settings.zones_file, "wb") as f:
                f.write(zones_file_backup)
        elif os.path.isfile(settings.zones_file):
            os.remove(settings.zones_file)
    except OSError:
        pass


# ------------------------------------------------------------
# fake YOLO 模型
# ------------------------------------------------------------
class FakeYOLOModel:
    """最小可用的模型替身：只提供路由/绘制用到的属性。

    `model.model.parameters()` 故意抛异常，让 /model_info 走
    device = "unknown" 的兜底分支（与无权重环境下的真实行为一致）。
    """

    def __init__(self, names=None):
        self.names = names if names is not None else {0: "person", 1: "cup", 2: "laptop"}
        self.model = self

    def parameters(self):
        raise RuntimeError("fake model has no torch parameters")

    def track(self, *args, **kwargs):
        return []


@pytest.fixture
def fake_model():
    return FakeYOLOModel()


# ------------------------------------------------------------
# fake 摄像头（VideoCapture 替身）
# ------------------------------------------------------------
class FakeVideoCapture:
    """cv2.VideoCapture 的最小替身：按序吐出合成帧，可配置失败/尺寸。"""

    def __init__(self, frames=None, width=16, height=8):
        if frames is None:
            frames = [np.full((height, width, 3), i * 10 % 255, dtype=np.uint8)
                      for i in range(8)]
        self.frames = list(frames)
        self._pos = 0
        self._opened = bool(self.frames)
        self.released = False

    def isOpened(self):
        return self._opened and not self.released

    def read(self):
        if not self.isOpened() or self._pos >= len(self.frames):
            return False, None
        frame = self.frames[self._pos]
        self._pos += 1
        return True, frame

    def grab(self):
        return self.isOpened() and self._pos < len(self.frames)

    def retrieve(self):
        return self.read()

    def set(self, prop, value):
        return True

    def get(self, prop):
        import cv2
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.frames[0].shape[1]) if self.frames else 0.0
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.frames[0].shape[0]) if self.frames else 0.0
        return 0.0

    def release(self):
        self.released = True
        self._opened = False


@pytest.fixture
def fake_video_capture():
    return FakeVideoCapture()
