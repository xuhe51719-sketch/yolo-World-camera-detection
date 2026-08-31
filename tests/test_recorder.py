# -*- coding: utf-8 -*-
"""recorder.py 测试：清理策略、入队丢旧帧、录制线程（打桩 VideoWriter + 可控时钟）。

隔离策略：
  - settings.record_dir 等录制配置用 monkeypatch 指向 tmp_path，绝不污染项目目录；
  - cv2.VideoWriter 打桩为 FakeWriter（不真实编码），段轮转/重开逻辑可离线验证；
  - 录制线程内的时间（monotonic/strftime）全部替换为可控时钟，无真实等待；
  - state.running 用例内保证恢复，绝不影响其他测试。
"""

import os
import queue
import threading
import time as real_time

import numpy as np
import pytest

import recorder
import state
from config import settings


# ============================================================
# 公共夹具
# ============================================================
@pytest.fixture
def rec_dirs(tmp_path, monkeypatch, _isolate_state):
    """录制配置重定向到临时目录，并把节流参数调到测试友好值。
    显式依赖 _isolate_state：保证其拆卸晚于本夹具，
    用例断言期间录制配置补丁仍有效。"""
    rec = tmp_path / "recordings"
    rec.mkdir()
    monkeypatch.setattr(settings, "record_dir", str(rec))
    monkeypatch.setattr(settings, "record_enabled", True)
    monkeypatch.setattr(settings, "record_fps", 1000)        # 不节流
    monkeypatch.setattr(settings, "record_segment_s", 1)
    monkeypatch.setattr(settings, "record_retention_min", 1)  # 上限 60 段
    return rec


class FakeWriter:
    """cv2.VideoWriter 替身：记录开关段与写帧，不真实编码"""
    instances = []
    fail_open = False       # True 时 isOpened() 恒 False（模拟编码器不可用）

    def __init__(self, path, fourcc, fps, size):
        self.path = path
        self.fourcc = fourcc
        self.fps = fps
        self.size = size
        self.frames = []
        self.released = False
        # 与真实 VideoWriter 行为一致：打开即在磁盘创建段文件（占位）
        try:
            with open(path, "ab"):
                pass
        except OSError:
            pass
        FakeWriter.instances.append(self)

    def isOpened(self):
        return not FakeWriter.fail_open

    def write(self, frame):
        self.frames.append(frame.shape[:2])
        return True

    def release(self):
        self.released = True


@pytest.fixture
def fake_writer(monkeypatch):
    FakeWriter.instances = []
    FakeWriter.fail_open = False
    monkeypatch.setattr(recorder.cv2, "VideoWriter", FakeWriter)
    return FakeWriter


def _make_seg(dirpath, stamp, content=b"x"):
    """在目录里伪造一个段文件"""
    p = dirpath / f"seg_{stamp}.mp4"
    p.write_bytes(content)
    return p.name


# ============================================================
# 段文件管理与清理策略（纯逻辑）
# ============================================================
class TestSegmentManagement:
    def test_list_segments_sorted_and_filtered(self, rec_dirs):
        _make_seg(rec_dirs, "20260831_120002")
        _make_seg(rec_dirs, "20260831_120000")
        _make_seg(rec_dirs, "20260831_120001")
        (rec_dirs / "not_a_segment.mp4").write_bytes(b"x")   # 不匹配白名单
        (rec_dirs / "seg_bad.mp4").write_bytes(b"x")
        assert recorder._list_segments() == [
            "seg_20260831_120000.mp4",
            "seg_20260831_120001.mp4",
            "seg_20260831_120002.mp4",
        ]

    def test_list_segments_missing_dir_returns_empty(self, rec_dirs, monkeypatch):
        monkeypatch.setattr(settings, "record_dir",
                            str(rec_dirs / "does_not_exist"))
        assert recorder._list_segments() == []

    def test_cleanup_deletes_oldest_over_retention(self, rec_dirs, monkeypatch):
        # 保留上限 = 1 分钟 / 1 秒每段 = 60 段；monkeypatch 收紧到 2 段
        monkeypatch.setattr(settings, "record_retention_min", 1)
        monkeypatch.setattr(settings, "record_segment_s", 30)   # 上限 2 段
        for i in range(5):
            _make_seg(rec_dirs, f"20260831_12000{i}")
        recorder._cleanup_old_segments()
        assert recorder._list_segments() == [
            "seg_20260831_120003.mp4", "seg_20260831_120004.mp4"]

    def test_cleanup_cap_at_least_one(self, rec_dirs, monkeypatch):
        """计算出的上限被收敛到至少 1 段，不会全删"""
        monkeypatch.setattr(settings, "record_retention_min", 0)
        _make_seg(rec_dirs, "20260831_120000")
        _make_seg(rec_dirs, "20260831_120001")
        recorder._cleanup_old_segments()
        assert recorder._list_segments() == ["seg_20260831_120001.mp4"]

    def test_disk_free_missing_dir_returns_none(self, rec_dirs, monkeypatch):
        monkeypatch.setattr(settings, "record_dir",
                            str(rec_dirs / "does_not_exist"))
        assert recorder._disk_free() is None


# ============================================================
# enqueue：旁路不阻塞，队列满丢最旧帧
# ============================================================
class TestEnqueue:
    def test_enqueue_skipped_when_disabled(self, monkeypatch):
        monkeypatch.setattr(settings, "record_enabled", False)
        recorder.recorder_active = True
        recorder.enqueue(np.zeros((4, 4, 3), dtype=np.uint8))
        assert recorder.frame_queue.qsize() == 0

    def test_enqueue_skipped_when_recorder_inactive(self, monkeypatch):
        monkeypatch.setattr(settings, "record_enabled", True)
        recorder.recorder_active = False
        recorder.enqueue(np.zeros((4, 4, 3), dtype=np.uint8))
        assert recorder.frame_queue.qsize() == 0

    def test_enqueue_full_queue_drops_oldest_without_blocking(self, monkeypatch):
        monkeypatch.setattr(settings, "record_enabled", True)
        recorder.recorder_active = True
        f1 = np.full((4, 4, 3), 1, dtype=np.uint8)
        f2 = np.full((4, 4, 3), 2, dtype=np.uint8)
        f3 = np.full((4, 4, 3), 3, dtype=np.uint8)
        recorder.enqueue(f1)
        recorder.enqueue(f2)
        assert recorder.frame_queue.qsize() == 2     # 队列已满（maxsize=2）
        recorder.enqueue(f3)                          # 必须不阻塞
        assert recorder.frame_queue.qsize() == 2
        kept = []
        while True:
            try:
                kept.append(recorder.frame_queue.get_nowait()[0, 0, 0])
            except queue.Empty:
                break
        assert kept == [2, 3]                       # 最旧帧 1 被丢弃


# ============================================================
# _open_writer：三级降级
# ============================================================
class TestOpenWriter:
    def test_fallback_to_mjpg_when_mp4v_fails(self, rec_dirs, fake_writer,
                                               monkeypatch):
        def _is_opened(self):
            return not self.path.endswith(".mp4")   # mp4v 失败，MJPG 可用

        monkeypatch.setattr(FakeWriter, "isOpened", _is_opened)
        w, path = recorder._open_writer((64, 64))
        assert w is not None
        assert path.endswith(".avi")
        # mp4v 的失败写入器被 release
        assert FakeWriter.instances[0].released is True

    def test_all_encoders_fail_returns_none(self, rec_dirs, fake_writer):
        FakeWriter.fail_open = True
        w, path = recorder._open_writer((64, 64))
        assert w is None and path is None


# ============================================================
# recorder_loop：段轮转 / 尺寸变化重开 / 编码器全失败停录
# （打桩 VideoWriter + 可控时钟，无真实编码与真实等待）
# ============================================================
@pytest.fixture
def fake_loop_clock(monkeypatch):
    """录制线程内所有时间调用走可控时钟：
    auto_tick=True 时 monotonic 每次调用 +0.05（段时长 1s ≈ 20 次调用后轮转）；
    strftime 每次调用返回递增时间戳，保证轮转后段文件名不同。"""
    t = {"v": 1000.0, "auto_tick": True, "stamp": 0}

    def _tick():
        if t["auto_tick"]:
            t["v"] += 0.05
        return t["v"]

    def _stamp(fmt):
        s = f"20260101_00{t['stamp']:04d}"
        t["stamp"] += 1
        return s

    monkeypatch.setattr(recorder.time, "monotonic", _tick)
    monkeypatch.setattr(recorder.time, "strftime", _stamp)
    return t


def _run_loop_briefly(feed_frames, idle_wait=0.6):
    """在后台线程跑 recorder_loop，主线程喂帧后停录并等待退出"""
    th = threading.Thread(target=recorder.recorder_loop, daemon=True)
    th.start()
    try:
        for frame in feed_frames:
            recorder.enqueue(frame)
            real_time.sleep(0.02)   # 让录制线程有机会消费
        real_time.sleep(idle_wait)
    finally:
        state.running = False       # 通知录制线程退出
        th.join(timeout=5)
        state.running = True        # 恢复（其他用例依赖）
    assert not th.is_alive(), "录制线程未能退出"


class TestRecorderLoop:
    def test_segment_rotation_and_filename_format(
            self, rec_dirs, fake_writer, fake_loop_clock, monkeypatch):
        frames = [np.full((64, 64, 3), i % 255, dtype=np.uint8)
                  for i in range(30)]
        _run_loop_briefly(frames)
        names = recorder._list_segments()
        assert len(names) >= 2, "段时长到点后应关段开新段"
        import re as _re
        for n in names:
            assert _re.fullmatch(r"seg_\d{8}_\d{6}\.mp4", n), n
        assert names == sorted(names)   # 文件名排序即时间排序
        # 所有被打开的写入器都已 release（关段即释放）
        assert len(fake_writer.instances) >= 2
        assert all(w.released for w in fake_writer.instances)
        assert recorder.recorder_active is False

    def test_frame_size_change_reopens_writer(
            self, rec_dirs, fake_writer, fake_loop_clock, monkeypatch):
        monkeypatch.setattr(settings, "record_segment_s", 3600)  # 轮转不介入
        fake_loop_clock["auto_tick"] = False   # 冻结时钟，排除轮转干扰
        frames = [np.zeros((64, 64, 3), dtype=np.uint8),
                  np.zeros((64, 64, 3), dtype=np.uint8),
                  np.zeros((32, 32, 3), dtype=np.uint8)]   # 尺寸变化
        _run_loop_briefly(frames)
        assert len(fake_writer.instances) >= 2
        sizes = [w.size for w in fake_writer.instances]
        assert sizes[0] == (64, 64)
        assert (32, 32) in sizes
        assert fake_writer.instances[0].released is True

    def test_all_encoders_fail_stops_with_reason(
            self, rec_dirs, fake_writer, monkeypatch):
        FakeWriter.fail_open = True
        _run_loop_briefly([np.zeros((64, 64, 3), dtype=np.uint8)],
                          idle_wait=0.3)
        assert recorder.recorder_active is False
        assert recorder.stopped_reason is not None
        assert "编码器" in recorder.stopped_reason
