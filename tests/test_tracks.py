# -*- coding: utf-8 -*-
"""tracks.py 测试：TrajectoryStore 的追加/去重/截断/过期清除 + 绘制打桩。

隔离策略：
  - 时间通过 monkeypatch 替换 tracks.time.monotonic，用可控时钟测试过期清除；
  - detector.get_color 与 cv2.polylines 打桩，避免真实绘制依赖；
  - settings.trail_enabled 用 monkeypatch 还原。
"""

import numpy as np
import pytest

import tracks
from config import settings


@pytest.fixture
def clock(monkeypatch):
    """可控单调时钟：clock["v"] 前进即时间流逝"""
    t = {"v": 1000.0}
    monkeypatch.setattr(tracks.time, "monotonic", lambda: t["v"])
    return t


@pytest.fixture(autouse=True)
def _enable_trail(monkeypatch):
    monkeypatch.setattr(settings, "trail_enabled", True)


def _det(track_id, x1, y1, x2, y2):
    return {"id": track_id, "class": "person", "confidence": 0.9,
            "bbox": [x1, y1, x2, y2]}


def _points_of(track_id):
    return list(tracks._store.points[track_id])


@pytest.fixture(autouse=True)
def _no_draw(monkeypatch):
    """默认禁真实绘制：get_color 与 polylines 均打桩"""
    monkeypatch.setattr("detector.get_color", lambda tid: (0, 255, 0))
    monkeypatch.setattr(tracks.cv2, "polylines", lambda *a, **k: None)


class TestTrajectoryAppend:
    def test_append_bottom_center_point(self, clock):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        tracks.update_and_draw_trajectories(frame, [_det(1, 10, 20, 30, 50)])
        # 底边中心：((10+30)/2, 50) = (20, 50)
        assert _points_of(1) == [(20, 50)]
        assert 1 in tracks._store.last_seen

    def test_duplicate_point_not_appended(self, clock):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        for _ in range(5):
            tracks.update_and_draw_trajectories(frame, [_det(1, 10, 20, 30, 50)])
        assert _points_of(1) == [(20, 50)]   # 同点去重，仅一条

    def test_distinct_points_appended_in_order(self, clock):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        tracks.update_and_draw_trajectories(frame, [_det(1, 0, 0, 10, 10)])
        tracks.update_and_draw_trajectories(frame, [_det(1, 10, 0, 20, 20)])
        assert _points_of(1) == [(5, 10), (15, 20)]

    def test_detection_without_id_ignored(self, clock):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        tracks.update_and_draw_trajectories(
            frame, [{"class": "person", "bbox": [0, 0, 10, 10]}])
        assert tracks._store.points == {}

    def test_truncates_to_trail_max_points(self, clock, monkeypatch):
        monkeypatch.setattr(settings, "trail_max_points", 3)
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        for i in range(6):
            tracks.update_and_draw_trajectories(
                frame, [_det(1, i * 10, 0, i * 10 + 10, 10)])
        pts = _points_of(1)
        assert len(pts) == 3                      # 超过上限被截断
        assert pts == [(35, 10), (45, 10), (55, 10)]   # 只保留最新 3 点


class TestTrajectoryStaleCleanup:
    def test_stale_id_removed_after_3s(self, clock):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        tracks.update_and_draw_trajectories(frame, [_det(1, 0, 0, 10, 10)])
        tracks.update_and_draw_trajectories(frame, [_det(2, 20, 0, 30, 10)])
        # ID=1 离场 4 秒；ID=2 持续出现
        clock["v"] += 4.0
        tracks.update_and_draw_trajectories(frame, [_det(2, 22, 0, 32, 12)])
        assert 1 not in tracks._store.points
        assert 1 not in tracks._store.last_seen
        assert 2 in tracks._store.points

    def test_not_removed_within_3s(self, clock):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        tracks.update_and_draw_trajectories(frame, [_det(1, 0, 0, 10, 10)])
        clock["v"] += 2.0
        # 用另一个目标触发一次清理扫描
        tracks.update_and_draw_trajectories(frame, [_det(2, 20, 0, 30, 10)])
        assert 1 in tracks._store.points


class TestTrajectoryToggleAndDraw:
    def test_disabled_is_noop(self, clock, monkeypatch):
        monkeypatch.setattr(settings, "trail_enabled", False)
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        tracks.update_and_draw_trajectories(frame, [_det(1, 0, 0, 10, 10)])
        assert tracks._store.points == {}

    def test_polyline_drawn_when_enough_points(self, clock, monkeypatch):
        drawn = []
        monkeypatch.setattr(
            tracks.cv2, "polylines",
            lambda img, pts, closed, color, thickness: drawn.append(
                (len(pts[0]), color, thickness)))
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        tracks.update_and_draw_trajectories(frame, [_det(1, 0, 0, 10, 10)])
        assert drawn == []                        # 单点不画
        tracks.update_and_draw_trajectories(frame, [_det(1, 10, 0, 20, 20)])
        assert drawn == [(2, (0, 255, 0), 2)]     # 两点起画折线

    def test_get_model_never_called_for_trails(self, clock):
        """轨迹模块只依赖 detector.get_color，绝不应触碰模型加载"""
        import detector
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        tracks.update_and_draw_trajectories(frame, [_det(1, 0, 0, 10, 10)])
        tracks.update_and_draw_trajectories(frame, [_det(1, 5, 0, 15, 10)])
        assert detector._model is None
