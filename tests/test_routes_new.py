# -*- coding: utf-8 -*-
"""新增路由组件测试：/zones、/events、/alert/test、/recordings/*、
/set_orientation 的区域清空联动。

隔离策略（与 test_routes.py 的 client 风格一致）：
  - get_model 替换为 FakeYOLOModel（绝不真实加载权重）；
  - settings.zones_file / record_dir 重定向到 tmp_path，绝不污染项目目录；
  - /alert/test 中 zones.trigger_alarm 整体打桩，绝不真实发声；
  - 不启动任何后台线程（录制/区域检查线程均不启动）。
"""

import os

import numpy as np
import pytest

import routes
import state
import zones
from app import create_app
from config import settings

VALID_REGION = {"name": "门口", "points": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9]]}


@pytest.fixture
def client(fake_model, monkeypatch, tmp_path, _isolate_state):
    """构造应用与测试客户端；录制/区域持久化全部重定向到临时目录。
    显式依赖 _isolate_state：保证其拆卸晚于本夹具，
    zones.json 还原始终锚定真实配置文件。"""
    monkeypatch.setattr(routes, "get_model", lambda: fake_model)
    monkeypatch.setattr(settings, "zones_file", str(tmp_path / "zones.json"))
    monkeypatch.setattr(settings, "record_dir", str(tmp_path / "recordings"))
    monkeypatch.setattr(settings, "record_enabled", True)
    with zones._zones_lock:
        zones._zones = []          # 空态起步（conftest 会在用例后还原）
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


# ============================================================
# /zones：GET 空态 / POST 合法回显 / 各类非法输入 400
# ============================================================
class TestZonesRoutes:
    def test_get_zones_empty(self, client):
        resp = client.get("/zones")
        assert resp.status_code == 200
        assert resp.get_json() == {"regions": []}

    def test_post_zones_valid_multi_regions_echoed(self, client, tmp_path):
        payload = {"regions": [
            VALID_REGION,
            {"points": [[0, 0], [1, 0], [1, 1], [0, 1]]},
        ]}
        resp = client.post("/zones", json=payload)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        regions = data["regions"]
        assert len(regions) == 2
        assert regions[0]["name"] == "门口"
        assert regions[0]["id"] == "zone_1"
        assert regions[1]["name"] == "区域 2"
        # 回显后 GET 与持久化文件一致
        assert client.get("/zones").get_json()["regions"] == regions
        assert os.path.isfile(str(tmp_path / "zones.json"))

    @pytest.mark.parametrize("payload, hint", [
        ({"regions": [{"points": [[0, 0], [1, 0], [0.5, 1]]}] * 11}, "10"),
        ({"regions": [{"points": [[0, 0], [1, 1]]}]}, "3 个顶点"),
        ({"regions": [{"points": [[0, 0], [1, 0], [1.5, 0.5]]}]}, "0~1"),
        ({"regions": [{"points": [[0, 0], [1, 0], [-0.2, 0.5]]}]}, "0~1"),
        ({"regions": [{"points": [[0, 0], [1, 0], [0.1]]}]}, "二元组"),
        ({"regions": "not-a-list"}, "regions"),
        ({}, "regions"),
        # 非字符串 name 不得 500：返回 400 + 中文错误（回归：(123).strip() 崩溃）
        ({"regions": [{"name": 123,
                       "points": [[0, 0], [1, 0], [0.5, 1]]}]}, "字符串"),
        ({"regions": [{"name": {"a": 1},
                       "points": [[0, 0], [1, 0], [0.5, 1]]}]}, "字符串"),
        # 超长 name（>50 字符）返回 400
        ({"regions": [{"name": "区" * 51,
                       "points": [[0, 0], [1, 0], [0.5, 1]]}]}, "过长"),
    ])
    def test_post_zones_invalid_returns_400(self, client, payload, hint):
        resp = client.post("/zones", json=payload)
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["ok"] is False
        assert hint in data["error"]
        # 非法请求不得改动既有区域
        assert client.get("/zones").get_json()["regions"] == []


# ============================================================
# /events 与 /alert/test（winsound 经 trigger_alarm 整体打桩）
# ============================================================
class TestEventsAndAlert:
    def test_get_events_empty(self, client):
        resp = client.get("/events")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["events"] == []
        assert data["latest_id"] == 0
        assert data["alarm_active"] is False

    def test_alert_test_adds_event_without_real_sound(self, client, monkeypatch):
        alarm_calls = []
        monkeypatch.setattr(zones, "trigger_alarm",
                            lambda: alarm_calls.append(1))   # 打桩：绝不发声
        with state.events_lock:
            before = len(state.events)
        resp = client.post("/alert/test")
        assert resp.status_code == 200
        assert resp.get_json() == {"ok": True}
        assert alarm_calls == [1]
        data = client.get("/events").get_json()
        assert len(data["events"]) == before + 1
        latest = data["events"][-1]
        assert latest["zone_name"] == "测试"
        assert data["latest_id"] == latest["id"]


# ============================================================
# /recordings/*：状态五字段 / 空段列表 / 回放参数校验
# ============================================================
class TestRecordingRoutes:
    def test_recordings_status_five_fields(self, client):
        resp = client.get("/recordings/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert set(data.keys()) == {"enabled", "segment_count",
                                    "total_size_mb", "disk_free_gb",
                                    "stopped_reason"}
        assert data["enabled"] is True
        assert data["segment_count"] == 0
        assert data["total_size_mb"] == 0
        assert data["stopped_reason"] is None
        assert data["disk_free_gb"] is None or data["disk_free_gb"] > 0

    def test_recordings_clips_empty(self, client):
        resp = client.get("/recordings/clips")
        assert resp.status_code == 200
        assert resp.get_json() == {"clips": []}

    def test_clips_exclude_active_segment(self, client, tmp_path, monkeypatch):
        """正在写入的活动段不列出（打桩 _active_segment_name 验证）；
        start_ts 为真实 Unix 秒；缓存命中时不得打开 VideoCapture"""
        import datetime

        import recorder

        rec_dir = tmp_path / "recordings"
        rec_dir.mkdir(exist_ok=True)
        (rec_dir / "seg_20260831_113248.mp4").write_bytes(b"x")
        (rec_dir / "seg_20260831_113348.mp4").write_bytes(b"x")
        monkeypatch.setattr(recorder, "_active_segment_name",
                            "seg_20260831_113348.mp4")
        recorder._duration_cache["seg_20260831_113248.mp4"] = 60.0

        def _boom(path):
            raise AssertionError("缓存命中时不应打开 VideoCapture")
        monkeypatch.setattr(recorder.cv2, "VideoCapture", _boom)

        resp = client.get("/recordings/clips")
        assert resp.status_code == 200
        clips = resp.get_json()["clips"]
        assert [c["file"] for c in clips] == ["seg_20260831_113248.mp4"]
        c = clips[0]
        assert isinstance(c["start_ts"], int)
        expect = int(datetime.datetime.strptime(
            "20260831_113248", "%Y%m%d_%H%M%S").timestamp())
        assert c["start_ts"] == expect
        assert c["duration_s"] == 60.0   # 来自关段缓存，非 VideoCapture 探测
        assert c["size_mb"] >= 0

    def test_clips_fallback_opens_videocapture_on_cache_miss(
            self, client, tmp_path, monkeypatch):
        """缓存缺失时回退打开 VideoCapture 探测时长（打桩替身）"""
        import cv2 as real_cv2

        import recorder

        rec_dir = tmp_path / "recordings"
        rec_dir.mkdir(exist_ok=True)
        (rec_dir / "seg_20260831_113248.mp4").write_bytes(b"x")

        class _FakeCap:
            def __init__(self, path):
                pass

            def isOpened(self):
                return True

            def get(self, prop):
                if prop == real_cv2.CAP_PROP_FRAME_COUNT:
                    return 300.0
                if prop == real_cv2.CAP_PROP_FPS:
                    return 10.0
                return 0.0

            def release(self):
                pass

        monkeypatch.setattr(recorder.cv2, "VideoCapture", _FakeCap)
        clips = client.get("/recordings/clips").get_json()["clips"]
        assert len(clips) == 1
        assert clips[0]["duration_s"] == 30.0   # 300 帧 / 10fps

    @pytest.mark.parametrize("speed", ["3", "0", "abc", "-1"])
    def test_stream_invalid_speed_returns_400(self, client, speed):
        resp = client.get("/recordings/stream",
                          query_string={"clip": "seg_20260101_000000.mp4",
                                        "speed": speed})
        assert resp.status_code == 400
        assert "倍速" in resp.get_json()["error"]

    @pytest.mark.parametrize("clip", [
        "../secret.mp4",          # 路径穿越（basename 后仍不匹配白名单）
        "..%2F..%2Fetc%2Fpasswd", # 编码后的路径穿越
        "not_a_clip.mp4",         # 不匹配白名单格式
        "seg_evil.txt",           # 扩展名非法
        "",                       # 空名
    ])
    def test_stream_invalid_clip_returns_400(self, client, clip):
        resp = client.get("/recordings/stream",
                          query_string={"clip": clip, "speed": "1"})
        assert resp.status_code == 400
        assert "文件名" in resp.get_json()["error"]

    def test_stream_missing_clip_returns_404(self, client):
        """合法文件名但不存在 → 404"""
        resp = client.get("/recordings/stream",
                          query_string={"clip": "seg_20260101_000000.mp4",
                                        "speed": "1"})
        assert resp.status_code == 404
        assert "不存在" in resp.get_json()["error"]

    def test_stream_existing_clip_starts_mjpeg(self, client, tmp_path,
                                               monkeypatch):
        """存在的段通过校验并进入 MJPEG 响应（解码用 VideoCapture 打桩）"""
        import recorder

        rec_dir = tmp_path / "recordings"
        rec_dir.mkdir(exist_ok=True)
        (rec_dir / "seg_20260101_000000.mp4").write_bytes(b"placeholder")

        class _FakeCap:
            """按序吐出合成帧的最小 VideoCapture 替身"""
            def __init__(self, path):
                self._frames = [np.full((8, 16, 3), 128, dtype=np.uint8)]
                self._pos = 0

            def isOpened(self):
                return True

            def read(self):
                if self._pos >= len(self._frames):
                    return False, None
                f = self._frames[self._pos]
                self._pos += 1
                return True, f

            def get(self, prop):
                return 0.0

            def release(self):
                pass

        monkeypatch.setattr(recorder.cv2, "VideoCapture", _FakeCap)
        resp = client.get("/recordings/stream",
                          query_string={"clip": "seg_20260101_000000.mp4",
                                        "speed": "1"})
        assert resp.status_code == 200
        assert "multipart/x-mixed-replace" in resp.headers["Content-Type"]
        resp.close()


# ============================================================
# /set_orientation：区域清空联动
# ============================================================
class TestOrientationClearsZones:
    def test_set_orientation_clears_regions(self, client):
        # 先保存一个区域
        resp = client.post("/zones", json={"regions": [VALID_REGION]})
        assert resp.status_code == 200
        assert len(client.get("/zones").get_json()["regions"]) == 1
        # 调整方向（相对默认值变化）：响应含 regions_cleared，且区域被清空
        cur = dict(state.phone_orientation)
        new_rot = 90 if cur["rotate"] != 90 else 180
        resp = client.post("/set_orientation", json={"rotate": new_rot})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert data["regions_cleared"] is True
        assert client.get("/zones").get_json()["regions"] == []

    def test_same_orientation_does_not_clear_regions(self, client):
        """同值重设（方向未变）：不清空区域，regions_cleared 为 false
        （回归：前端镜像三态按钮循环点击不得毁掉区域配置）"""
        state.phone_orientation["rotate"] = 90
        state.phone_orientation["mirror"] = "h"
        assert client.post("/zones",
                           json={"regions": [VALID_REGION]}).status_code == 200
        resp = client.post("/set_orientation",
                           json={"rotate": 90, "mirror": "h"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["regions_cleared"] is False
        assert len(client.get("/zones").get_json()["regions"]) == 1

    def test_invalid_orientation_does_not_clear_regions(self, client):
        """非法参数被忽略（方向未变）：不得清空区域"""
        state.phone_orientation["rotate"] = 90
        state.phone_orientation["mirror"] = ""
        assert client.post("/zones",
                           json={"regions": [VALID_REGION]}).status_code == 200
        resp = client.post("/set_orientation",
                           json={"rotate": 45, "mirror": "x"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["orientation"]["rotate"] == 90   # 非法值被忽略，方向未变
        assert data["regions_cleared"] is False
        assert len(client.get("/zones").get_json()["regions"]) == 1
