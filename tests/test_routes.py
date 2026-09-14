# -*- coding: utf-8 -*-
"""routes.py 组件测试：create_app() + Flask test client。

隔离策略：
  - get_model 被替换为 FakeYOLOModel（绝不真实加载权重）；
  - 不启动 detection_loop 后台线程，不打开摄像头，不联网；
  - metrics 窗口数据在每个用例前后保存/还原，保证可重复运行。
"""

import os
import time

import pytest

import routes
import state
from app import create_app
from config import settings
from metrics import metrics, metrics_lock


@pytest.fixture(autouse=True)
def _isolate_metrics():
    """还原 /metrics 依赖的滚动窗口与累计值，避免用例互相污染"""
    with metrics_lock:
        snapshot = {
            "loop_times": list(metrics["loop_times"]),
            "infer_times": list(metrics["infer_times"]),
            "id_switches": metrics["id_switches"],
            "detections_total": metrics["detections_total"],
        }
        metrics["loop_times"].clear()
        metrics["infer_times"].clear()
        metrics["id_switches"] = 0
        metrics["detections_total"] = 0
    yield
    with metrics_lock:
        metrics["loop_times"].clear()
        metrics["loop_times"].extend(snapshot["loop_times"])
        metrics["infer_times"].clear()
        metrics["infer_times"].extend(snapshot["infer_times"])
        metrics["id_switches"] = snapshot["id_switches"]
        metrics["detections_total"] = snapshot["detections_total"]


@pytest.fixture
def client(fake_model, monkeypatch):
    """构造应用与测试客户端；routes 模块持有的是 import 时绑定的
    get_model 引用，必须打补丁到 routes 命名空间"""
    monkeypatch.setattr(routes, "get_model", lambda: fake_model)
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


# ============================================================
# 基础页面与健康检查
# ============================================================
class TestIndexAndHealth:
    def test_index_renders_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["Content-Type"]

    def test_health_no_frame_yet(self, client):
        """无帧写入时：frame_fresh=False，frame_age_s=None"""
        state.last_frame_time = 0.0
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert data["running"] is True
        assert data["frame_fresh"] is False
        assert data["frame_age_s"] is None
        assert data["uptime_s"] >= 0

    def test_health_fresh_frame(self, client):
        """刚写入新帧：frame_fresh=True 且帧龄很小"""
        state.last_frame_time = time.time()
        data = client.get("/health").get_json()
        assert data["frame_fresh"] is True
        assert data["frame_age_s"] is not None
        assert data["frame_age_s"] <= 3.0

    def test_health_stale_frame(self, client):
        """帧龄超过 3 秒：判为不新鲜"""
        state.last_frame_time = time.time() - 10.0
        data = client.get("/health").get_json()
        assert data["frame_fresh"] is False
        assert data["frame_age_s"] > 3.0


# ============================================================
# /detections
# ============================================================
class TestDetections:
    def test_empty_detections(self, client):
        with state.detections_lock:
            state.latest_detections = []
        resp = client.get("/detections")
        assert resp.status_code == 200
        assert resp.get_json() == []

    def test_detections_returns_latest(self, client):
        det = {"class": "cup", "confidence": 0.87, "id": 3,
               "bbox": [10.0, 20.0, 100.0, 200.0]}
        with state.detections_lock:
            state.latest_detections = [det]
        resp = client.get("/detections")
        assert resp.status_code == 200
        assert resp.get_json() == [det]


# ============================================================
# /model_info（使用 fake 模型，绝不加载权重）
# ============================================================
class TestModelInfo:
    def test_model_info_with_fake(self, client, fake_model):
        resp = client.get("/model_info")
        assert resp.status_code == 200
        data = resp.get_json()
        # 对外展示文件名而非绝对路径（内部加载仍用完整路径）
        assert data["model"] == os.path.basename(settings.model_path)
        assert data["confidence_threshold"] == settings.confidence_threshold
        assert data["imgsz"] == settings.detect_imgsz
        assert data["total_classes"] == len(fake_model.names)
        assert data["class_names"] == {str(k): v for k, v in fake_model.names.items()}
        # fake 模型的 parameters() 抛异常 -> device 走兜底分支
        assert data["device"] == "unknown"
        assert isinstance(data["camera"], dict)

    def test_model_info_never_touches_real_model(self, client, monkeypatch):
        """若路由误触真实加载路径，哨兵 get_model 抛出的异常会穿透视图：
        TESTING 模式下 Flask 不吞异常，直接冒泡即证明没有被静默回退掩盖"""
        def _boom():
            raise RuntimeError("测试中禁止真实加载 YOLO 模型")
        monkeypatch.setattr(routes, "get_model", _boom)
        with pytest.raises(RuntimeError, match="禁止真实加载"):
            client.get("/model_info")

    def test_model_info_includes_available_models(self, client, monkeypatch):
        """/model_info 携带可切换模型列表（前端下拉框数据源）"""
        fake_list = [{"name": "yolov8n.pt", "is_world": False, "active": False},
                     {"name": "yolov8x-worldv2.pt", "is_world": True, "active": True}]
        monkeypatch.setattr(routes, "list_available_models", lambda *a, **k: fake_list)
        data = client.get("/model_info").get_json()
        assert data["available_models"] == fake_list


# ============================================================
# /source
# ============================================================
class TestSource:
    def test_source_fields(self, client):
        state.phone_url_override = None
        resp = client.get("/source")
        assert resp.status_code == 200
        data = resp.get_json()
        for key in ("configured_url", "active_url", "resolved_url",
                    "camera", "connected", "frame_mean",
                    "black_warning", "orientation"):
            assert key in data
        assert data["active_url"] is None
        assert data["resolved_url"] == settings.phone_camera_url
        assert data["orientation"] == {
            "rotate": state.phone_orientation["rotate"],
            "mirror": state.phone_orientation["mirror"],
        }

    def test_source_with_override(self, client):
        state.phone_url_override = "http://10.0.0.9:8080/video"
        data = client.get("/source").get_json()
        assert data["active_url"] == "http://10.0.0.9:8080/video"
        assert data["resolved_url"] == "http://10.0.0.9:8080/video"


# ============================================================
# /metrics（含空数据边界）
# ============================================================
class TestMetrics:
    def test_metrics_empty_data(self, client):
        """无任何采集数据时所有数值字段安全返回 0，不抛异常"""
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["pipeline_fps"] == 0
        assert data["loop_ms_avg"] == 0
        assert data["infer_ms_avg"] == 0
        assert data["infer_ms_p95"] == 0
        assert data["infer_fps"] == 0
        assert data["id_switches_total"] == 0
        assert data["detections_total"] == 0
        assert data["top_classes"] == []
        assert data["uptime_s"] >= 0
        assert data["model"] == os.path.basename(settings.model_path)

    def test_metrics_with_data(self, client):
        with metrics_lock:
            metrics["loop_times"].extend([0.04, 0.04])   # 平均 40ms -> 25 FPS
            metrics["infer_times"].extend([0.03, 0.03])
            metrics["detections_total"] = 5
        data = client.get("/metrics").get_json()
        assert data["loop_ms_avg"] == 40.0
        assert data["pipeline_fps"] == 25.0
        assert data["infer_ms_avg"] == 30.0
        assert data["infer_fps"] == pytest.approx(25.0 / settings.detection_interval)
        assert data["detections_total"] == 5

    def test_metrics_config_fields(self, client):
        data = client.get("/metrics").get_json()
        assert data["detect_interval"] == settings.detection_interval
        assert data["conf"] == settings.confidence_threshold
        assert data["half"] == settings.use_half
        assert data["imgsz"] == settings.detect_imgsz


# ============================================================
# /set_orientation：合法 / 非法 / 缺参
# ============================================================
class TestSetOrientation:
    @pytest.mark.parametrize("rot", [0, 90, 180, 270])
    def test_valid_rotate(self, client, rot):
        resp = client.post("/set_orientation", json={"rotate": rot})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert data["orientation"]["rotate"] == rot
        assert state.phone_orientation["rotate"] == rot

    def test_rotate_as_numeric_string(self, client):
        """数字字符串可被 int() 解析，视为合法"""
        data = client.post("/set_orientation", json={"rotate": "180"}).get_json()
        assert data["orientation"]["rotate"] == 180

    def test_invalid_rotate_ignored(self, client):
        """45° 不在白名单：保持原值"""
        state.phone_orientation["rotate"] = 90
        data = client.post("/set_orientation", json={"rotate": 45}).get_json()
        assert data["ok"] is True
        assert data["orientation"]["rotate"] == 90

    def test_non_numeric_rotate_falls_back_to_zero(self, client):
        """无法 int() 的值按 0 处理（实现的既定行为）"""
        state.phone_orientation["rotate"] = 90
        data = client.post("/set_orientation", json={"rotate": "abc"}).get_json()
        assert data["orientation"]["rotate"] == 0

    @pytest.mark.parametrize("mir", ["", "h", "v"])
    def test_valid_mirror(self, client, mir):
        data = client.post("/set_orientation", json={"mirror": mir}).get_json()
        assert data["orientation"]["mirror"] == mir

    def test_invalid_mirror_ignored(self, client):
        state.phone_orientation["mirror"] = "h"
        data = client.post("/set_orientation", json={"mirror": "x"}).get_json()
        assert data["orientation"]["mirror"] == "h"

    def test_missing_params_keeps_state(self, client):
        """缺参（空 JSON）：状态不变，仍返回 ok"""
        before = dict(state.phone_orientation)
        resp = client.post("/set_orientation", json={})
        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True
        assert state.phone_orientation == before

    def test_get_method_not_allowed(self, client):
        assert client.get("/set_orientation").status_code == 405


# ============================================================
# /set_source：合法地址 / 非法协议 / 空串
# ============================================================
class TestSetSource:
    def test_valid_http_url(self, client):
        resp = client.post("/set_source",
                           json={"url": "http://192.168.1.100:8080/video"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert data["url"] == "http://192.168.1.100:8080/video"
        assert state.phone_url_override == "http://192.168.1.100:8080/video"
        assert state.force_reconnect is True

    def test_url_without_scheme_auto_completed(self, client):
        data = client.post("/set_source",
                           json={"url": "192.168.1.100:8080"}).get_json()
        assert data["ok"] is True
        assert data["url"] == "http://192.168.1.100:8080"

    def test_rtsp_allowed(self, client):
        data = client.post("/set_source",
                           json={"url": "rtsp://192.168.1.5:554/stream"}).get_json()
        assert data["ok"] is True
        assert data["url"] == "rtsp://192.168.1.5:554/stream"

    def test_file_scheme_rejected(self, client):
        resp = client.post("/set_source", json={"url": "file:///C:/secret.txt"})
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["ok"] is False
        assert "协议" in data["error"]
        assert state.phone_url_override is None
        assert state.force_reconnect is False

    def test_javascript_scheme_rejected(self, client):
        resp = client.post("/set_source", json={"url": "javascript:alert(1)"})
        assert resp.status_code == 400
        assert resp.get_json()["ok"] is False

    def test_empty_url_resets_to_local(self, client):
        """空串 = 恢复本地摄像头"""
        state.phone_url_override = "http://192.168.1.100:8080/video"
        resp = client.post("/set_source", json={"url": ""})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert data["url"] == ""
        assert state.phone_url_override == ""
        assert state.force_reconnect is True

    def test_missing_url_field_same_as_empty(self, client):
        resp = client.post("/set_source", json={})
        assert resp.status_code == 200
        assert resp.get_json()["url"] == ""

    def test_missing_hostname_rejected(self, client):
        resp = client.post("/set_source", json={"url": "http://"})
        assert resp.status_code == 400
        assert "主机名" in resp.get_json()["error"]

    def test_underscore_hostname_accepted(self, client):
        """含下划线的主机名合法（手机推流 App 的设备名常含下划线）"""
        resp = client.post("/set_source", json={"url": "http://foo_bar.local"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert data["url"] == "http://foo_bar.local"

    def test_underscore_hostname_with_port_accepted(self, client):
        resp = client.post("/set_source", json={"url": "http://my_phone:8080/video"})
        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True

    def test_hostname_with_space_rejected(self, client):
        resp = client.post("/set_source", json={"url": "http://foo bar.local"})
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["ok"] is False
        assert "主机名" in data["error"]

    def test_hostname_with_consecutive_dots_rejected(self, client):
        resp = client.post("/set_source", json={"url": "http://foo..bar"})
        assert resp.status_code == 400
        assert resp.get_json()["ok"] is False

    def test_invalid_port_rejected(self, client):
        resp = client.post("/set_source", json={"url": "http://192.168.1.100:abc"})
        assert resp.status_code == 400
        assert resp.get_json()["ok"] is False

    def test_rejected_request_keeps_previous_source(self, client):
        state.phone_url_override = "http://10.0.0.9:8080/video"
        resp = client.post("/set_source", json={"url": "ftp://1.2.3.4"})
        assert resp.status_code == 400
        assert state.phone_url_override == "http://10.0.0.9:8080/video"


# ============================================================
# /set_model：切换模型（switch_model 在 routes 命名空间被打桩，绝不真加载）
# ============================================================
class TestSetModel:
    def test_set_model_success(self, client, monkeypatch):
        calls = {}

        def fake_switch(name):
            calls["name"] = name
            return {"model": name, "open_vocab": False}
        monkeypatch.setattr(routes, "switch_model", fake_switch)
        resp = client.post("/set_model", json={"model": "yolov8n.pt"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert data["model"] == "yolov8n.pt"
        assert data["open_vocab"] is False
        assert calls["name"] == "yolov8n.pt"

    def test_set_model_invalid_name_returns_400(self, client, monkeypatch):
        def fake_switch(name):
            raise ValueError("模型 'evil.pt' 不在可用列表中")
        monkeypatch.setattr(routes, "switch_model", fake_switch)
        resp = client.post("/set_model", json={"model": "evil.pt"})
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["ok"] is False
        assert "error" in data

    def test_set_model_load_failure_returns_500(self, client, monkeypatch):
        """白名单合法但加载失败（权重损坏）：500，旧模型保留"""
        def fake_switch(name):
            raise RuntimeError("权重损坏")
        monkeypatch.setattr(routes, "switch_model", fake_switch)
        resp = client.post("/set_model", json={"model": "yolov8m.pt"})
        assert resp.status_code == 500
        assert resp.get_json()["ok"] is False

    def test_set_model_missing_field_returns_400_without_switching(
            self, client, monkeypatch):
        """缺 model 字段：路由应在调用 switch_model 前就拦下"""
        def _boom(name):
            raise AssertionError("缺字段时不应调用 switch_model")
        monkeypatch.setattr(routes, "switch_model", _boom)
        resp = client.post("/set_model", json={})
        assert resp.status_code == 400
        assert resp.get_json()["ok"] is False

    def test_set_model_get_method_not_allowed(self, client):
        assert client.get("/set_model").status_code == 405

    @pytest.mark.parametrize("body", [
        {"model": 123},          # 非字符串（回归：(123).strip() 崩溃致 500）
        {"model": {"a": 1}},     # 非字符串 dict
        [1, 2],                  # 非 dict body
    ])
    def test_set_model_non_string_or_non_dict_returns_400(
            self, client, monkeypatch, body):
        """非法类型输入不得 500：路由应在调用 switch_model 前拦下返回 400"""
        def _boom(name):
            raise AssertionError("非法输入不应到达 switch_model")
        monkeypatch.setattr(routes, "switch_model", _boom)
        resp = client.post("/set_model", json=body)
        assert resp.status_code == 400
        assert resp.get_json()["ok"] is False
