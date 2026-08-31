# -*- coding: utf-8 -*-
"""zones.py 测试：point_in_polygon 纯函数、区域校验、报警冷却、报警音（打桩）。

隔离策略：
  - 冷却/时间逻辑通过 monkeypatch 替换 zones.time，用可控时钟测试；
  - winsound 全程打桩（绝不真实发声），报警计时器替换为不启动线程的替身；
  - 不触碰 detector.get_model（conftest 哨兵兜底）。
"""

import pytest

import zones
from config import settings


# ============================================================
# point_in_polygon：凸/凹多边形、贴边、退化输入
# ============================================================
class TestPointInPolygon:
    SQUARE = [(0, 0), (1, 0), (1, 1), (0, 1)]
    # L 形凹多边形：右下象限 (0.6,0.6) 在凹口外
    CONCAVE = [(0, 0), (1, 0), (1, 0.5), (0.5, 0.5), (0.5, 1), (0, 1)]

    def test_inside_convex(self):
        assert zones.point_in_polygon(0.5, 0.5, self.SQUARE) is True

    def test_outside_convex(self):
        assert zones.point_in_polygon(2.0, 2.0, self.SQUARE) is False
        assert zones.point_in_polygon(-0.1, 0.5, self.SQUARE) is False

    def test_inside_concave(self):
        assert zones.point_in_polygon(0.2, 0.8, self.CONCAVE) is True
        assert zones.point_in_polygon(0.8, 0.2, self.CONCAVE) is True

    def test_outside_concave_notch(self):
        """凹口区域应判为外部（射线法对凹多边形同样适用）"""
        assert zones.point_in_polygon(0.8, 0.8, self.CONCAVE) is False

    def test_point_on_edge_is_deterministic(self):
        """贴边点不抛异常且结果稳定（射线法对边界点不保证语义，
        只要求同一输入结果确定、不崩溃）"""
        r1 = zones.point_in_polygon(0.5, 0.0, self.SQUARE)
        r2 = zones.point_in_polygon(0.5, 0.0, self.SQUARE)
        assert r1 == r2
        assert isinstance(r1, bool)

    @pytest.mark.parametrize("degenerate", [[], [(0, 0)], [(0, 0), (1, 1)]])
    def test_degenerate_polygon_returns_false(self, degenerate):
        """点数 <3 的退化多边形一律判为外部"""
        assert zones.point_in_polygon(0.5, 0.5, degenerate) is False


# ============================================================
# validate_regions：合法 / 各类非法输入（纯函数直测）
# ============================================================
class TestValidateRegions:
    def test_valid_multi_regions(self):
        data = {"regions": [
            {"name": "门口", "points": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9]]},
            {"points": [[0, 0], [1, 0], [1, 1], [0, 1]]},
        ]}
        regions, error = zones.validate_regions(data)
        assert error is None
        assert len(regions) == 2
        assert regions[0]["id"] == "zone_1"
        assert regions[0]["name"] == "门口"
        # 未提供名称时给默认名
        assert regions[1]["name"] == "区域 2"
        assert regions[1]["points"] == [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]

    def test_empty_regions_is_valid(self):
        regions, error = zones.validate_regions({"regions": []})
        assert error is None
        assert regions == []

    def test_too_many_regions_rejected(self):
        tri = {"points": [[0, 0], [1, 0], [0.5, 1]]}
        regions, error = zones.validate_regions({"regions": [tri] * 11})
        assert regions is None
        assert "10" in error

    def test_too_few_points_rejected(self):
        regions, error = zones.validate_regions(
            {"regions": [{"points": [[0, 0], [1, 1]]}]})
        assert regions is None
        assert "3 个顶点" in error

    @pytest.mark.parametrize("bad_pt", [[1.5, 0.5], [0.5, -0.1]])
    def test_coord_out_of_range_rejected(self, bad_pt):
        regions, error = zones.validate_regions(
            {"regions": [{"points": [[0, 0], [1, 0], bad_pt]}]})
        assert regions is None
        assert "0~1" in error

    @pytest.mark.parametrize("bad_pt", [[0.5], [0.1, 0.2, 0.3], "xy"])
    def test_point_not_a_pair_rejected(self, bad_pt):
        regions, error = zones.validate_regions(
            {"regions": [{"points": [[0, 0], [1, 0], bad_pt]}]})
        assert regions is None
        assert "二元组" in error

    def test_non_numeric_coord_rejected(self):
        regions, error = zones.validate_regions(
            {"regions": [{"points": [[0, 0], [1, 0], ["a", "b"]]}]})
        assert regions is None
        assert "数字" in error

    @pytest.mark.parametrize("bad_body", [None, {}, {"regions": "nope"}, {"regions": 3}])
    def test_regions_not_a_list_rejected(self, bad_body):
        regions, error = zones.validate_regions(bad_body)
        assert regions is None
        assert "regions" in error

    def test_region_not_a_dict_rejected(self):
        regions, error = zones.validate_regions({"regions": ["x"]})
        assert regions is None
        assert "对象" in error

    @pytest.mark.parametrize("bad_name", [123, {"a": 1}, [1]])
    def test_non_string_name_rejected(self, bad_name):
        """非字符串 name 返回错误而非 AttributeError（回归：(123).strip() 500）"""
        regions, error = zones.validate_regions(
            {"regions": [{"name": bad_name,
                          "points": [[0, 0], [1, 0], [0.5, 1]]}]})
        assert regions is None
        assert "字符串" in error

    def test_overlong_name_rejected(self):
        regions, error = zones.validate_regions(
            {"regions": [{"name": "区" * 51,
                          "points": [[0, 0], [1, 0], [0.5, 1]]}]})
        assert regions is None
        assert "过长" in error
        # 边界：刚好 50 字符合法
        regions, error = zones.validate_regions(
            {"regions": [{"name": "区" * 50,
                          "points": [[0, 0], [1, 0], [0.5, 1]]}]})
        assert error is None
        assert regions[0]["name"] == "区" * 50


# ============================================================
# 报警冷却（可控时钟）
# ============================================================
class TestAlertCooldown:
    def test_cooldown_suppresses_repeat_and_expires(self, monkeypatch):
        t = {"v": 1000.0}
        monkeypatch.setattr(zones.time, "monotonic", lambda: t["v"])
        zones._last_alert.clear()
        try:
            assert zones._cooldown_ok("zone_1", "person") is True
            # 冷却期内：不重复触发
            t["v"] += max(1, settings.alert_cooldown_s - 1)
            assert zones._cooldown_ok("zone_1", "person") is False
            # 冷却期过后：允许再次触发
            t["v"] += 2.0
            assert zones._cooldown_ok("zone_1", "person") is True
            # 不同 (区域, 类别) 组合互不影响
            assert zones._cooldown_ok("zone_2", "person") is True
            assert zones._cooldown_ok("zone_1", "dog") is True
        finally:
            zones._last_alert.clear()


# ============================================================
# 报警音：winsound 全程打桩，绝不真实发声
# ============================================================
class _FakeTimer:
    """threading.Timer 替身：记录但不真正起线程"""
    instances = []

    def __init__(self, interval, func, args=None, kwargs=None):
        self.interval = interval
        self.func = func
        self.cancelled = False
        _FakeTimer.instances.append(self)

    def start(self):
        pass

    def cancel(self):
        self.cancelled = True


class TestTriggerAlarm:
    @pytest.fixture
    def fake_sound(self, monkeypatch):
        """把 winsound 换成记录器、计时器换成替身，保证无声且不产生游离线程"""
        import sys
        import types

        calls = []
        fake_ws = types.ModuleType("winsound")
        fake_ws.PlaySound = lambda *a, **k: calls.append(a)
        fake_ws.SND_FILENAME = 0x20000
        fake_ws.SND_ASYNC = 1
        fake_ws.SND_LOOP = 8
        monkeypatch.setitem(sys.modules, "winsound", fake_ws)
        monkeypatch.setattr(zones.threading, "Timer", _FakeTimer)
        _FakeTimer.instances = []
        return calls

    def test_trigger_plays_silently_with_fake_winsound(self, monkeypatch, fake_sound):
        monkeypatch.setattr(zones.sys, "platform", "win32")
        zones._alarm_active = False
        monkeypatch.setattr(zones, "_ensure_alarm_wav", lambda: "/tmp/fake.wav")
        zones.trigger_alarm()
        assert len(fake_sound) == 1                      # 播放了一次（打桩）
        assert fake_sound[0][0] == "/tmp/fake.wav"
        assert zones._alarm_active is True
        assert len(_FakeTimer.instances) == 1            # 安排了停止计时器

    def test_retrigger_does_not_stack_playback(self, monkeypatch, fake_sound):
        """播放中再触发：只重置计时器，不叠加播放"""
        monkeypatch.setattr(zones.sys, "platform", "win32")
        zones._alarm_active = False
        monkeypatch.setattr(zones, "_ensure_alarm_wav", lambda: "/tmp/fake.wav")
        zones.trigger_alarm()
        first_timer = _FakeTimer.instances[-1]
        zones.trigger_alarm()
        assert len(fake_sound) == 1                      # 未重复 PlaySound
        assert first_timer.cancelled is True             # 旧计时器被重置
        assert len(_FakeTimer.instances) == 2

    def test_stop_alarm_resets_active_flag(self):
        zones._alarm_active = True
        zones._stop_alarm_sound()   # 内部 winsound 导入失败/异常均被吞
        assert zones._alarm_active is False


# ============================================================
# 事件记录
# ============================================================
class TestRecordEvent:
    def test_record_event_appends_to_state(self):
        import state
        with state.events_lock:
            before = len(state.events)
        ev = zones._record_event("zone_1", "门口", "person", 7, 0.91)
        assert ev["zone_name"] == "门口"
        assert ev["class"] == "person"
        assert ev["track_id"] == 7
        assert ev["confidence"] == 0.91
        assert isinstance(ev["id"], int)
        with state.events_lock:
            assert len(state.events) == before + 1
            assert state.events[-1] is ev or state.events[-1]["id"] == ev["id"]


# ============================================================
# clear_zones：联动清空区内状态与报警冷却（回归）
# ============================================================
class TestClearZones:
    def test_clear_zones_resets_inside_and_cooldown(self, monkeypatch, tmp_path):
        """旋转清空后冷却被重置：重画同 ID 区域，首次入侵立即可再触发，
        不被旧冷却吞掉（可控时钟下验证）"""
        monkeypatch.setattr(settings, "zones_file", str(tmp_path / "zones.json"))
        t = {"v": 1000.0}
        monkeypatch.setattr(zones.time, "monotonic", lambda: t["v"])

        # 预置区内状态 + 触发一次报警冷却（冷却 30s 未过）
        zones._inside[(("id", 7), "zone_1")] = t["v"]
        assert zones._cooldown_ok("zone_1", "person") is True
        assert zones._cooldown_ok("zone_1", "person") is False   # 冷却内不再触发
        t["v"] += 5.0   # 仍在冷却期内

        zones.clear_zones()

        assert zones.get_zones() == []
        assert zones._inside == {}
        assert zones._last_alert == {}
        # 冷却已重置：仍在原冷却时间窗内，但可立即再次触发报警
        assert zones._cooldown_ok("zone_1", "person") is True


# ============================================================
# zones.json 加载：损坏文件数值校验（回归）
# ============================================================
class TestLoadZonesValidation:
    def test_corrupt_coords_skipped_without_raising(self, monkeypatch, tmp_path):
        """非数字坐标/NaN/超范围区域被跳过，合法区域保留，全程不抛错"""
        import json

        zf = tmp_path / "zones.json"
        zf.write_text(json.dumps({"regions": [
            {"name": "good",
             "points": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9]]},
            {"name": "bad-str",
             "points": [["a", 0.2], [0.3, 0.4], [0.5, 0.6]]},
            {"name": "bad-nan",
             "points": [[float("nan"), 0.1], [0.3, 0.4], [0.5, 0.6]]},
            {"name": "bad-range",
             "points": [[1.5, 0.1], [0.3, 0.4], [0.5, 0.6]]},
        ]}), encoding="utf-8")
        monkeypatch.setattr(settings, "zones_file", str(zf))

        zones._load_zones_from_file()   # 不得抛异常（否则检查线程每 0.2s 报错）
        zs = zones.get_zones()
        assert len(zs) == 1
        assert zs[0]["name"] == "good"
        assert zs[0]["points"] == [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9]]

    def test_all_invalid_zones_yields_empty(self, monkeypatch, tmp_path):
        """全部非法 → 空区域且不抛错"""
        import json

        zf = tmp_path / "zones.json"
        zf.write_text(json.dumps({"regions": [
            {"points": [["x", "y"], [0.3, 0.4], [0.5, 0.6]]},
            {"points": [[float("nan"), float("nan")], [0.3, 0.4], [0.5, 0.6]]},
        ]}), encoding="utf-8")
        monkeypatch.setattr(settings, "zones_file", str(zf))

        zones._load_zones_from_file()
        assert zones.get_zones() == []
