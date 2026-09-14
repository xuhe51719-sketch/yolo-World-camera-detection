# -*- coding: utf-8 -*-
"""detector.py 单元测试：纯函数（_box_iou / get_color）与模型热切换
（list_available_models / switch_model / apply_model_reload）。"""

import os

import pytest

import detector
from detector import _box_iou, get_color


# ============================================================
# _box_iou：相交 / 不相交 / 包含 / 零面积框
# ============================================================
class TestBoxIou:
    def test_identical_boxes_iou_1(self):
        assert _box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)

    def test_partial_overlap(self):
        # 交集 5x10=50，并集 100+100-50=150
        assert _box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50.0 / 150.0)

    def test_half_overlap(self):
        assert _box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)

    def test_disjoint_boxes_iou_0(self):
        assert _box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0

    def test_touching_edges_iou_0(self):
        """仅边相邻（交集面积为 0）"""
        assert _box_iou([0, 0, 10, 10], [10, 0, 20, 10]) == 0.0

    def test_containment(self):
        # 小框完全在大框内：交集=小框面积 25，并集=100
        assert _box_iou([0, 0, 10, 10], [0, 0, 5, 5]) == pytest.approx(25.0 / 100.0)

    def test_zero_area_box(self):
        """宽或高为 0 的退化框：面积为 0，IoU 为 0"""
        assert _box_iou([0, 0, 0, 10], [0, 0, 10, 10]) == 0.0
        assert _box_iou([5, 5, 5, 5], [5, 5, 5, 5]) == 0.0

    def test_both_zero_area_no_division_error(self):
        """两个零面积框：并集为 0，走兜底分支返回 0 而不是除零"""
        assert _box_iou([0, 0, 0, 0], [0, 0, 0, 0]) == 0.0

    def test_inverted_box_treated_as_zero_area(self):
        """x2<x1 的非法框按 0 面积处理（max(0, ...) 保护）"""
        assert _box_iou([10, 10, 0, 0], [0, 0, 10, 10]) == 0.0

    def test_symmetric(self):
        a, b = [0, 0, 10, 10], [3, 4, 12, 15]
        assert _box_iou(a, b) == pytest.approx(_box_iou(b, a))


# ============================================================
# get_color：索引取模 / 稳定可复现
# ============================================================
class TestGetColor:
    def test_returns_bgr_triple_of_ints(self):
        color = get_color(0)
        assert isinstance(color, tuple)
        assert len(color) == 3
        assert all(isinstance(c, int) for c in color)

    def test_color_values_within_range(self):
        """randint(60, 255) 的低边界为 60，高边界为 254（numpy 上界开区间）"""
        for idx in (0, 1, 37, 99, 100, 250):
            b, g, r = get_color(idx)
            assert 60 <= b <= 254
            assert 60 <= g <= 254
            assert 60 <= r <= 254

    def test_deterministic_and_reproducible(self):
        """同一索引多次调用颜色一致（内部固定种子）"""
        for idx in (0, 7, 42, 99):
            assert get_color(idx) == get_color(idx)

    def test_index_wraps_modulo_100(self):
        """索引越界按 100 取模：idx 与 idx+100 同色"""
        for idx in (0, 3, 99):
            assert get_color(idx) == get_color(idx + 100)
            assert get_color(idx) == get_color(idx + 300)

    def test_different_indices_usually_differ(self):
        """不同索引大概率不同色（100 组随机颜色全相同的概率为零）"""
        colors = {get_color(i) for i in range(20)}
        assert len(colors) > 1


# ============================================================
# 模型热切换：list_available_models（扫描根目录 *.pt，不递归）
# ============================================================
class TestListAvailableModels:
    def test_scans_only_pt_files_sorted(self, tmp_path):
        (tmp_path / "yolov8n.pt").write_bytes(b"x")
        (tmp_path / "yolov8m.pt").write_bytes(b"x")
        (tmp_path / "readme.txt").write_text("忽略我")
        names = [m["name"]
                 for m in detector.list_available_models(search_dir=str(tmp_path))]
        assert names == ["yolov8m.pt", "yolov8n.pt"]   # 仅 .pt，按名排序

    def test_does_not_recurse_into_subdirs(self, tmp_path):
        (tmp_path / "yolov8n.pt").write_bytes(b"x")
        clip = tmp_path / "weights" / "clip"
        clip.mkdir(parents=True)
        (clip / "ViT-B-32.pt").write_bytes(b"x")
        names = [m["name"]
                 for m in detector.list_available_models(search_dir=str(tmp_path))]
        assert names == ["yolov8n.pt"]   # 子目录（weights/clip）里的 .pt 不算

    def test_marks_world_model(self, tmp_path):
        (tmp_path / "yolov8x-worldv2.pt").write_bytes(b"x")
        (tmp_path / "yolov8n.pt").write_bytes(b"x")
        by_name = {m["name"]: m
                   for m in detector.list_available_models(search_dir=str(tmp_path))}
        assert by_name["yolov8x-worldv2.pt"]["is_world"] is True
        assert by_name["yolov8n.pt"]["is_world"] is False

    def test_marks_active_from_settings(self, tmp_path, monkeypatch):
        (tmp_path / "yolov8n.pt").write_bytes(b"x")
        (tmp_path / "yolov8m.pt").write_bytes(b"x")
        monkeypatch.setattr(detector.settings, "model_path",
                            str(tmp_path / "yolov8n.pt"))
        by_name = {m["name"]: m
                   for m in detector.list_available_models(search_dir=str(tmp_path))}
        assert by_name["yolov8n.pt"]["active"] is True
        assert by_name["yolov8m.pt"]["active"] is False

    def test_empty_dir_returns_empty_list(self, tmp_path):
        assert detector.list_available_models(search_dir=str(tmp_path)) == []


# ============================================================
# 模型热切换：switch_model 白名单校验（非法名在加载前即拒绝）
# ============================================================
class TestSwitchModelValidation:
    def test_rejects_name_not_in_whitelist(self, tmp_path):
        (tmp_path / "yolov8n.pt").write_bytes(b"x")
        with pytest.raises(ValueError):
            detector.switch_model("nonexistent.pt", search_dir=str(tmp_path))

    def test_rejects_path_traversal(self, tmp_path):
        (tmp_path / "yolov8n.pt").write_bytes(b"x")
        with pytest.raises(ValueError):
            detector.switch_model("../../evil.pt", search_dir=str(tmp_path))

    def test_rejects_subdir_reference(self, tmp_path):
        (tmp_path / "yolov8n.pt").write_bytes(b"x")
        with pytest.raises(ValueError):
            detector.switch_model("weights/clip/ViT-B-32.pt",
                                  search_dir=str(tmp_path))

    def test_rejects_non_pt_file(self, tmp_path):
        (tmp_path / "notes.txt").write_text("x")
        with pytest.raises(ValueError):
            detector.switch_model("notes.txt", search_dir=str(tmp_path))


# ============================================================
# 模型热切换：switch_model 成功路径（打桩加载，绝不 load 真实权重）
# ============================================================
class TestSwitchModelSuccess:
    def test_swaps_model_updates_settings_and_sets_flag(
            self, tmp_path, monkeypatch, fake_model):
        (tmp_path / "yolov8n.pt").write_bytes(b"x")
        loaded_paths = []
        monkeypatch.setattr(
            detector, "_load_model",
            lambda p: (loaded_paths.append(p), fake_model)[1])
        monkeypatch.setattr(detector.settings, "model_path", "OLD.pt")
        monkeypatch.setattr(detector.settings, "is_world_model", True)
        monkeypatch.setattr(detector, "_model", None)
        monkeypatch.setattr(detector, "_model_reload_requested", False)

        info = detector.switch_model("yolov8n.pt", search_dir=str(tmp_path))

        assert detector._model is fake_model
        assert os.path.basename(detector.settings.model_path) == "yolov8n.pt"
        assert detector.settings.is_world_model is False
        assert detector._model_reload_requested is True
        assert info == {"model": "yolov8n.pt", "open_vocab": False}
        assert loaded_paths and os.path.basename(loaded_paths[0]) == "yolov8n.pt"

    def test_load_failure_keeps_old_model(
            self, tmp_path, monkeypatch, fake_model):
        (tmp_path / "yolov8m.pt").write_bytes(b"x")
        monkeypatch.setattr(detector, "_model", fake_model)   # 旧模型在位
        monkeypatch.setattr(detector.settings, "model_path", "OLD.pt")

        def _boom(p):
            raise RuntimeError("权重损坏")
        monkeypatch.setattr(detector, "_load_model", _boom)

        with pytest.raises(RuntimeError):
            detector.switch_model("yolov8m.pt", search_dir=str(tmp_path))
        assert detector._model is fake_model            # 旧模型未被替换
        assert detector.settings.model_path == "OLD.pt"  # settings 未改


# ============================================================
# 模型热切换：apply_model_reload（检测循环 reload 分支的可测单元）
# ============================================================
class TestApplyModelReload:
    def test_refetches_model_resets_tracker_and_clears_detections(
            self, monkeypatch, fake_model):
        import metrics as metrics_mod
        import state as state_mod
        monkeypatch.setattr(detector, "get_model", lambda: fake_model)
        monkeypatch.setattr(detector, "_tracker_reset_requested", False)
        monkeypatch.setattr(detector, "_model_reload_requested", True)
        monkeypatch.setitem(metrics_mod.metrics, "prev_tracks",
                            [("person", [0, 0, 1, 1], 1)])
        metrics_mod.metrics["detections_total"] = 7
        metrics_mod.metrics["class_counts"]["person"] = 3
        with state_mod.detections_lock:
            state_mod.latest_detections = [{"class": "person"}]

        returned = detector.apply_model_reload()

        assert returned is fake_model
        assert detector._model_reload_requested is False    # 标志已复位
        assert detector._tracker_reset_requested is True    # 请求重建跟踪
        assert state_mod.latest_detections == []            # 旧检测框已清空
        assert metrics_mod.metrics["prev_tracks"] == []     # 防 ID 切换虚增
        # 换模型与换源同口径：跨模型无意义的累计统计一并清零
        assert metrics_mod.metrics["detections_total"] == 0
        assert not metrics_mod.metrics["class_counts"]
