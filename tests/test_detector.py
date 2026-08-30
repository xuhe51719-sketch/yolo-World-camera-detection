# -*- coding: utf-8 -*-
"""detector.py 纯函数单元测试（_box_iou / get_color）"""

import pytest

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
