# -*- coding: utf-8 -*-
"""tools/audit_labels.py 的判定合成逻辑测试（纯函数，不加载模型）。

判定规则是「哪些框可以安全改名」的唯一依据，必须锁死：
只有双模型证据一致时才给 HIGH，单侧证据降级，证据冲突或缺失一律 AMBIGUOUS。
"""

import pytest

from audit_labels import verdict


class TestVerdict:
    def test_both_crop_models_agree_person_is_high(self):
        assert verdict(("person", 0.9, 0.6), ("person", 0.8, 0.6), None, None) \
            == ("person", "HIGH")

    def test_both_crop_models_agree_bus_is_high(self):
        """双模型裁剪都说是 bus 时必须保留 bus，不能被全局改名带走"""
        assert verdict(("bus", 0.95, 0.7), ("bus", 0.97, 0.7), ("bus", 0.9), ("bus", 0.9)) \
            == ("bus", "HIGH")

    def test_crop_models_disagree_is_ambiguous(self):
        assert verdict(("person", 0.5, 0.6), ("bus", 0.5, 0.6), None, None) \
            == (None, "AMBIGUOUS")

    def test_crop_conflict_stays_ambiguous_even_if_full_agrees(self):
        """裁剪证据冲突时，即使全图取证偏向其中一方也不得改名"""
        assert verdict(("person", 0.5, 0.6), ("bicycle", 0.4, 0.6),
                       ("person", 0.6), ("person", 0.5)) == (None, "AMBIGUOUS")

    def test_single_crop_plus_full_agreement_is_high(self):
        assert verdict(("person", 0.4, 0.6), None, ("person", 0.6), None) \
            == ("person", "HIGH")

    def test_single_crop_alone_is_medium(self):
        assert verdict(None, ("person", 0.6, 0.6), None, None) == ("person", "MEDIUM")

    def test_full_models_agree_without_crop_is_medium(self):
        assert verdict(None, None, ("person", 0.6), ("person", 0.5)) == ("person", "MEDIUM")

    def test_only_one_full_match_is_low(self):
        assert verdict(None, None, ("person", 0.6), None) == ("person", "LOW")

    def test_no_evidence_is_ambiguous(self):
        assert verdict(None, None, None, None) == (None, "AMBIGUOUS")

    def test_crop_person_overrides_conflicting_full(self):
        """裁剪取证优先于全图取证：全图把行人叫成 bus 时仍以裁剪为准，但降为 MEDIUM"""
        cls, conf = verdict(("person", 0.6, 0.6), None, ("bus", 0.3), None)
        assert (cls, conf) == ("person", "MEDIUM")
