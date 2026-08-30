# -*- coding: utf-8 -*-
"""camera.py 纯函数单元测试（无硬件、无模型、无网络）"""

import numpy as np
import pytest

import state
from camera import apply_orientation, candidate_urls, normalize_camera_url


# ============================================================
# normalize_camera_url：全角转半角 / 空串 / 协议头补全
# ============================================================
class TestNormalizeCameraUrl:
    def test_fullwidth_to_halfwidth(self):
        """中文输入法下常见的全角数字/符号应转为半角并自动补协议头。
        注意：实现的映射表只覆盖全角数字与常用符号，不含全角字母"""
        assert normalize_camera_url("１９２．１６８．１．１００：８０８０／８０") \
            == "http://192.168.1.100:8080/80"

    def test_fullwidth_digits_and_symbols(self):
        """只有单个冒号（非 ://）仍会补 http:// 协议头"""
        assert normalize_camera_url("１２３．４５：８０") == "http://123.45:80"

    def test_empty_string(self):
        assert normalize_camera_url("") == ""

    def test_none_input(self):
        assert normalize_camera_url(None) == ""

    def test_whitespace_only(self):
        assert normalize_camera_url("   ") == ""

    def test_no_scheme_gets_http_prefix(self):
        assert normalize_camera_url("192.168.1.100:8080/video") \
            == "http://192.168.1.100:8080/video"

    def test_existing_http_scheme_unchanged(self):
        assert normalize_camera_url("http://192.168.1.100:8080") \
            == "http://192.168.1.100:8080"

    def test_existing_rtsp_scheme_unchanged(self):
        assert normalize_camera_url("rtsp://192.168.1.5:554/stream") \
            == "rtsp://192.168.1.5:554/stream"

    def test_surrounding_whitespace_stripped(self):
        assert normalize_camera_url("  http://1.2.3.4:8080  ") == "http://1.2.3.4:8080"


# ============================================================
# candidate_urls：/video 补全 / https 降级 / 尾斜杠 / 去重
# ============================================================
class TestCandidateUrls:
    def test_root_path_prepends_video_first(self):
        """无路径时优先尝试 /video（IP Webcam 的流实际在 /video）"""
        urls = candidate_urls("http://192.168.1.100:8080")
        assert urls == ["http://192.168.1.100:8080/video",
                        "http://192.168.1.100:8080"]

    def test_trailing_slash_also_gets_video(self):
        urls = candidate_urls("http://192.168.1.100:8080/")
        assert urls[0] == "http://192.168.1.100:8080/video"
        assert "http://192.168.1.100:8080/" in urls

    def test_https_falls_back_to_http_first(self):
        """误填 https 时，对应 http 地址优先"""
        urls = candidate_urls("https://192.168.1.100:8080")
        assert urls[0] == "http://192.168.1.100:8080/video"
        assert urls[1] == "http://192.168.1.100:8080"
        assert urls[2] == "https://192.168.1.100:8080/video"
        assert urls[3] == "https://192.168.1.100:8080"

    def test_existing_path_not_appended_video(self):
        urls = candidate_urls("http://192.168.1.100:8080/video")
        assert urls == ["http://192.168.1.100:8080/video"]

    def test_query_string_not_merged_into_video_path(self):
        """/video 只拼在主机名后，不能拼进查询串"""
        urls = candidate_urls("http://192.168.1.100:8080?a=1")
        assert "http://192.168.1.100:8080/video" in urls

    def test_dedup(self):
        urls = candidate_urls("http://192.168.1.100:8080/video")
        assert urls == list(dict.fromkeys(urls))
        assert len(urls) == len(set(urls))

    def test_rtsp_no_video_candidate(self):
        """非 http(s) 协议不补 /video"""
        urls = candidate_urls("rtsp://192.168.1.5:554")
        assert urls == ["rtsp://192.168.1.5:554"]


# ============================================================
# apply_orientation：4 旋转 × 镜像组合（合成图验证形状/像素）
# ============================================================
class TestApplyOrientation:
    @pytest.fixture(autouse=True)
    def _reset_orientation(self):
        """每个用例独立控制方向状态"""
        saved = dict(state.phone_orientation)
        state.phone_orientation.update({"rotate": 0, "mirror": ""})
        yield
        state.phone_orientation.clear()
        state.phone_orientation.update(saved)

    @pytest.fixture
    def asym_frame(self):
        """2 行 3 列的非对称图（旋转后形状可区分，像素可追踪）：
        [[1, 2, 3],
         [4, 5, 6]]"""
        return np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)

    def test_rotate_0_no_change(self, asym_frame):
        out = apply_orientation(asym_frame)
        assert np.array_equal(out, asym_frame)

    def test_rotate_90_clockwise(self, asym_frame):
        state.phone_orientation["rotate"] = 90
        out = apply_orientation(asym_frame)
        assert out.shape == (3, 2)
        assert np.array_equal(out, np.array([[4, 1], [5, 2], [6, 3]], dtype=np.uint8))

    def test_rotate_180(self, asym_frame):
        state.phone_orientation["rotate"] = 180
        out = apply_orientation(asym_frame)
        assert out.shape == (2, 3)
        assert np.array_equal(out, np.array([[6, 5, 4], [3, 2, 1]], dtype=np.uint8))

    def test_rotate_270(self, asym_frame):
        state.phone_orientation["rotate"] = 270
        out = apply_orientation(asym_frame)
        assert out.shape == (3, 2)
        assert np.array_equal(out, np.array([[3, 6], [2, 5], [1, 4]], dtype=np.uint8))

    def test_mirror_horizontal(self, asym_frame):
        state.phone_orientation["mirror"] = "h"
        out = apply_orientation(asym_frame)
        assert np.array_equal(out, np.array([[3, 2, 1], [6, 5, 4]], dtype=np.uint8))

    def test_mirror_vertical(self, asym_frame):
        state.phone_orientation["mirror"] = "v"
        out = apply_orientation(asym_frame)
        assert np.array_equal(out, np.array([[4, 5, 6], [1, 2, 3]], dtype=np.uint8))

    @pytest.mark.parametrize("rot", [0, 90, 180, 270])
    @pytest.mark.parametrize("mirror", ["", "h", "v"])
    def test_rotation_x_mirror_shape_matrix(self, asym_frame, rot, mirror):
        """全组合形状矩阵：仅 90/270 交换宽高"""
        state.phone_orientation.update({"rotate": rot, "mirror": mirror})
        out = apply_orientation(asym_frame)
        expected_shape = (3, 2) if rot in (90, 270) else (2, 3)
        assert out.shape == expected_shape
        # 像素集合不因刚体变换改变
        assert sorted(out.flatten().tolist()) == sorted(asym_frame.flatten().tolist())

    def test_rotate90_plus_mirror_h_matches_known_composition(self, asym_frame):
        """先顺时针旋转 90°，再水平镜像（代码实际顺序）"""
        state.phone_orientation.update({"rotate": 90, "mirror": "h"})
        out = apply_orientation(asym_frame)
        rotated = np.array([[4, 1], [5, 2], [6, 3]], dtype=np.uint8)
        assert np.array_equal(out, np.fliplr(rotated))
