# -*- coding: utf-8 -*-
"""tools/eval_dataset.py 的 load_yaml_names 手写 YAML 解析测试。

注意：只测纯解析函数；eval_dataset.main 会真实加载模型并跑推理，
属于硬件/权重依赖路径，不在自动化测试范围内。
"""

import pytest

import eval_dataset
from config import BASE_DIR
import os


def _write(tmp_path, content):
    p = tmp_path / "data.yaml"
    p.write_text(content, encoding="utf-8")
    return str(p)


class TestLoadYamlNames:
    def test_real_dataset_yaml(self):
        """解析项目内真实 data.yaml：124 个类别，首尾类别可验证"""
        path = os.path.join(BASE_DIR, "dataset", "data.yaml")
        names = eval_dataset.load_yaml_names(path)
        assert len(names) == 124
        assert names[0] == "person"
        assert names[1] == "bicycle"
        assert names[79] == "toothbrush"
        assert names[-1] == "plastic bag"

    def test_map_style_names(self, tmp_path):
        content = "path: ./dataset\nnc: 3\nnames:\n  0: cup\n  1: bottle\n  2: phone\n"
        assert eval_dataset.load_yaml_names(_write(tmp_path, content)) \
            == ["cup", "bottle", "phone"]

    def test_names_with_colon_and_spaces_in_value(self, tmp_path):
        """值中含空格应被 strip；解析按第一个冒号切分"""
        content = "names:\n  0:   dining table  \n  1: traffic light\n"
        assert eval_dataset.load_yaml_names(_write(tmp_path, content)) \
            == ["dining table", "traffic light"]

    def test_stops_at_first_non_numeric_entry(self, tmp_path):
        """names 段遇到非 '数字: 值' 的有效行即停止解析"""
        content = "names:\n  0: cup\n  1: pen\ntrain: images\n  9: ghost\n"
        assert eval_dataset.load_yaml_names(_write(tmp_path, content)) == ["cup", "pen"]

    def test_no_names_section_returns_empty(self, tmp_path):
        content = "path: ./dataset\nnc: 0\ntrain: images\nval: images\n"
        assert eval_dataset.load_yaml_names(_write(tmp_path, content)) == []

    def test_empty_names_section(self, tmp_path):
        content = "names:\n"
        assert eval_dataset.load_yaml_names(_write(tmp_path, content)) == []

    def test_comment_lines_after_names_do_not_break(self, tmp_path):
        """names 段内的注释行（# 开头）被跳过而不是终止"""
        content = "names:\n  0: cup\n  # 注释行应被忽略\n  1: pen\n"
        # 注意：空行/注释行不满足 isdigit 分支，但 '#' 开头行被显式跳过
        assert eval_dataset.load_yaml_names(_write(tmp_path, content)) == ["cup", "pen"]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises((FileNotFoundError, OSError)):
            eval_dataset.load_yaml_names(str(tmp_path / "not_exist.yaml"))
