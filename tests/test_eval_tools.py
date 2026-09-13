# -*- coding: utf-8 -*-
"""tools/eval_dataset.py 的纯函数测试：YAML 解析、同义词归一、split 发现、标注重映射。

注意：只测纯解析/文件重组函数；eval_dataset.main 会真实加载模型并跑推理，
属于硬件/权重依赖路径，不在自动化测试范围内。
"""

import os

import pytest

import eval_dataset
from config import BASE_DIR


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


class TestLoadYamlNamesFlowList:
    """Roboflow 导出的 data.yaml 用 names: ['a', 'b'] 行内列表写法"""

    def test_real_roboflow_dataset_yaml(self):
        path = os.path.join(BASE_DIR, "datasets", "world-monitoring-v2-121", "data.yaml")
        if not os.path.exists(path):
            pytest.skip("Roboflow 导出数据集不在本地")
        names = eval_dataset.load_yaml_names(path)
        assert len(names) == 17
        assert names[0] == "YOLO-World-Monitoring-System"
        assert names[4] == "car"
        assert names[-1] == "well lid"

    def test_quoted_flow_list(self, tmp_path):
        content = "nc: 3\nnames: ['bottle', 'trash can', 'water dispenser']\n"
        assert eval_dataset.load_yaml_names(_write(tmp_path, content)) == \
            ["bottle", "trash can", "water dispenser"]

    def test_unquoted_flow_list(self, tmp_path):
        content = "names: [car, bus, truck]\n"
        assert eval_dataset.load_yaml_names(_write(tmp_path, content)) == ["car", "bus", "truck"]

    def test_flow_list_multiline(self, tmp_path):
        content = "names: ['car',\n  'bus',\n  'truck']\ntrain: images\n"
        assert eval_dataset.load_yaml_names(_write(tmp_path, content)) == ["car", "bus", "truck"]

    def test_empty_flow_list(self, tmp_path):
        assert eval_dataset.load_yaml_names(_write(tmp_path, "names: []\n")) == []


class TestCanonical:
    def test_synonyms_map_to_coco_names(self):
        assert eval_dataset.canonical("motorbike") == "motorcycle"
        assert eval_dataset.canonical("Fridge") == "refrigerator"
        assert eval_dataset.canonical(" sofa ") == "couch"
        assert eval_dataset.canonical("table") == "dining table"

    def test_unknown_name_lowered_and_kept(self):
        assert eval_dataset.canonical("Well Lid") == "well lid"


def _make_split_dataset(root, names, per_split):
    """造一个 split 式数据集：per_split = {split: {stem: [标注行]}}"""
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(
        "nc: %d\nnames: %s\n" % (len(names), names), encoding="utf-8")
    for split, items in per_split.items():
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
        for stem, lines in items.items():
            # 1x1 白色 PNG（最小合法图像，避开第三方图像库编码差异）
            (root / split / "images" / (stem + ".png")).write_bytes(
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00"
                b"\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx"
                b"\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82")
            (root / split / "labels" / (stem + ".txt")).write_text(
                "".join(ln + "\n" for ln in lines), encoding="utf-8")
    return str(root)


class TestFindSplits:
    def test_split_layout(self, tmp_path):
        root = _make_split_dataset(tmp_path / "ds", ["car"], {
            "train": {"a": ["0 0.5 0.5 0.2 0.2"]},
            "valid": {"b": ["0 0.5 0.5 0.2 0.2"]},
            "test": {"c": []},
        })
        assert [s[0] for s in eval_dataset.find_splits(root)] == ["train", "valid", "test"]

    def test_flat_layout(self, tmp_path):
        root = tmp_path / "flat"
        (root / "images").mkdir(parents=True)
        (root / "labels").mkdir(parents=True)
        assert [s[0] for s in eval_dataset.find_splits(str(root))] == ["all"]

    def test_count_dataset(self, tmp_path):
        root = _make_split_dataset(tmp_path / "ds", ["car"], {
            "train": {"a": ["0 0.5 0.5 0.2 0.2", "0 0.1 0.1 0.1 0.1"]},
            "valid": {"b": []},
        })
        rows = eval_dataset.count_dataset(root)
        assert {r["split"]: (r["images"], r["boxes"]) for r in rows} == \
            {"train": (1, 2), "valid": (1, 0)}


class TestBuildEvalDataset:
    def _ds(self, tmp_path):
        return _make_split_dataset(tmp_path / "ds", ["car", "motorbike", "tree"], {
            "train": {"t1": ["0 0.5 0.5 0.4 0.4", "1 0.2 0.2 0.1 0.1"]},
            "valid": {"v1": ["2 0.3 0.3 0.2 0.2"]},
        })

    def _read_label(self, work_dir, stem):
        with open(os.path.join(work_dir, "labels", stem + ".txt"), encoding="utf-8") as f:
            return [ln.split() for ln in f if ln.strip()]

    def test_remaps_to_target_indices(self, tmp_path):
        root = self._ds(tmp_path)
        work = str(tmp_path / "work")
        yaml_path, st = eval_dataset.build_eval_dataset(
            root, ["car", "motorbike", "tree"], ["car", "motorcycle"], [], "all", work)
        assert st["images"] == 2 and st["gt_kept"] == 2
        assert st["unmappable"]["tree"] == 1
        # motorbike(源下标 1) 应被重写为 motorcycle(目标下标 1)，car 保持 0
        assert [r[0] for r in self._read_label(work, "train__t1")] == ["0", "1"]
        assert st["per_class"][0] == 1 and st["per_class"][1] == 1
        assert os.path.isfile(yaml_path)

    def test_bbox_coords_preserved(self, tmp_path):
        root = self._ds(tmp_path)
        work = str(tmp_path / "work")
        eval_dataset.build_eval_dataset(root, ["car", "motorbike", "tree"],
                                        ["car", "motorcycle"], [], "train", work)
        assert self._read_label(work, "t1")[0] == ["0", "0.5", "0.5", "0.4", "0.4"]

    def test_drop_classes(self, tmp_path):
        root = self._ds(tmp_path)
        work = str(tmp_path / "work")
        _, st = eval_dataset.build_eval_dataset(
            root, ["car", "motorbike", "tree"], ["car", "motorbike", "tree"],
            ["tree"], "all", work)
        assert st["gt_dropped"] == 1 and st["gt_kept"] == 2
        assert not st["unmappable"]

    def test_all_dropped_image_becomes_background(self, tmp_path):
        root = self._ds(tmp_path)
        work = str(tmp_path / "work")
        _, st = eval_dataset.build_eval_dataset(
            root, ["car", "motorbike", "tree"], ["car", "motorcycle"], [], "valid", work)
        assert st["bg_images"] == 1 and st["gt_kept"] == 0
        assert self._read_label(work, "v1") == []

    def test_out_of_range_class_index_skipped(self, tmp_path):
        root = _make_split_dataset(tmp_path / "ds", ["car"], {
            "train": {"a": ["0 0.5 0.5 0.2 0.2", "9 0.1 0.1 0.1 0.1"]},
        })
        work = str(tmp_path / "work")
        _, st = eval_dataset.build_eval_dataset(root, ["car"], ["car"], [], "all", work)
        assert st["gt_out_of_range"] == 1 and st["gt_kept"] == 1

    def test_generated_yaml_is_parseable_and_absolute(self, tmp_path):
        root = self._ds(tmp_path)
        work = str(tmp_path / "work")
        yaml_path, _ = eval_dataset.build_eval_dataset(
            root, ["car", "motorbike", "tree"], ["car", "motorcycle"], [], "all", work)
        text = open(yaml_path, encoding="utf-8").read()
        assert "path: " in text and os.path.isabs(text.split("path: ")[1].splitlines()[0])
        assert eval_dataset.load_yaml_names(yaml_path) == ["car", "motorcycle"]

    def test_rename_before_mapping(self, tmp_path):
        """--rename 先把 GT 类别名改对，再映射到词汇表下标（bus→person 场景）"""
        root = self._ds(tmp_path)
        work = str(tmp_path / "work")
        _, st = eval_dataset.build_eval_dataset(
            root, ["car", "motorbike", "tree"], ["person", "car"], [], "all", work,
            rename={"tree": "person"})
        assert st["renamed"]["tree->person"] == 1
        assert st["unmappable"]["motorbike"] == 1  # 目标词汇表里没有 motorcycle
        assert st["gt_kept"] == 2
        assert st["per_class"][0] == 1 and st["per_class"][1] == 1
        assert self._read_label(work, "valid__v1") == [["0", "0.3", "0.3", "0.2", "0.2"]]

    def test_rename_to_name_outside_vocab_is_unmappable(self, tmp_path):
        root = self._ds(tmp_path)
        work = str(tmp_path / "work")
        _, st = eval_dataset.build_eval_dataset(
            root, ["car", "motorbike", "tree"], ["car"], [], "all", work,
            rename={"tree": "person"})
        assert st["unmappable"]["person"] == 1 and st["gt_kept"] == 1

    def test_unknown_split_exits(self, tmp_path):
        root = self._ds(tmp_path)
        with pytest.raises(SystemExit):
            eval_dataset.build_eval_dataset(root, ["car"], ["car"], [], "test",
                                            str(tmp_path / "work"))

