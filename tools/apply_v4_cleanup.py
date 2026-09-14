# -*- coding: utf-8 -*-
"""
v4 黄金集清洗：把 Roboflow "v4-121-images-1280-fit-within" 原始导出（18 脏类）
一次性治理为 canonical 124 类，口径与旧 v2 完全一致。

v4 与 v2 是同一 Roboflow 项目 yolo-world-monitoring-system 的不同版本：
  - 几何：v4 用 1280 "Fit within"（保比例、不拉伸），修复了 v2 的几何拉伸瓶颈
          （昨天消融实验定位的 person 精度崩盘根因）。
  - bus→person：v4 已在源头修好——原 135 个 bus 框全部归 person，v4 无 bus 类。
  - 车辆类被改成占位名 '1'/'2'/'3'，经框数精确逆推确认其真身：
        '1'(200) + car(27) = 227 == v2 car(227)
        '2'(157)            == v2 motorbike(157)
        '3'(17)             == v2 truck(17)
    故本脚本把 '1'→car、'2'→motorcycle、'3'→truck 显式重映射，保住全部 374 个车辆框。
  - 伪类 YOLO-World-Monitoring-System 与长尾 box/tree/well lid 一律删除（共 15 框）。

下标基准：config_classes.DETECTION_CLASSES（124 类，COCO 序），与 dataset/data.yaml 同序。
写盘前整体备份原始标注到 .eval_cache/v4_raw_backup/。

用法：
    ".venv\\Scripts\\python.exe" tools\\apply_v4_cleanup.py
"""
import os
import sys
import io
import shutil
import collections

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config import BASE_DIR
from config_classes import DETECTION_CLASSES
from eval_dataset import find_splits, load_yaml_names, canonical

V4 = os.path.join(BASE_DIR, "datasets", "world-monitoring-v4-121")
BACKUP = os.path.join(BASE_DIR, ".eval_cache", "v4_raw_backup")

# 占位类名 -> canonical 车辆类（框数逆推确认，见文件头注释）
PLACEHOLDER_MAP = {
    "1": "car",
    "2": "motorcycle",
    "3": "truck",
}
# 伪类 / 长尾类：整类删除
DROP_CLASSES = {"YOLO-World-Monitoring-System", "box", "tree", "well lid"}


def main():
    yaml_path = os.path.join(V4, "data.yaml")
    if not os.path.isfile(yaml_path):
        raise SystemExit(f"[错误] 未找到 {yaml_path}，请先把 v4 导出复制到 datasets/world-monitoring-v4-121")

    raw_names = load_yaml_names(yaml_path)
    canon_index = {canonical(n): i for i, n in enumerate(DETECTION_CLASSES)}

    # 构建 原始下标 -> canonical 下标 映射；不在 old2new 中的原始下标 = 丢弃
    old2new = {}
    print("类别映射（v4 原始 -> canonical）:")
    for i, n in enumerate(raw_names):
        if n in DROP_CLASSES:
            print(f"  {i:>2} {n:<28} -> [DROP 伪类/长尾]")
            continue
        target = PLACEHOLDER_MAP.get(n, n)
        key = canonical(target)
        if key not in canon_index:
            raise SystemExit(f"[错误] 类别 {n}(->{target}) 不在 DETECTION_CLASSES，无法统一")
        old2new[i] = canon_index[key]
        tag = "  [占位重映射]" if n in PLACEHOLDER_MAP else ""
        print(f"  {i:>2} {n:<28} -> {old2new[i]:>3} {DETECTION_CLASSES[old2new[i]]}{tag}")

    # 备份原始 labels + data.yaml
    if os.path.isdir(BACKUP):
        shutil.rmtree(BACKUP)
    os.makedirs(BACKUP)
    for sp, img_dir, lbl_dir in find_splits(V4):
        shutil.copytree(lbl_dir, os.path.join(BACKUP, sp, "labels"))
    shutil.copy2(yaml_path, os.path.join(BACKUP, "data.yaml"))
    print(f"\n已备份原始标注到 {BACKUP}")

    # 重写所有 label 文件
    hist = collections.Counter()
    dropped = collections.Counter()
    for sp, img_dir, lbl_dir in find_splits(V4):
        for fn in sorted(os.listdir(lbl_dir)):
            if not fn.endswith(".txt"):
                continue
            path = os.path.join(lbl_dir, fn)
            out = []
            for line in open(path, encoding="utf-8").read().splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                ci = int(float(p[0]))
                if ci not in old2new:
                    dropped[raw_names[ci] if ci < len(raw_names) else str(ci)] += 1
                    continue
                ni = old2new[ci]
                hist[ni] += 1
                out.append(" ".join([str(ni)] + p[1:5]))
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(out) + ("\n" if out else ""))

    # 重写 data.yaml 为 canonical 124（与 unify_class_ids.py 输出同格式）
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("train: ../train/images\nval: ../valid/images\ntest: ../test/images\n\n")
        f.write(f"nc: {len(DETECTION_CLASSES)}\nnames:\n")
        for i, n in enumerate(DETECTION_CLASSES):
            f.write(f"  {i}: {n}\n")

    print(f"\ndata.yaml 重写: nc={len(DETECTION_CLASSES)}（canonical 统一下标）")
    print("清洗后有 GT 的类别:")
    for i in sorted(hist):
        print(f"  {i:>3} {DETECTION_CLASSES[i]:<20} {hist[i]}")
    print(f"总保留框数 = {sum(hist.values())}")
    print("删除的伪类/长尾框:")
    for n, c in dropped.most_common():
        print(f"  {n:<28} {c}")
    print(f"删除框合计 = {sum(dropped.values())}")


if __name__ == "__main__":
    main()
