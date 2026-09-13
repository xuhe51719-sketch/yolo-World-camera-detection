# -*- coding: utf-8 -*-
"""
统一黄金数据集类别下标：把 datasets/world-monitoring-v2-121 的紧凑类别表
重映射到全项目权威下标 config_classes.DETECTION_CLASSES（与 dataset/data.yaml 同序）。

动机：ultralytics 的 val() 按类别下标匹配 GT 与预测。统一后黄金集 GT 下标
== 部署词汇表下标 == 124 类采集集下标，评估无需再做任何映射，
--vocab dataset 与 --vocab project 口径自然重合。
顺带把 Roboflow 习惯名归一到 canonical 名（motorbike→motorcycle、sofa→couch、
table→dining table、fridge→refrigerator）。
写盘前备份到 .eval_cache/v2_pre_unify_backup/。
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

V2 = os.path.join(BASE_DIR, "datasets", "world-monitoring-v2-121")
BACKUP = os.path.join(BASE_DIR, ".eval_cache", "v2_pre_unify_backup")


def main():
    old_names = load_yaml_names(os.path.join(V2, "data.yaml"))
    canon_index = {canonical(n): i for i, n in enumerate(DETECTION_CLASSES)}
    old2new = {}
    for i, n in enumerate(old_names):
        key = canonical(n)
        if key not in canon_index:
            raise SystemExit(f"[错误] 类别 {n} 不在 DETECTION_CLASSES 中，无法统一")
        old2new[i] = canon_index[key]
    print("下标映射（旧→新）:")
    for i, n in enumerate(old_names):
        print(f"  {i:>2} {n:<20} -> {old2new[i]:>3} {DETECTION_CLASSES[old2new[i]]}")

    if os.path.isdir(BACKUP):
        shutil.rmtree(BACKUP)
    os.makedirs(BACKUP)
    for sp, img_dir, lbl_dir in find_splits(V2):
        shutil.copytree(lbl_dir, os.path.join(BACKUP, sp, "labels"))
    shutil.copy2(os.path.join(V2, "data.yaml"), os.path.join(BACKUP, "data.yaml"))
    print(f"已备份到 {BACKUP}")

    hist = collections.Counter()
    for sp, img_dir, lbl_dir in find_splits(V2):
        for fn in sorted(os.listdir(lbl_dir)):
            if not fn.endswith(".txt"):
                continue
            path = os.path.join(lbl_dir, fn)
            out = []
            for line in open(path, encoding="utf-8").read().splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                ni = old2new[int(float(p[0]))]
                hist[ni] += 1
                out.append(" ".join([str(ni)] + p[1:5]))
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(out) + ("\n" if out else ""))

    with open(os.path.join(V2, "data.yaml"), "w", encoding="utf-8") as f:
        f.write("train: ../train/images\nval: ../valid/images\ntest: ../test/images\n\n")
        f.write(f"nc: {len(DETECTION_CLASSES)}\nnames:\n")
        for i, n in enumerate(DETECTION_CLASSES):
            f.write(f"  {i}: {n}\n")
    print(f"data.yaml 重写: nc={len(DETECTION_CLASSES)}（统一下标）")
    print("有 GT 的类别:")
    for i in sorted(hist):
        print(f"  {i:>3} {DETECTION_CLASSES[i]:<20} {hist[i]}")


if __name__ == "__main__":
    main()
