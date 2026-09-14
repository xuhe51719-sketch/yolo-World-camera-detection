# -*- coding: utf-8 -*-
"""
[历史脚本 / 已归档] 本脚本作用于旧黄金集 world-monitoring-v2-121（拉伸几何），该集已被
1280 fit-within 的 world-monitoring-v4-121 取代并删除；v2 原始标注备份见
.eval_cache/v2_pre_bigfix_backup，清洗后状态见 git 历史；v4 的等价清洗脚本为
tools/apply_v4_cleanup.py。保留本文件仅为复现 v2 迁移过程，V2 路径常量已失效属预期。

黄金数据集大改：在 datasets/world-monitoring-v2-121 上一次性完成
  1) bus→person 定向改名（118 审计确证 + 3 机器误判 + 8 存疑判人 = 129）；
  2) 删除 3 个不可读框；
  3) 删除伪类 YOLO-World-Monitoring-System 与 tree/box/well lid 的全部框并重排类别下标。
写盘前整体备份到 .eval_cache/v2_pre_bigfix_backup/。

存疑 11 框在 512 分辨率下不可读、且复核拼图标签与文件名无法可靠对应，
为免引入新错标，全部丢弃（后续可在 1280 重审计时回收）。
"""
import os
import sys
import io
import csv
import shutil
import collections

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config import BASE_DIR
from eval_dataset import find_splits, load_yaml_names

V2 = os.path.join(BASE_DIR, "datasets", "world-monitoring-v2-121")
AUDIT = os.path.join(BASE_DIR, ".eval_cache", "bus_audit", "audit.csv")
BACKUP = os.path.join(BASE_DIR, ".eval_cache", "v2_pre_bigfix_backup")

DROP_CLASSES = ["YOLO-World-Monitoring-System", "box", "tree", "well lid"]

# 机器误判、肉眼行人（nr_678/nr_704/nr_017 的第 0 行）
MACHINE_WRONG_PREFIX = ("nr_678", "nr_704", "nr_017")


def main():
    names = load_yaml_names(os.path.join(V2, "data.yaml"))
    bus_id = names.index("bus")
    drop_ids = {names.index(c) for c in DROP_CLASSES}

    rows = list(csv.DictReader(open(AUDIT, encoding="utf-8-sig")))
    rename, drop = set(), set()
    for r in rows:
        img, li = r["image"], int(r["line"])
        if r["verdict"] == "person":
            rename.add((img, li))
        elif r["verdict"] in ("bus", "cat") and li == 0 \
                and img.startswith(MACHINE_WRONG_PREFIX):
            rename.add((img, li))
    amb = [r for r in rows if r["verdict"] == "AMBIGUOUS"]
    for r in amb:
        drop.add((r["image"], int(r["line"])))
    print(f"改名白名单 {len(rename)} 框（期望 121）| 丢弃 {len(drop)} 框（期望 11）")
    assert len(rename) == 121 and len(drop) == 11, "白名单数量不符，中止"

    # 备份
    if os.path.isdir(BACKUP):
        shutil.rmtree(BACKUP)
    os.makedirs(BACKUP)
    for sp, img_dir, lbl_dir in find_splits(V2):
        dst = os.path.join(BACKUP, sp)
        shutil.copytree(lbl_dir, os.path.join(dst, "labels"))
    shutil.copy2(os.path.join(V2, "data.yaml"), os.path.join(BACKUP, "data.yaml"))
    print(f"已备份到 {BACKUP}")

    # 新类别表：去掉 DROP_CLASSES，追加 person
    keep = [n for n in names if n not in DROP_CLASSES]
    old2new = {i: keep.index(n) for i, n in enumerate(names) if n not in DROP_CLASSES}
    person_id = len(keep)
    new_names = keep + ["person"]

    hist_before, hist_after = collections.Counter(), collections.Counter()
    for sp, img_dir, lbl_dir in find_splits(V2):
        for fn in sorted(os.listdir(lbl_dir)):
            if not fn.endswith(".txt"):
                continue
            stem = fn[:-4]
            path = os.path.join(lbl_dir, fn)
            lines = open(path, encoding="utf-8").read().splitlines()
            out = []
            for li, line in enumerate(lines):
                p = line.split()
                if len(p) < 5:
                    continue
                ci = int(float(p[0]))
                hist_before[ci] += 1
                if (stem, li) in drop:
                    continue
                if ci in drop_ids:
                    continue
                if (stem, li) in rename:
                    assert ci == bus_id, f"{stem}#{li} 类别 {ci} 非 bus，中止"
                    ni = person_id
                else:
                    ni = old2new[ci]
                hist_after[ni] += 1
                out.append(" ".join([str(ni)] + p[1:5]))
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(out) + ("\n" if out else ""))

    yaml_path = os.path.join(V2, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("train: ../train/images\nval: ../valid/images\ntest: ../test/images\n\n")
        f.write(f"nc: {len(new_names)}\nnames: {new_names}\n")
    print(f"data.yaml 重写: nc={len(new_names)}")
    print("改后直方图:")
    for i, n in enumerate(new_names):
        print(f"  {i:>2} {n:<28} {hist_after.get(i, 0)}")


if __name__ == "__main__":
    main()
