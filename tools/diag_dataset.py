# -*- coding: utf-8 -*-
"""
黄金数据集体检：在跑精度评估之前/之后，先看清数据集本身的质量与统计特性。

mAP 偏低时，先区分是「模型不行」还是「数据集不行」——本工具给出三类证据：
    1. 类别平衡度：每类标注框数（长尾类样本过少时 AP 天然为 0，会拉低 mAP）
    2. 目标尺度分布：GT 框边长在导出图上的像素尺寸（小目标占比过高 → 漏检主因）
    3. 词汇表覆盖度：数据集类别名能否映射到部署词汇表 DETECTION_CLASSES

用法（在项目根目录运行）：
    ".venv\\Scripts\\python.exe" tools\\diag_dataset.py
    ".venv\\Scripts\\python.exe" tools\\diag_dataset.py --data datasets\\world-monitoring-v2-121\\data.yaml
"""
import os
import sys
import io
import argparse
import statistics
import collections

# 被 pytest 导入时不能重包 stdout：TextIOWrapper 会在回收时关掉 pytest 的捕获流
if "pytest" not in sys.modules:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config import BASE_DIR
from config_classes import DETECTION_CLASSES
from eval_dataset import IMG_EXTS, canonical, find_splits, load_yaml_names

# COCO 小目标定义：<32² 像素为 small，32²~96² 为 medium，>96² 为 large
SMALL_PX, LARGE_PX = 32.0, 96.0


def image_size(img_dir):
    """取一张图读出实际分辨率（同一导出批次内尺寸一致）"""
    import cv2
    for fn in sorted(os.listdir(img_dir)):
        if os.path.splitext(fn)[1].lower() in IMG_EXTS:
            im = cv2.imread(os.path.join(img_dir, fn))
            if im is not None:
                return im.shape[1], im.shape[0]
    return 0, 0


def main():
    parser = argparse.ArgumentParser(description="黄金数据集体检")
    parser.add_argument("--data",
                        default=os.path.join(BASE_DIR, "datasets",
                                             "world-monitoring-v2-121", "data.yaml"))
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"[错误] 找不到 {args.data}")
        sys.exit(1)
    src_root = os.path.dirname(os.path.abspath(args.data))
    names = load_yaml_names(args.data)
    splits = find_splits(src_root)
    print(f"数据集: {src_root}")
    print(f"类别数: {len(names)} | 划分: {[s[0] for s in splits]}")

    boxes = collections.defaultdict(list)      # 类别下标 -> [边长 px]
    per_split = collections.Counter()
    per_image = collections.Counter()
    n_img = 0
    w0 = h0 = 0

    for sp, img_dir, lbl_dir in splits:
        W, H = image_size(img_dir)
        w0, h0 = W or w0, H or h0
        for fn in sorted(os.listdir(img_dir)):
            stem, ext = os.path.splitext(fn)
            if ext.lower() not in IMG_EXTS:
                continue
            n_img += 1
            per_split[sp] += 1
            lf = os.path.join(lbl_dir, stem + ".txt")
            if not os.path.isfile(lf):
                continue
            with open(lf, encoding="utf-8") as f:
                for line in f:
                    p = line.split()
                    if len(p) < 5:
                        continue
                    ci = int(float(p[0]))
                    side = ((float(p[3]) * W) * (float(p[4]) * H)) ** 0.5
                    boxes[ci].append(side)
                    per_image[stem] += 1

    allv = [x for v in boxes.values() for x in v]
    if not allv:
        print("[错误] 没有任何标注框，无法体检")
        sys.exit(1)

    print(f"图像: {n_img} 张（{dict(per_split)}） | 分辨率: {w0}x{h0} | 标注框: {len(allv)} 个")
    print(f"每图框数: 均值 {len(allv) / n_img:.2f}  中位 {statistics.median(per_image.values()):.0f}"
          f"  最大 {max(per_image.values())}")

    print("\n=== 1. 类别平衡度与目标尺度（边长 px，按导出图实际分辨率换算）===")
    print(f"{'类别':<28}{'框数':>6}{'占比':>7}{'中位边长':>9}{'最小':>7}{'最大':>7}"
          f"{'small':>7}{'medium':>8}{'large':>7}")
    print("-" * 86)
    for ci in sorted(boxes, key=lambda c: -len(boxes[c])):
        v = sorted(boxes[ci])
        nm = names[ci] if ci < len(names) else f"<越界:{ci}>"
        small = sum(1 for x in v if x < SMALL_PX)
        large = sum(1 for x in v if x > LARGE_PX)
        print(f"{nm:<28}{len(v):>6}{100 * len(v) / len(allv):>6.1f}%"
              f"{statistics.median(v):>9.1f}{v[0]:>7.1f}{v[-1]:>7.1f}"
              f"{small:>7}{len(v) - small - large:>8}{large:>7}")
    small = sum(1 for x in allv if x < SMALL_PX)
    large = sum(1 for x in allv if x > LARGE_PX)
    print("-" * 86)
    print(f"{'合计':<28}{len(allv):>6}{100.0:>6.1f}%{statistics.median(allv):>9.1f}"
          f"{min(allv):>7.1f}{max(allv):>7.1f}{small:>7}{len(allv) - small - large:>8}{large:>7}")
    print(f"  小目标(边长<{SMALL_PX:.0f}px)占比 {100 * small / len(allv):.1f}%  "
          f"中目标 {100 * (len(allv) - small - large) / len(allv):.1f}%  "
          f"大目标(>{LARGE_PX:.0f}px) {100 * large / len(allv):.1f}%")

    tail = [names[c] for c in boxes if len(boxes[c]) <= 5]
    print(f"  长尾类（框数<=5，AP 基本无统计意义）: {len(tail)} 个 → {tail}")

    print("\n=== 2. 部署词汇表 DETECTION_CLASSES 覆盖度 ===")
    proj = {canonical(n) for n in DETECTION_CLASSES}
    hit, miss = [], []
    for ci in sorted(boxes):
        nm = names[ci] if ci < len(names) else f"<越界:{ci}>"
        (hit if canonical(nm) in proj else miss).append((nm, len(boxes[ci])))
    n_hit = sum(n for _, n in hit)
    n_miss = sum(n for _, n in miss)
    print(f"  可覆盖类别 {len(hit)} 个 / {n_hit} 框"
          f"（{100 * n_hit / len(allv):.1f}%）: {[h[0] for h in hit]}")
    print(f"  覆盖不到   {len(miss)} 个 / {n_miss} 框"
          f"（{100 * n_miss / len(allv):.1f}%）: {[(m[0], m[1]) for m in miss]}")
    renamed = [(names[ci], canonical(names[ci])) for ci in boxes
               if ci < len(names) and canonical(names[ci]) != names[ci].strip().lower()]
    if renamed:
        print(f"  经同义词表归一后匹配的类别: {renamed}")

    print("\n=== 3. 判读提示 ===")
    if 100 * small / len(allv) > 40:
        print(f"  [!] 小目标占比 {100 * small / len(allv):.0f}%：Recall 偏低首先怀疑导出分辨率不足，"
              f"而非模型能力。建议 Roboflow 导出时关闭 Resize 或改到 1280，再重跑评估对比。")
    if len(tail) >= max(1, len(boxes) // 2):
        print(f"  [!] {len(tail)}/{len(boxes)} 个类别是长尾类：mAP 会被大量 AP=0 的类平均拉低，"
              f"汇报时应同时给出「仅主要类别」的 mAP。")
    if n_miss:
        print(f"  [!] {n_miss} 个标注框的类别不在部署词汇表内，线上系统结构上不可能检出，"
              f"需要补进 config_classes.DETECTION_CLASSES 或从数据集剔除。")


if __name__ == "__main__":
    main()
