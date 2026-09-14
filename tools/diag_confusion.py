# -*- coding: utf-8 -*-
"""
类别混淆诊断：按 IoU 做「类别无关」匹配，看模型把每个 GT 类别认成了什么。

用途：当某类 Recall 异常低、但目标尺度并不小的时候，用它区分两种根因——
    A. 标注语义/命名错位：GT 叫 bus，模型一致地叫它 truck/car → 是数据集标签口径问题
    B. 真实漏检：GT 框压根没有匹配到任何预测（"未检出"占多数）→ 是模型/分辨率问题

用法（在项目根目录运行）：
    ".venv\\Scripts\\python.exe" tools\\diag_confusion.py
    ".venv\\Scripts\\python.exe" tools\\diag_confusion.py --vocab project --conf 0.05 --iou 0.5
"""
import os
import sys
import io
import time
import argparse
import collections

# 被 pytest 导入时不能重包 stdout：TextIOWrapper 会在回收时关掉 pytest 的捕获流
if "pytest" not in sys.modules:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ultralytics import YOLO

from config import BASE_DIR
from config_classes import DETECTION_CLASSES
from eval_dataset import build_eval_dataset, canonical, load_yaml_names


def iou_xywh(a, b):
    """两个归一化 cxcywh 框的 IoU"""
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def read_labels(path):
    """读 YOLO txt → [(cls, cx, cy, w, h)]"""
    out = []
    if not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 5:
                out.append((int(float(p[0])),) + tuple(float(x) for x in p[1:5]))
    return out


def main():
    parser = argparse.ArgumentParser(description="GT 类别 vs 模型识别结果 混淆诊断")
    parser.add_argument("--data",
                        default=os.path.join(BASE_DIR, "datasets",
                                             "world-monitoring-v4-121", "data.yaml"))
    parser.add_argument("--model", default=os.path.join(BASE_DIR, "yolov8x-worldv2.pt"))
    parser.add_argument("--vocab", choices=["dataset", "project", "model"], default="dataset")
    parser.add_argument("--split", choices=["all", "train", "valid", "test"], default="all")
    parser.add_argument("--drop-classes", default="")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.05,
                        help="诊断用低阈值，尽量不漏掉模型的候选框")
    parser.add_argument("--iou", type=float, default=0.5, help="判定同一个目标所需的 IoU 门槛")
    parser.add_argument("--topk", type=int, default=4, help="每个 GT 类别最多列出的混淆去向数")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    src_root = os.path.dirname(os.path.abspath(args.data))
    src_names = load_yaml_names(args.data)
    drop_names = [x.strip() for x in args.drop_classes.split(",") if x.strip()]

    m = YOLO(args.model)
    if "world" in os.path.basename(args.model).lower():
        if args.vocab == "dataset":
            target = [n for n in src_names
                      if canonical(n) not in {canonical(d) for d in drop_names}]
        elif args.vocab == "project":
            target = list(DETECTION_CLASSES)
        else:
            target = [m.names[i] for i in sorted(m.names)]
        m.set_classes(target)
    else:
        target = [m.names[i] for i in sorted(m.names)]

    tag = args.tag or f"confusion_{args.vocab}_{args.split}"
    work_dir = os.path.join(BASE_DIR, ".eval_cache", tag)
    yaml_path, st = build_eval_dataset(src_root, src_names, target, drop_names,
                                       args.split, work_dir)
    img_dir = os.path.join(work_dir, "images")
    lbl_dir = os.path.join(work_dir, "labels")
    print(f"模型={os.path.basename(args.model)} vocab={args.vocab}({len(target)}类) "
          f"split={args.split} imgsz={args.imgsz} conf={args.conf} iou>={args.iou}")
    print(f"评估集 {st['images']} 张 / 有效 GT {st['gt_kept']} 框 | 缓存 {yaml_path}")

    # 混淆矩阵：GT 类别 -> {模型给出的类别 or "未检出": 次数}
    confuse = collections.defaultdict(collections.Counter)
    fp = collections.Counter()          # 没有任何 GT 与之匹配的预测类别（误报）
    t0 = time.time()
    for fn in sorted(os.listdir(img_dir)):
        stem = os.path.splitext(fn)[0]
        gts = read_labels(os.path.join(lbl_dir, stem + ".txt"))
        res = m.predict(os.path.join(img_dir, fn), imgsz=args.imgsz, conf=args.conf,
                        verbose=False)[0]
        preds = []
        if res.boxes is not None and len(res.boxes):
            for b in res.boxes:
                preds.append((int(b.cls[0]), float(b.conf[0]),
                              tuple(float(x) for x in b.xywhn[0])))
        preds.sort(key=lambda x: -x[1])          # 高分优先占位
        used = set()
        for gt in gts:
            gci, gbox = gt[0], gt[1:]
            best, best_iou = None, 0.0
            for pi, (pci, pconf, pbox) in enumerate(preds):
                if pi in used:
                    continue
                v = iou_xywh(gbox, pbox)
                if v > best_iou:
                    best, best_iou = pi, v
            if best is not None and best_iou >= args.iou:
                used.add(best)
                confuse[gci][target[preds[best][0]]] += 1
            else:
                confuse[gci]["<<未检出>>"] += 1
        for pi, (pci, _, _) in enumerate(preds):
            if pi not in used:
                fp[target[pci]] += 1
    dt = time.time() - t0

    print(f"\n推理完成（{dt:.1f}s）")
    print("=" * 78)
    print("每个 GT 类别被模型识别成了什么（按 IoU 类别无关匹配）")
    print("=" * 78)
    order = sorted(confuse, key=lambda c: -sum(confuse[c].values()))
    for gci in order:
        row = confuse[gci]
        tot = sum(row.values())
        gname = target[gci] if gci < len(target) else str(gci)
        missed = row.get("<<未检出>>", 0)
        print(f"\n[GT] {gname}  共 {tot} 框 | 匹配到 {tot - missed} | 未检出 {missed}"
              f"（{100 * missed / tot:.0f}%）")
        shown = [(k, v) for k, v in row.most_common() if k != "<<未检出>>"][:args.topk]
        for k, v in shown:
            agree = "← 一致" if canonical(k) == canonical(gname) else "← 不一致"
            print(f"      识别为 {k:<24} {v:>4} 框 ({100 * v / tot:>5.1f}%)  {agree}")

    print("\n" + "=" * 78)
    print(f"未匹配到任何 GT 的预测（误报）Top 15，合计 {sum(fp.values())} 框")
    print("=" * 78)
    for k, v in fp.most_common(15):
        print(f"      {k:<28}{v:>5}")

    # 一致性汇总：只看「匹配上的」那部分，模型叫对名字的比例
    agree = disagree = 0
    for gci, row in confuse.items():
        gname = target[gci] if gci < len(target) else str(gci)
        for k, v in row.items():
            if k == "<<未检出>>":
                continue
            if canonical(k) == canonical(gname):
                agree += v
            else:
                disagree += v
    tot = agree + disagree
    print("\n" + "-" * 78)
    if tot:
        print(f"定位成功（IoU>={args.iou}）的 {tot} 框中：类别名一致 {agree} 框"
              f"（{100 * agree / tot:.1f}%），名字被叫错 {disagree} 框（{100 * disagree / tot:.1f}%）")
        print("→ 若「叫错」占比高，说明主要瓶颈是标签口径/词汇表命名，不是检测能力；"
              "应先统一类别名再谈调阈值。")
        print("→ 若「未检出」占比高，才是真漏检，优先查目标尺度与输入分辨率。")


if __name__ == "__main__":
    main()
