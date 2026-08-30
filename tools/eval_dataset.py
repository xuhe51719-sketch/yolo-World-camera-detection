# -*- coding: utf-8 -*-
"""
黄金数据集精度评估：在人工标注的数据上计算 mAP50 / mAP50-95 / 各类别 Precision/Recall。
这是衡量识别准确率的权威指标（比主观感受可靠），每次改模型/词汇表/阈值后重跑对比。

前置条件：
    1. 已用 tools/capture_frames.py 采集图片
    2. 已按 dataset/README_标注流程.md 完成人工标注（images/ 与 labels/ 配对）

用法（在项目根目录运行）：
    ".venv\\Scripts\\python.exe" tools\\eval_dataset.py --data dataset\\data.yaml
    ".venv\\Scripts\\python.exe" tools\\eval_dataset.py --data dataset\\data.yaml --imgsz 960 --conf 0.2

输出：
    - 控制台打印总体指标与各类别 P/R/mAP50 表格
    - evaluation_report.csv 便于多次实验横向对比
"""
import os
import sys
import io
import csv
import time
import argparse

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ultralytics import YOLO

from config import BASE_DIR  # 项目根已在上方 sys.path.insert 中加入


def load_yaml_names(path):
    """简单解析 data.yaml 的 names 字段（避免引入 yaml 依赖）"""
    names = []
    in_names = False
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s.startswith("names:"):
                in_names = True
                continue
            if in_names:
                if ":" in s and s.split(":")[0].strip().isdigit():
                    names.append(s.split(":", 1)[1].strip())
                elif s and not s.startswith("#"):
                    break
    return names


def main():
    parser = argparse.ArgumentParser(description="黄金数据集 mAP 评估")
    parser.add_argument("--data", default=os.path.join(BASE_DIR, "dataset", "data.yaml"))
    parser.add_argument("--model", default=os.path.join(BASE_DIR, "yolov8x-worldv2.pt"))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--out", default=os.path.join(BASE_DIR, "evaluation_report.csv"))
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"[错误] 找不到 {args.data}。请先运行 tools/capture_frames.py 采集并标注")
        sys.exit(1)

    names = load_yaml_names(args.data)
    labels_dir = os.path.join(os.path.dirname(args.data), "labels")
    n_labeled = len([f for f in os.listdir(labels_dir) if f.endswith(".txt")]) if os.path.isdir(labels_dir) else 0
    if n_labeled == 0:
        print("[错误] labels/ 里没有标注文件。请先完成人工标注再评估")
        sys.exit(1)
    print(f"数据集: {args.data} | 已标注 {n_labeled} 张 | 类别 {len(names)} 个")

    m = YOLO(args.model)
    if "world" in args.model.lower():
        # 开放词汇模型：注册数据集中实际标注的类别
        m.set_classes(names)
        print(f"已注册开放词汇类别 {len(names)} 个")

    print("开始评估（首次运行较慢，含模型预热）...")
    t0 = time.time()
    results = m.val(data=args.data, imgsz=args.imgsz, conf=args.conf,
                    verbose=False, plots=False)
    dt = time.time() - t0

    # 总体指标
    box = results.box
    print("=" * 64)
    print(f"评估完成（{dt:.1f}s）  imgsz={args.imgsz} conf={args.conf} 模型={args.model}")
    print(f"  mAP50-95 : {box.map:.4f}   ← 综合精度（最严格）")
    print(f"  mAP50    : {box.map50:.4f}   ← 常用精度指标")
    print(f"  mAP75    : {box.map75:.4f}")
    print(f"  Precision: {box.mp:.4f}   Recall: {box.mr:.4f}")
    print("=" * 64)

    # 各类别明细
    print(f"{'类别':<20}{'P':>8}{'R':>8}{'mAP50':>8}{'mAP50-95':>10}")
    print("-" * 54)
    class_rows = []
    for i, name in enumerate(names):
        if i >= len(box.ap50):
            break
        p = float(box.p[i]) if i < len(box.p) else 0.0
        r = float(box.r[i]) if i < len(box.r) else 0.0
        ap50 = float(box.ap50[i])
        ap = float(box.ap[i])
        class_rows.append({"class": name, "precision": round(p, 4),
                           "recall": round(r, 4), "mAP50": round(ap50, 4),
                           "mAP50-95": round(ap, 4)})
        if ap50 > 0 or ap > 0:  # 只打印有真值的类别
            print(f"{name:<20}{p:>8.3f}{r:>8.3f}{ap50:>8.3f}{ap:>10.3f}")

    # 追加写入 CSV，便于多次实验对比
    write_header = not os.path.exists(args.out)
    with open(args.out, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["时间", "模型", "imgsz", "conf", "mAP50-95", "mAP50",
                             "precision", "recall", "类别明细"])
        writer.writerow([time.strftime("%Y-%m-%d %H:%M"), args.model, args.imgsz,
                         args.conf, round(float(box.map), 4), round(float(box.map50), 4),
                         round(float(box.mp), 4), round(float(box.mr), 4),
                         "; ".join(f"{c['class']}:{c['mAP50']}" for c in class_rows if c["mAP50"] > 0)])
    print("-" * 54)
    print(f"报告已追加到: {os.path.abspath(args.out)}")
    print("调参建议: 某类 Recall 低→该类别样本少或被漏检(加采该类图/降conf)；")
    print("          Precision 低→误报多(提高该类 conf 或从词汇表移除相近干扰词)")


if __name__ == "__main__":
    main()
