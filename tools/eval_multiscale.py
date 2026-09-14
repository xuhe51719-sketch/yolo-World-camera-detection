# -*- coding: utf-8 -*-
"""
多尺度推理评估 harness（评估侧，不改部署）：量化「单尺度 vs 多尺度合并」的精度收益与延迟代价。

动机：v4 黄金集上 person 多为极大特写框，imgsz=1280 会漏检崩盘（640 正常），而 car/moto
等常规尺度目标在 1280 更好。多尺度推理 = 同一张图跑多个 imgsz，把各尺度检测框并起来做
跨尺度类别感知 NMS，理论上能「person 取 640、车辆取 1280」两头兼顾。但代价是成倍推理延迟，
对实时监控是硬约束——本工具把精度与延迟一并量出来，供取舍。

对每种策略输出：总体 mAP50、主要类别 mAP50、person/car/motorcycle/truck 各自 AP50、
以及每帧推理延迟(ms)与等效 FPS。策略 = 每个单尺度 + 全尺度合并。

用法（项目根目录）：
    ".venv\\Scripts\\python.exe" tools\\eval_multiscale.py --scales 640,1280
    ".venv\\Scripts\\python.exe" tools\\eval_multiscale.py --model yolov8m.pt --scales 640,960,1280
"""
import os
import sys
import io
import time
import argparse
import collections

import numpy as np

if "pytest" not in sys.modules:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from torchvision.ops import batched_nms
from ultralytics import YOLO

from config import BASE_DIR, settings
from config_classes import DETECTION_CLASSES
from eval_dataset import find_splits, load_yaml_names, IMG_EXTS, EVAL_REPORT_DIR

MIN_GT_DEFAULT = 10   # 主要类别门槛，与 eval_dataset 一致


def _iou_xyxy(a, b):
    """两个归一化 xyxy 框的 IoU"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _voc_ap(rec, prec):
    """全点插值 AP（VOC 风格：精度包络线下面积）"""
    if len(rec) == 0:
        return 0.0
    mrec = np.concatenate(([0.0], np.asarray(rec, dtype=float), [1.0]))
    mpre = np.concatenate(([0.0], np.asarray(prec, dtype=float), [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def compute_ap50(preds_by_img, gt_by_img, iou_thr=0.5):
    """按类别名计算 AP50。preds_by_img: {img: [(name, conf, xyxy)]}; gt_by_img: {img: {name: [xyxy,...]}}
    返回 {name: (ap, n_gt)}"""
    classes = sorted({n for g in gt_by_img.values() for n in g})
    out = {}
    for cls in classes:
        n_gt = sum(len(g.get(cls, [])) for g in gt_by_img.values())
        if n_gt == 0:
            continue
        dets = [(conf, img, box) for img, pl in preds_by_img.items()
                for (name, conf, box) in pl if name == cls]
        dets.sort(key=lambda x: -x[0])
        tp = np.zeros(len(dets))
        fp = np.zeros(len(dets))
        matched = set()
        for k, (_, img, box) in enumerate(dets):
            gts = gt_by_img.get(img, {}).get(cls, [])
            best_iou, best_j = 0.0, -1
            for j, gbox in enumerate(gts):
                v = _iou_xyxy(box, gbox)
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_iou >= iou_thr and (img, best_j) not in matched:
                matched.add((img, best_j))
                tp[k] = 1
            else:
                fp[k] = 1
        ctp, cfp = np.cumsum(tp), np.cumsum(fp)
        rec = ctp / n_gt
        prec = ctp / np.maximum(ctp + cfp, 1e-9)
        out[cls] = (_voc_ap(rec, prec), n_gt)
    return out


def nms_merge(per_img_preds, name2idx, iou_thr=0.6):
    """对单张图的多尺度预测并集做类别感知 NMS。per_img_preds: [(name,conf,xyxy)] -> 去重后列表"""
    if not per_img_preds:
        return []
    boxes = torch.tensor([p[2] for p in per_img_preds], dtype=torch.float32)
    scores = torch.tensor([p[1] for p in per_img_preds], dtype=torch.float32)
    idxs = torch.tensor([name2idx.get(p[0], 0) for p in per_img_preds], dtype=torch.int64)
    keep = batched_nms(boxes, scores, idxs, iou_thr)
    return [per_img_preds[i] for i in keep.tolist()]


def collect_images(src_root):
    """返回 [(img_key, img_path, lbl_path)]"""
    items = []
    for sp, img_dir, lbl_dir in find_splits(src_root):
        for fn in sorted(os.listdir(img_dir)):
            if os.path.splitext(fn)[1].lower() not in IMG_EXTS:
                continue
            stem = os.path.splitext(fn)[0]
            items.append((os.path.join(sp, stem), os.path.join(img_dir, fn),
                          os.path.join(lbl_dir, stem + ".txt")))
    return items


def read_gt(lbl_path, names):
    """YOLO txt（canonical 下标）-> {name: [xyxy 归一化,...]}"""
    g = collections.defaultdict(list)
    if not os.path.isfile(lbl_path):
        return g
    for line in open(lbl_path, encoding="utf-8").read().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        ci = int(float(p[0]))
        if ci < 0 or ci >= len(names):
            continue
        cx, cy, w, h = (float(x) for x in p[1:5])
        g[names[ci]].append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return g


def main():
    ap = argparse.ArgumentParser(description="多尺度推理：精度 vs 延迟评估")
    ap.add_argument("--data", default=os.path.join(BASE_DIR, "datasets",
                                                    "world-monitoring-v4-121", "data.yaml"))
    ap.add_argument("--model", default=settings.model_path,
                    help="默认用部署模型（settings.model_path）以反映真实延迟")
    ap.add_argument("--scales", default="640,1280", help="逗号分隔的 imgsz 列表")
    ap.add_argument("--conf", type=float, default=0.001, help="低阈值走完整 PR 曲线算 mAP")
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--iou-match", type=float, default=0.5, help="算 AP 的 IoU 匹配门槛")
    ap.add_argument("--iou-nms", type=float, default=0.6, help="跨尺度合并的 NMS IoU 门槛")
    ap.add_argument("--min-gt", type=int, default=MIN_GT_DEFAULT)
    ap.add_argument("--half", type=int, default=1 if settings.use_half else 0,
                    help="FP16 半精度（与部署一致默认开）")
    ap.add_argument("--split", default="all")
    args = ap.parse_args()

    scales = [int(s) for s in args.scales.split(",") if s.strip()]
    src_root = os.path.dirname(os.path.abspath(args.data))
    src_names = load_yaml_names(args.data)          # canonical 124
    half = bool(args.half) and torch.cuda.is_available()

    m = YOLO(args.model)
    is_world = "world" in os.path.basename(args.model).lower()
    if is_world:
        m.set_classes(list(DETECTION_CLASSES))
        pred_names = list(DETECTION_CLASSES)
    else:
        pred_names = [m.names[i] for i in sorted(m.names)]
    name2idx = {n: i for i, n in enumerate(DETECTION_CLASSES)}

    imgs = [it for it in collect_images(src_root)
            if args.split == "all" or it[0].startswith(args.split)]
    print(f"模型={os.path.basename(args.model)}（{'开放词汇' if is_world else 'COCO闭集'}） "
          f"half={half} scales={scales} conf={args.conf}")
    print(f"数据集 {src_root} | {len(imgs)} 张图 | GT 类别表 {len(src_names)}")

    # GT（canonical 名 -> xyxy）
    gt_by_img = {key: read_gt(lbl, src_names) for key, _, lbl in imgs}

    # 逐尺度预测 + 计时
    preds_by_scale = {s: {} for s in scales}
    latency = {s: [] for s in scales}
    for s in scales:
        for i, (key, img_path, _) in enumerate(imgs):
            t0 = time.time()
            res = m.predict(img_path, imgsz=s, conf=args.conf, max_det=args.max_det,
                            half=half, verbose=False)[0]
            dt = time.time() - t0
            if i > 0:                       # 跳过首张（模型预热）
                latency[s].append(dt)
            pl = []
            if res.boxes is not None and len(res.boxes):
                for b in res.boxes:
                    ci = int(b.cls[0])
                    nm = pred_names[ci] if ci < len(pred_names) else str(ci)
                    xy = [float(x) for x in b.xyxyn[0]]
                    pl.append((nm, float(b.conf[0]), xy))
            preds_by_scale[s][key] = pl
        print(f"  尺度 {s}: 平均推理 {1000*np.mean(latency[s]):.1f} ms/帧 "
              f"（≈{1.0/np.mean(latency[s]):.1f} FPS，仅推理）")

    # 合并策略：各尺度并集 -> 跨尺度类别感知 NMS
    merged = {}
    t_merge = []
    for key, _, _ in imgs:
        union = []
        for s in scales:
            union.extend(preds_by_scale[s][key])
        t0 = time.time()
        merged[key] = nms_merge(union, name2idx, args.iou_nms)
        t_merge.append(time.time() - t0)

    # 各策略 mAP
    strategies = {f"single_{s}": preds_by_scale[s] for s in scales}
    strategies[f"merge_{'_'.join(str(s) for s in scales)}"] = merged

    rows = []
    print("\n" + "=" * 92)
    hdr = f"{'策略':<20}{'总mAP50':>9}{'主要类':>9}{'person':>9}{'car':>8}{'moto':>8}{'truck':>8}{'延迟ms':>9}{'FPS':>7}"
    print(hdr)
    print("-" * 92)
    for name, preds in strategies.items():
        aps = compute_ap50(preds, gt_by_img, args.iou_match)
        all_ap = [v[0] for v in aps.values()]
        main_ap = [v[0] for v in aps.values() if v[1] >= args.min_gt]
        def g(c):
            return aps.get(c, (0.0, 0))[0]
        if name.startswith("single_"):
            s = int(name.split("_")[1])
            lat = float(np.mean(latency[s])) * 1000
        else:
            lat = sum(float(np.mean(latency[s])) for s in scales) * 1000 + float(np.mean(t_merge)) * 1000
        fps = 1000.0 / lat if lat > 0 else 0.0
        rows.append({"strategy": name, "mAP50_all": round(float(np.mean(all_ap)), 4) if all_ap else 0.0,
                     "mAP50_main": round(float(np.mean(main_ap)), 4) if main_ap else 0.0,
                     "person": round(g("person"), 4), "car": round(g("car"), 4),
                     "motorcycle": round(g("motorcycle"), 4), "truck": round(g("truck"), 4),
                     "latency_ms": round(lat, 1), "fps": round(fps, 1)})
        r = rows[-1]
        print(f"{name:<20}{r['mAP50_all']:>9.3f}{r['mAP50_main']:>9.3f}{r['person']:>9.3f}"
              f"{r['car']:>8.3f}{r['motorcycle']:>8.3f}{r['truck']:>8.3f}{r['latency_ms']:>9.1f}{r['fps']:>7.1f}")
    print("=" * 92)
    print("注：延迟为单帧纯推理（不含读流/绘制/跟踪）；合并策略延迟≈各尺度之和+NMS。"
          "部署每 detection_interval 帧才推理一次，可据此折算实际帧率预算。")

    # 写报告到 eval_reports/
    os.makedirs(EVAL_REPORT_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(EVAL_REPORT_DIR, f"multiscale_{os.path.basename(args.model)}_{ts}.csv")
    import csv
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["# 多尺度推理评估", f"模型={os.path.basename(args.model)}", f"half={half}",
                    f"scales={scales}", f"conf={args.conf}", f"iou_nms={args.iou_nms}",
                    f"时间={time.strftime('%Y-%m-%d %H:%M:%S')}"])
        w.writerow([])
        cols = ["strategy", "mAP50_all", "mAP50_main", "person", "car", "motorcycle",
                "truck", "latency_ms", "fps"]
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])
    print(f"报告已写入: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
