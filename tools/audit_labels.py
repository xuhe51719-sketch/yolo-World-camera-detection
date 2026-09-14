# -*- coding: utf-8 -*-
"""
标签逐框审计：判断某个类别的 GT 框是否「全部」标错，避免全局改名误伤正确标注。

背景：全局 --rename bus=person 这类映射会把少数标对的框一起改坏。本工具对指定类别的
每一个 GT 框单独取证，给出四路证据与一个判定：
    1. 裁剪取证（主证据）：把该框外扩 15% 裁出并放大到 >=320px，分别用两个模型在裁剪图内
       检测"占满裁剪区"的目标——裁剪图里是什么，这个框标的就该是什么；
    2. 全图取证：两个模型在整图上做类别无关 IoU 匹配，看该位置被认成什么；
    3. 几何先验：宽高比与像素面积（行人 w/h≈0.3~0.5，公交车侧视 w/h≈2~4、正视≈0.8~1.2）；
    4. 判定：两模型裁剪一致=HIGH，裁剪+全图一致=HIGH，单模型裁剪=MEDIUM，
       仅全图一致=MEDIUM，其余=AMBIGUOUS（交人工复核）。

输出：
    - 控制台：判定分布汇总（多少框确证为 person / 多少确证仍为 bus / 多少存疑）
    - .eval_cache/<class>_audit/audit.csv：逐框四路证据明细
    - .eval_cache/<class>_audit/sheet_*.jpg：非 person 判定框的拼图，供肉眼复核

用法（在项目根目录运行）：
    ".venv\\Scripts\\python.exe" tools\\audit_labels.py --audit-class bus
    ".venv\\Scripts\\python.exe" tools\\audit_labels.py --audit-class bus --min-conf 0.25
"""
import os
import sys
import io
import csv
import glob
import argparse
import collections

# 被 pytest 导入时不能重包 stdout：TextIOWrapper 会在回收时关掉 pytest 的捕获流
if "pytest" not in sys.modules:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import cv2
from ultralytics import YOLO

from config import BASE_DIR
from eval_dataset import IMG_EXTS, find_splits, load_yaml_names

# 裁剪取证用的候选词汇表（覆盖该场景可能出现的混淆类别）
VOCAB = ["person", "bus", "truck", "car", "bicycle", "motorcycle", "backpack", "umbrella"]
CROP_MIN_SIDE = 320      # 裁剪图放大下限，保证小目标也有足够像素
CROP_PAD = 0.15          # 外扩比例
COVER_MIN = 0.35         # 预测框占裁剪图面积下限（目标应基本占满裁剪区）


def name_of(m, idx):
    """模型类别下标 → 类别名（names 为 dict 或 list 都可用整数下标取）"""
    return m.names[idx]


def iou(a, b):
    """两个归一化 cxcywh 框的 IoU"""
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def crop_verify(crop, models, min_conf):
    """在裁剪图内找"占满裁剪区"的目标，返回 {模型名: (类别, conf, 覆盖率)}"""
    ch, cw = crop.shape[:2]
    scale = max(1.0, CROP_MIN_SIDE / float(min(ch, cw)))
    if scale > 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        ch, cw = crop.shape[:2]
    out = {}
    for name, m in models.items():
        r = m.predict(crop, imgsz=640, conf=min_conf, verbose=False)[0]
        best = None
        for b in (r.boxes if r.boxes is not None else []):
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            cover = (x2 - x1) * (y2 - y1) / float(cw * ch)
            mx, my = (x1 + x2) / 2 / cw, (y1 + y2) / 2 / ch
            if cover < COVER_MIN or not (0.15 <= mx <= 0.85 and 0.15 <= my <= 0.85):
                continue
            cls = name_of(m, int(b.cls[0]))
            if cls not in VOCAB:
                continue
            conf = float(b.conf[0])
            if best is None or conf > best[1]:
                best = (cls, conf, round(cover, 2))
        out[name] = best
    return out


def full_match(gbox, preds, iou_min):
    """整图上与该 GT 框 IoU 最高的预测（类别无关）"""
    best, best_iou = None, 0.0
    for cls, conf, pbox in preds:
        v = iou(gbox, pbox)
        if v > best_iou:
            best, best_iou = (cls, conf), v
    return best if best_iou >= iou_min else None


def verdict(cw_, cm_, fw_, fm_):
    """四路证据合成判定：返回 (类别, 置信档)

    裁剪取证两模型都给出结果但意见冲突时，一律 AMBIGUOUS——
    宁可交人工复核，也不能在证据矛盾时擅自改名误伤正确标注。
    """
    cw = cw_[0] if cw_ else None
    cm = cm_[0] if cm_ else None
    fw = fw_[0] if fw_ else None
    fm = fm_[0] if fm_ else None
    if cw and cm:
        if cw == cm:
            return cw, "HIGH"
        return None, "AMBIGUOUS"
    single = cw or cm
    if single and (single == fw or single == fm):
        return single, "HIGH"
    if single:
        return single, "MEDIUM"
    if fw and fm and fw == fm:
        return fw, "MEDIUM"
    if fw or fm:
        return fw or fm, "LOW"
    return None, "AMBIGUOUS"


def main():
    parser = argparse.ArgumentParser(description="标签逐框审计")
    parser.add_argument("--data",
                        default=os.path.join(BASE_DIR, "datasets",
                                             "world-monitoring-v4-121", "data.yaml"))
    parser.add_argument("--audit-class", default="bus", help="要审计的数据集类别名")
    parser.add_argument("--min-conf", type=float, default=0.15, help="裁剪取证的置信度下限")
    parser.add_argument("--iou", type=float, default=0.5, help="全图取证的 IoU 门槛")
    args = parser.parse_args()

    src_root = os.path.dirname(os.path.abspath(args.data))
    names = load_yaml_names(args.data)
    if args.audit_class not in names:
        print(f"[错误] 数据集类别表里没有 {args.audit_class}，可用: {names}")
        sys.exit(1)
    audit_id = names.index(args.audit_class)

    out_dir = os.path.join(BASE_DIR, ".eval_cache", f"{args.audit_class}_audit")
    os.makedirs(out_dir, exist_ok=True)

    print("加载双模型（开放词汇 + 闭集 COCO）...")
    m_world = YOLO(os.path.join(BASE_DIR, "yolov8x-worldv2.pt"))
    m_world.set_classes(VOCAB)
    m_coco = YOLO(os.path.join(BASE_DIR, "yolov8m.pt"))
    models = {"world": m_world, "coco80m": m_coco}

    rows = []
    crops_for_sheet = []      # (判定, 裁剪图) 用于肉眼复核
    for sp, img_dir, lbl_dir in find_splits(src_root):
        for fn in sorted(os.listdir(img_dir)):
            stem, ext = os.path.splitext(fn)
            if ext.lower() not in IMG_EXTS:
                continue
            lf = os.path.join(lbl_dir, stem + ".txt")
            if not os.path.isfile(lf):
                continue
            gts = []
            with open(lf, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    p = line.split()
                    if len(p) >= 5 and int(float(p[0])) == audit_id:
                        gts.append((i, tuple(float(x) for x in p[1:5])))
            if not gts:
                continue
            img = cv2.imread(os.path.join(img_dir, fn))
            H, W = img.shape[:2]
            # 全图取证：两模型各跑一次
            full_preds = {}
            for name, m in models.items():
                r = m.predict(img, imgsz=640, conf=0.05, verbose=False)[0]
                full_preds[name] = [
                    (name_of(m, int(b.cls[0])),
                     float(b.conf[0]), tuple(float(x) for x in b.xywhn[0]))
                    for b in (r.boxes if r.boxes is not None else [])]
            for li, gbox in gts:
                cx, cy, w, h = gbox
                x1 = max(0, int((cx - w / 2 - CROP_PAD * w) * W))
                y1 = max(0, int((cy - h / 2 - CROP_PAD * h) * H))
                x2 = min(W, int((cx + w / 2 + CROP_PAD * w) * W))
                y2 = min(H, int((cy + h / 2 + CROP_PAD * h) * H))
                crop = img[y1:y2, x1:x2]
                cv = crop_verify(crop, models, args.min_conf) if crop.size else {"world": None, "coco80m": None}
                fw = full_match(gbox, full_preds["world"], args.iou)
                fm = full_match(gbox, full_preds["coco80m"], args.iou)
                cls, conf = verdict(cv["world"], cv["coco80m"], fw, fm)
                rows.append({
                    "split": sp, "image": stem, "line": li,
                    "cx": round(cx, 3), "cy": round(cy, 3),
                    "w": round(w, 3), "h": round(h, 3),
                    "aspect_wh": round(w / h, 2) if h else 0,
                    "side_px": round((w * W * h * H) ** 0.5, 1),
                    "crop_world": (cv["world"] or ("", "", ""))[0],
                    "crop_world_conf": (cv["world"] or ("", 0, ""))[1],
                    "crop_coco80m": (cv["coco80m"] or ("", "", ""))[0],
                    "crop_coco80m_conf": (cv["coco80m"] or ("", 0, ""))[1],
                    "full_world": (fw or ("", 0))[0],
                    "full_coco80m": (fm or ("", 0))[0],
                    "verdict": cls or "AMBIGUOUS", "confidence": conf,
                })
                if (cls or "AMBIGUOUS") != "person":
                    crops_for_sheet.append((cls or "AMBIGUOUS", crop, stem, li))

    csv_path = os.path.join(out_dir, "audit.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n审计类别: {args.audit_class}（下标 {audit_id}） 共 {len(rows)} 框")
    print("=" * 70)
    by_verdict = collections.Counter(r["verdict"] for r in rows)
    for k, v in by_verdict.most_common():
        print(f"  判定为 {k:<12} {v:>4} 框 ({100 * v / len(rows):>5.1f}%)")
    print("-" * 70)
    by_conf = collections.Counter((r["verdict"], r["confidence"]) for r in rows)
    for (k, c), v in sorted(by_conf.items()):
        print(f"  {k:<12} [{c:<8}] {v:>4} 框")
    n_person = by_verdict.get("person", 0)
    n_bus = by_verdict.get(args.audit_class, 0)
    n_amb = by_verdict.get("AMBIGUOUS", 0)
    n_other = len(rows) - n_person - n_bus - n_amb
    print("-" * 70)
    print(f"结论: {n_person} 框确证为 person，{n_bus} 框确证仍为 {args.audit_class}，"
          f"{n_other} 框是其他类别，{n_amb} 框存疑")
    if n_bus + n_amb + n_other:
        print(f"→ 不能全局改名：应只改判定为 person 且置信档为 HIGH/MEDIUM 的框，"
              f"其余 {n_bus + n_amb + n_other} 框需保留或人工复核")
    else:
        print("→ 该类别全部框均确证为 person，可以安全全局改名")

    # 非 person 判定框的拼图，供肉眼复核
    if crops_for_sheet:
        per = 24
        for si in range(0, len(crops_for_sheet), per):
            chunk = crops_for_sheet[si:si + per]
            tiles = []
            for cls, crop, stem, li in chunk:
                t = cv2.resize(crop, (192, 192), interpolation=cv2.INTER_AREA)
                cv2.putText(t, f"{cls}", (4, 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)
                cv2.putText(t, f"{stem[-12:]}#{li}", (4, 186),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 255), 1)
                tiles.append(t)
            while len(tiles) % 6:
                tiles.append(tiles[-1].copy())
            grid = [cv2.hconcat(tiles[r:r + 6]) for r in range(0, len(tiles), 6)]
            sheet = os.path.join(out_dir, f"sheet_{si // per + 1}.jpg")
            cv2.imwrite(sheet, cv2.vconcat(grid))
            print(f"  复核拼图: {sheet}")
    print(f"逐框明细: {csv_path}")


if __name__ == "__main__":
    main()
