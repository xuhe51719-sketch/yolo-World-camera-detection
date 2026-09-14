# -*- coding: utf-8 -*-
"""
黄金数据集精度评估：在人工标注的数据上计算 mAP50 / mAP50-95 / 各类别 Precision/Recall。
这是衡量识别准确率的权威指标（比主观感受可靠），每次改模型/词汇表/阈值后重跑对比。

支持两种数据布局：
    1. 扁平式（dataset/：images/ + labels/ + data.yaml，names 为 "0: person" 映射式）
    2. split 式（datasets/world-monitoring-v4-121/：train|valid|test 各带 images/ labels/，
       names 为 ['a', 'b'] 列表式，且 data.yaml 里是 ../train/images 这类相对路径）

三种词汇表口径（--vocab）：
    dataset  用数据集自带类别名做开放词汇（标准 mAP 口径，类别下标天然对齐）
    project  用部署词汇表 config_classes.DETECTION_CLASSES（衡量线上系统真实表现）
    model    用模型内置类别（yolov8m/n 的固定 COCO 80 类，做常规模型基线）
GT 类别名先经 CLASS_SYNONYMS 归一（motorbike→motorcycle 等）再映射到目标词汇表下标，
映射不到的类别计入统计并丢弃，避免"模型类别下标 ≠ 数据集类别下标"导致的错配。

用法（在项目根目录运行）：
    ".venv\\Scripts\\python.exe" tools\\eval_dataset.py --vocab dataset --split all
    ".venv\\Scripts\\python.exe" tools\\eval_dataset.py --vocab project --split all
    ".venv\\Scripts\\python.exe" tools\\eval_dataset.py --model yolov8m.pt --vocab model

输出（统一收纳到 eval_reports/ 专用文件夹，项目根目录不再散落报告）：
    - 控制台打印总体指标与各类别 P/R/mAP50 表格
    - eval_reports/evaluation_report.csv：所有实验的汇总台账（追加式，便于横向对比）
    - eval_reports/runs/<时间戳>__<tag>.csv：每次 mAP 的独立快照（永不覆盖历史，
      含逐类明细，便于微调前后对比）
"""
import os
import sys
import io
import csv
import time
import shutil
import argparse
import collections

# 被 pytest 导入时不能重包 stdout：TextIOWrapper 会在回收时关掉 pytest 的捕获流
if "pytest" not in sys.modules:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ultralytics import YOLO

from config import BASE_DIR  # 项目根已在上方 sys.path.insert 中加入
from config_classes import DETECTION_CLASSES

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# 评估报告专用文件夹：汇总台账 + 每次 mAP 的独立快照都收纳于此，避免散落项目根目录
EVAL_REPORT_DIR = os.path.join(BASE_DIR, "eval_reports")

# Roboflow 标注习惯与 COCO / 本项目词汇表的同义词映射（键与值都按小写规范名比较）。
# 只收录真实会出现的别名，未列出的类别按原名精确匹配。
CLASS_SYNONYMS = {
    "motorbike": "motorcycle",
    "fridge": "refrigerator",
    "sofa": "couch",
    "table": "dining table",
    "aeroplane": "airplane",
    "lorry": "truck",
    "pottedplant": "potted plant",
    "mobile phone": "cell phone",
    "tv/monitor": "tv",
}


def canonical(name):
    """类别名归一：小写 + 去空白 + 同义词替换"""
    n = str(name).strip().lower()
    return CLASS_SYNONYMS.get(n, n)


def _parse_flow_list(text):
    """解析 YAML 行内列表：['a', 'b'] 或 [a, b] → ["a", "b"]"""
    items = []
    for chunk in text.split(","):
        v = chunk.strip().strip("'\"").strip()
        if v:
            items.append(v)
    return items


def load_yaml_names(path):
    """解析 data.yaml 的 names 字段（避免引入 yaml 依赖）

    同时支持映射式（"  0: person"）与行内列表式（"names: ['a', 'b']"）。
    """
    names = []
    in_names = False
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not in_names and s.startswith("names:"):
                rest = s.split(":", 1)[1].strip()
                if rest.startswith("["):
                    buf = rest
                    while "]" not in buf:  # 行内列表跨行书写的兜底
                        nxt = next(f, None)
                        if nxt is None:
                            break
                        buf += " " + nxt.strip()
                    return _parse_flow_list(buf[buf.index("[") + 1:buf.rindex("]")])
                in_names = True
                continue
            if in_names:
                if ":" in s and s.split(":")[0].strip().isdigit():
                    names.append(s.split(":", 1)[1].strip().strip("'\""))
                elif s and not s.startswith("#"):
                    break
    return names


def find_splits(src_root):
    """返回 [(split 名, images 目录, labels 目录)]，兼容 split 式与扁平式布局"""
    out, seen = [], set()
    for sp in ("train", "valid", "val", "test"):
        d = os.path.join(src_root, sp, "images")
        if os.path.isdir(d):
            key = os.path.normcase(os.path.abspath(d))
            if key in seen:
                continue
            seen.add(key)
            out.append(("valid" if sp == "val" else sp,
                        d, os.path.join(src_root, sp, "labels")))
    if not out:
        d = os.path.join(src_root, "images")
        if os.path.isdir(d):
            out.append(("all", d, os.path.join(src_root, "labels")))
    return out


def count_dataset(src_root):
    """统计各 split 的图像数 / 标注文件数 / 标注框数"""
    rows = []
    for sp, img_dir, lbl_dir in find_splits(src_root):
        n_img = len([f for f in os.listdir(img_dir)
                     if os.path.splitext(f)[1].lower() in IMG_EXTS])
        n_lbl, n_box = 0, 0
        if os.path.isdir(lbl_dir):
            for f in os.listdir(lbl_dir):
                if not f.endswith(".txt"):
                    continue
                n_lbl += 1
                with open(os.path.join(lbl_dir, f), encoding="utf-8") as fh:
                    n_box += sum(1 for ln in fh if ln.split())
        rows.append({"split": sp, "images": n_img, "labels": n_lbl, "boxes": n_box})
    return rows


def build_eval_dataset(src_root, src_names, target_names, drop_names, split, work_dir,
                       rename=None):
    """在 work_dir 下生成一个可被 ultralytics 直接解析的评估数据集

    - images/  labels/：把选定 split 的图拷入并重写标注（类别下标换成 target_names 的下标）
    - data.yaml：写入绝对 path，train/val/test 均指向 images/（评估时只跑 val）
    - rename：{源类别名小写: 改后的类别名}，用于验证「标签改对后精度是多少」
    这样保证「模型词汇表下标 == 数据集类别下标」，mAP 逐类统计不会错位。
    """
    rename = {k.strip().lower(): v.strip() for k, v in (rename or {}).items()}
    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir)
    img_out = os.path.join(work_dir, "images")
    lbl_out = os.path.join(work_dir, "labels")
    os.makedirs(img_out)
    os.makedirs(lbl_out)

    tgt_index = {}
    for i, n in enumerate(target_names):
        tgt_index.setdefault(canonical(n), i)
    drops = {canonical(d) for d in drop_names if d.strip()}

    splits = find_splits(src_root)
    if split != "all":
        picked = [s for s in splits if s[0] == split]
        if not picked:
            raise SystemExit(f"[错误] 数据集中找不到 split={split}，可用: "
                             f"{[s[0] for s in splits] or ['all(扁平式)']}")
        splits = picked
    multi = len(splits) > 1

    st = {"images": 0, "gt_kept": 0, "gt_dropped": 0, "bg_images": 0,
          "gt_out_of_range": 0, "unmappable": collections.Counter(),
          "renamed": collections.Counter(),
          "per_class": collections.Counter(), "splits": collections.Counter()}

    for sp, img_dir, lbl_dir in splits:
        for fn in sorted(os.listdir(img_dir)):
            stem, ext = os.path.splitext(fn)
            if ext.lower() not in IMG_EXTS:
                continue
            new_stem = f"{sp}__{stem}" if multi else stem
            shutil.copy2(os.path.join(img_dir, fn), os.path.join(img_out, new_stem + ext))
            st["images"] += 1
            st["splits"][sp] += 1

            kept = []
            src_lbl = os.path.join(lbl_dir, stem + ".txt")
            if os.path.isfile(src_lbl):
                with open(src_lbl, encoding="utf-8") as fh:
                    for line in fh:
                        parts = line.split()
                        if len(parts) < 5:
                            continue
                        try:
                            ci = int(float(parts[0]))
                        except ValueError:
                            st["gt_out_of_range"] += 1
                            continue
                        if not 0 <= ci < len(src_names):
                            st["gt_out_of_range"] += 1
                            continue
                        cname = src_names[ci]
                        low = cname.strip().lower()
                        if low in rename:
                            st["renamed"][f"{cname}->{rename[low]}"] += 1
                            cname = rename[low]
                        key = canonical(cname)
                        if key in drops:
                            st["gt_dropped"] += 1
                            continue
                        if key not in tgt_index:
                            st["unmappable"][cname] += 1
                            continue
                        kept.append([str(tgt_index[key])] + parts[1:5])
                        st["per_class"][tgt_index[key]] += 1
                        st["gt_kept"] += 1
            with open(os.path.join(lbl_out, new_stem + ".txt"), "w", encoding="utf-8") as fh:
                for row in kept:
                    fh.write(" ".join(row) + "\n")
            if not kept:
                st["bg_images"] += 1

    yaml_path = os.path.join(work_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as fh:
        fh.write(f"path: {work_dir.replace(os.sep, '/')}\n")
        fh.write("train: images\nval: images\ntest: images\n")
        fh.write(f"nc: {len(target_names)}\n")
        fh.write("names:\n")
        for i, n in enumerate(target_names):
            fh.write(f"  {i}: {n}\n")
    return yaml_path, st


def write_run_snapshot(run_dir, ts_file, tag, meta_rows, class_rows, min_gt):
    """把单次 mAP 评估写成独立快照 CSV（每次一个文件，永不覆盖历史）。

    文件名 = <时间戳>__<安全化 tag>.csv；同名已存在则追加 _2/_3 递增，确保不覆盖。
    meta_rows: 已格式化好的元信息行（每行是 list，如 ["# 模型", "yolov8m.pt"]）。
    class_rows: 逐类别指标 dict 列表（含 class/gt/precision/recall/mAP50/mAP50-95）。
    返回实际写入的文件绝对路径。
    """
    os.makedirs(run_dir, exist_ok=True)
    safe_tag = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in str(tag))
    path = os.path.join(run_dir, f"{ts_file}__{safe_tag}.csv")
    n = 2
    while os.path.exists(path):
        path = os.path.join(run_dir, f"{ts_file}__{safe_tag}_{n}.csv")
        n += 1
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["# 单次 mAP 评估快照（独立存档，不覆盖历史；便于微调前后逐类对比）"])
        for row in meta_rows:
            w.writerow(row)
        w.writerow([])
        w.writerow(["类别", "GT", "Precision", "Recall", "mAP50", "mAP50-95", "类别档位"])
        for c in class_rows:
            if c.get("gt", 0) <= 0:
                continue
            w.writerow([c["class"], c["gt"], c["precision"], c["recall"],
                        c["mAP50"], c["mAP50-95"],
                        "主要" if c["gt"] >= min_gt else "长尾"])
    return os.path.abspath(path)


def main():
    parser = argparse.ArgumentParser(description="黄金数据集 mAP 评估")
    parser.add_argument("--data",
                        default=os.path.join(BASE_DIR, "datasets",
                                             "world-monitoring-v4-121", "data.yaml"))
    parser.add_argument("--model", default=os.path.join(BASE_DIR, "yolov8x-worldv2.pt"))
    parser.add_argument("--vocab", choices=["dataset", "project", "model"], default="dataset",
                        help="评估用的类别词汇表口径")
    parser.add_argument("--split", choices=["all", "train", "valid", "test"], default="all",
                        help="参与评估的数据划分；模型未在该集上训练，all 样本量最大最稳")
    parser.add_argument("--drop-classes", default="",
                        help="逗号分隔；这些 GT 类别直接丢弃不参与评估（如 Roboflow 误生成的项目名类别）")
    parser.add_argument("--rename", default="",
                        help="逗号分隔的 原名=新名；先改 GT 类别名再映射到词汇表，"
                             "用于量化「标签修正后」的真实精度（如 bus=person）")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001,
                        help="标准 mAP 需低阈值走完整 PR 曲线；测部署工作点请用 0.2")
    parser.add_argument("--tag", default="", help="评估缓存目录与 CSV 记录标识，默认自动生成")
    parser.add_argument("--min-gt", type=int, default=10,
                        help="「主要类别」门槛：GT 框数 >= 此值才计入主要类别 mAP"
                             "（长尾类只有几个框时 AP 无统计意义，会把宏平均严重拉低）")
    parser.add_argument("--out",
                        default=os.path.join(EVAL_REPORT_DIR, "evaluation_report.csv"),
                        help="汇总台账 CSV（追加式）；默认收纳到 eval_reports/ 专用文件夹")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"[错误] 找不到 {args.data}。请先采集/导出数据集并完成标注")
        sys.exit(1)
    src_root = os.path.dirname(os.path.abspath(args.data))
    src_names = load_yaml_names(args.data)
    if not src_names:
        print(f"[错误] {args.data} 里解析不出 names 类别表")
        sys.exit(1)

    rows = count_dataset(src_root)
    n_img = sum(r["images"] for r in rows)
    n_box = sum(r["boxes"] for r in rows)
    if n_img == 0 or n_box == 0:
        print(f"[错误] 数据集为空（{n_img} 张图 / {n_box} 个标注框），请先完成标注再评估")
        sys.exit(1)
    print(f"数据集: {src_root}")
    for r in rows:
        print(f"  {r['split']:<6} 图像 {r['images']:>4}  标注文件 {r['labels']:>4}  标注框 {r['boxes']:>5}")
    print(f"  合计   图像 {n_img:>4}  标注框 {n_box:>5}  源类别 {len(src_names)} 个")

    drop_names = [x.strip() for x in args.drop_classes.split(",") if x.strip()]
    rename = {}
    for pair in args.rename.split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            if k.strip() and v.strip():
                rename[k.strip()] = v.strip()

    m = YOLO(args.model)
    is_world = "world" in os.path.basename(args.model).lower()
    if args.vocab == "dataset":
        target = [n for n in src_names if canonical(n) not in {canonical(d) for d in drop_names}]
    elif args.vocab == "project":
        target = list(DETECTION_CLASSES)
    else:
        target = [m.names[i] for i in sorted(m.names)] if isinstance(m.names, dict) else list(m.names)
    if is_world:
        m.set_classes(target)

    tag = args.tag or f"{os.path.basename(src_root)}_{args.vocab}_{args.split}"
    work_dir = os.path.join(BASE_DIR, ".eval_cache", tag)
    yaml_path, st = build_eval_dataset(src_root, src_names, target, drop_names,
                                       args.split, work_dir, rename)

    print("-" * 64)
    print(f"词汇表口径: {args.vocab} | 类别数 {len(target)} | split={args.split}")
    print(f"评估集: {st['images']} 张（{dict(st['splits'])}） | 有效 GT 框 {st['gt_kept']} 个")
    if st["gt_dropped"]:
        print(f"  按 --drop-classes 丢弃 GT 框 {st['gt_dropped']} 个: {drop_names}")
    if st["renamed"]:
        print("  按 --rename 改名的 GT 框: "
              + ", ".join(f"{k}×{v}" for k, v in st["renamed"].most_common()))
    if st["unmappable"]:
        tot = sum(st["unmappable"].values())
        print(f"  词汇表覆盖不到而丢弃 GT 框 {tot} 个: "
              + ", ".join(f"{k}×{v}" for k, v in st["unmappable"].most_common()))
    if st["bg_images"]:
        print(f"  重写后无正样本的背景图 {st['bg_images']} 张（只贡献误报，不计入 AP）")
    if st["gt_out_of_range"]:
        print(f"  [警告] 类别下标越界/非法的标注行 {st['gt_out_of_range']} 行已跳过")
    print(f"评估数据缓存: {yaml_path}")

    print("开始评估（首次运行较慢，含模型预热）...")
    t0 = time.time()
    results = m.val(data=yaml_path, imgsz=args.imgsz, conf=args.conf,
                    split="val", verbose=False, plots=False)
    dt = time.time() - t0

    box = results.box
    print("=" * 64)
    print(f"评估完成（{dt:.1f}s）  模型={os.path.basename(args.model)}  "
          f"imgsz={args.imgsz} conf={args.conf} split={args.split} vocab={args.vocab}")
    print(f"  mAP50-95 : {box.map:.4f}   ← 综合精度（最严格）")
    print(f"  mAP50    : {box.map50:.4f}   ← 常用精度指标")
    print(f"  mAP75    : {box.map75:.4f}")
    print(f"  Precision: {box.mp:.4f}   Recall: {box.mr:.4f}")
    print("=" * 64)

    idxs = getattr(box, "ap_class_index", None)
    # ap_class_index 是 numpy 数组，不能用 `or` 做真值判断（会抛 ambiguous 异常）
    idxs = [int(x) for x in idxs] if idxs is not None and len(idxs) else list(range(len(target)))
    print(f"{'类别':<26}{'GT':>6}{'P':>8}{'R':>8}{'mAP50':>8}{'mAP50-95':>10}")
    print("-" * 66)
    class_rows = []
    for k, ci in enumerate(idxs):
        ci = int(ci)
        name = target[ci] if ci < len(target) else str(ci)
        p = float(box.p[k]) if k < len(box.p) else 0.0
        r = float(box.r[k]) if k < len(box.r) else 0.0
        ap50 = float(box.ap50[k]) if k < len(box.ap50) else 0.0
        ap = float(box.ap[k]) if k < len(box.ap) else 0.0
        n_gt = st["per_class"].get(ci, 0)
        class_rows.append({"class": name, "gt": n_gt, "precision": round(p, 4),
                           "recall": round(r, 4), "mAP50": round(ap50, 4),
                           "mAP50-95": round(ap, 4)})
        print(f"{name:<26}{n_gt:>6}{p:>8.3f}{r:>8.3f}{ap50:>8.3f}{ap:>10.3f}")

    # 长尾类（GT 框极少）的 AP 基本是 0/1 跳变，没有统计意义，却与主要类别等权平均，
    # 会把 mAP 压得看不出真实水平。额外给出「仅主要类别」的宏平均作为主口径。
    main_rows = [c for c in class_rows if c["gt"] >= args.min_gt]
    tail_rows = [c for c in class_rows if 0 < c["gt"] < args.min_gt]
    main_map50 = sum(c["mAP50"] for c in main_rows) / len(main_rows) if main_rows else 0.0
    main_map = sum(c["mAP50-95"] for c in main_rows) / len(main_rows) if main_rows else 0.0
    main_p = sum(c["precision"] for c in main_rows) / len(main_rows) if main_rows else 0.0
    main_r = sum(c["recall"] for c in main_rows) / len(main_rows) if main_rows else 0.0
    print("-" * 66)
    print(f"全部有 GT 的类别({len(class_rows)} 个) mAP50={float(box.map50):.4f}  "
          f"mAP50-95={float(box.map):.4f}")
    if main_rows:
        print(f"主要类别(GT>={args.min_gt}, {len(main_rows)} 个: "
              f"{[c['class'] for c in main_rows]})")
        print(f"  mAP50={main_map50:.4f}  mAP50-95={main_map:.4f}  "
              f"P={main_p:.4f}  R={main_r:.4f}   ← 建议以这组作为汇报口径")
    if tail_rows:
        print(f"长尾类别(GT<{args.min_gt}, {len(tail_rows)} 个: "
              f"{[(c['class'], c['gt']) for c in tail_rows]}) → 样本太少，AP 不具统计意义")

    # 汇总台账（追加式，所有实验横向对比）；确保 eval_reports/ 目录存在
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    label_caliber = args.rename or ("drop:" + args.drop_classes if drop_names else "原始标签")
    write_header = not os.path.exists(args.out)
    with open(args.out, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["时间", "模型", "vocab", "split", "imgsz", "conf",
                             "图像数", "GT框数", "mAP50-95", "mAP50",
                             "precision", "recall",
                             "主要类别数", "主要类mAP50", "主要类mAP50-95",
                             "主要类P", "主要类R", "标签口径", "类别明细"])
        writer.writerow([time.strftime("%Y-%m-%d %H:%M"), os.path.basename(args.model),
                         args.vocab, args.split, args.imgsz, args.conf,
                         st["images"], st["gt_kept"],
                         round(float(box.map), 4), round(float(box.map50), 4),
                         round(float(box.mp), 4), round(float(box.mr), 4),
                         len(main_rows), round(main_map50, 4), round(main_map, 4),
                         round(main_p, 4), round(main_r, 4),
                         label_caliber,
                         "; ".join(f"{c['class']}(n={c['gt']}):{c['mAP50']}"
                                   for c in class_rows if c["gt"] > 0)])

    # 单次快照（每次 mAP 独立存档到 eval_reports/runs/，永不覆盖历史，便于微调前后对比）
    ts_file = time.strftime("%Y%m%d_%H%M%S")
    meta_rows = [
        ["# 时间", time.strftime("%Y-%m-%d %H:%M:%S")],
        ["# 模型", os.path.basename(args.model)],
        ["# 数据集", src_root],
        ["# vocab", args.vocab],
        ["# split", args.split],
        ["# imgsz", args.imgsz],
        ["# conf", args.conf],
        ["# tag", tag],
        ["# 标签口径", label_caliber],
        ["# 图像数", st["images"]],
        ["# GT框数", st["gt_kept"]],
        ["# 总体", f"mAP50-95={float(box.map):.4f}", f"mAP50={float(box.map50):.4f}",
         f"P={float(box.mp):.4f}", f"R={float(box.mr):.4f}"],
        ["# 主要类别", f"数={len(main_rows)}", f"mAP50={main_map50:.4f}",
         f"mAP50-95={main_map:.4f}", f"P={main_p:.4f}", f"R={main_r:.4f}"],
    ]
    run_path = write_run_snapshot(os.path.join(EVAL_REPORT_DIR, "runs"),
                                  ts_file, tag, meta_rows, class_rows, args.min_gt)

    print("-" * 66)
    print(f"汇总台账已追加到: {os.path.abspath(args.out)}")
    print(f"本次快照已存档到: {run_path}")
    print("调参建议: 某类 Recall 低→该类别样本少或被漏检(加采该类图/降conf)；")
    print("          Precision 低→误报多(提高该类 conf 或从词汇表移除相近干扰词)")


if __name__ == "__main__":
    main()
