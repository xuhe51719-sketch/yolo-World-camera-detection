# -*- coding: utf-8 -*-
"""
参数扫描基准：在不同 imgsz × conf 组合下实测推理速度与检出量，输出对照表（CSV）。
用于调参时量化"速度-检出"权衡，每次改动配置后重跑一遍对比。

用法（在项目根目录运行）：
    ".venv\\Scripts\\python.exe" tools\\sweep.py --images dataset\\images
    ".venv\\Scripts\\python.exe" tools\\sweep.py --images cam_test --imgsz 640,960 --conf 0.1,0.2,0.3

参数：
    --images  图片目录（用 capture_frames.py 采集的数据，或任意测试图）
    --imgsz   逗号分隔的分辨率列表（默认 640,800,960）
    --conf    逗号分隔的置信度阈值列表（默认 0.1,0.2,0.3,0.4）
    --model   模型路径（默认 yolov8x-worldv2.pt）
    --runs    每个组合跑的图片轮数（默认 1，图少时可调大取平均）
    --out     结果 CSV 路径（默认 benchmark_results.csv）

说明：
    - 速度 = 纯推理耗时（不含读流/编码），与 /metrics 的 infer_ms_avg 同口径
    - 检出量/置信度反映该配置下的"灵敏度"：conf 越低检出越多但误报越多
"""
import os
import sys
import io
import csv
import time
import glob
import argparse

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import cv2
from ultralytics import YOLO
from config_classes import DETECTION_CLASSES
from config import BASE_DIR


def main():
    parser = argparse.ArgumentParser(description="参数扫描基准测试")
    parser.add_argument("--images", default=os.path.join(BASE_DIR, "dataset", "images"))
    parser.add_argument("--imgsz", default="640,800,960")
    parser.add_argument("--conf", default="0.1,0.2,0.3,0.4")
    parser.add_argument("--model", default=os.path.join(BASE_DIR, "yolov8x-worldv2.pt"))
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--out", default=os.path.join(BASE_DIR, "benchmark_results.csv"))
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.images, "*.jpg")) +
                   glob.glob(os.path.join(args.images, "*.png")))
    if not files:
        print(f"[错误] {args.images} 里没有图片。先用 tools/capture_frames.py 采集，"
              f"或指定 --images 到已有图片目录")
        sys.exit(1)
    # 限制评估图数量，避免扫描过慢
    files = files[:30]
    frames = [cv2.imread(f) for f in files]
    frames = [f for f in frames if f is not None]
    print(f"测试图片: {len(frames)} 张 | 模型: {args.model}")

    m = YOLO(args.model)
    if "world" in args.model.lower():
        m.set_classes(DETECTION_CLASSES)

    imgsz_list = [int(x) for x in args.imgsz.split(",")]
    conf_list = [float(x) for x in args.conf.split(",")]

    rows = []
    print(f"{'imgsz':>6} {'conf':>6} {'推理ms/帧':>10} {'FPS':>7} {'检出/图':>8} {'平均置信度':>10}")
    print("-" * 60)
    for sz in imgsz_list:
        # 预热（首次推理含 CUDA 上下文初始化）
        m(frames[0], imgsz=sz, verbose=False)
        for conf in conf_list:
            total_dt, n_boxes, conf_sum = 0.0, 0, 0.0
            for _ in range(args.runs):
                for img in frames:
                    t0 = time.time()
                    r = m(img, imgsz=sz, conf=conf, verbose=False)[0]
                    total_dt += time.time() - t0
                    nb = 0 if r.boxes is None else len(r.boxes)
                    n_boxes += nb
                    if nb:
                        conf_sum += float(r.boxes.conf.sum())
            n_infer = len(frames) * args.runs
            avg_ms = total_dt / n_infer * 1000
            fps = 1000.0 / avg_ms if avg_ms > 0 else 0
            boxes_per_img = n_boxes / n_infer
            avg_conf = conf_sum / n_boxes if n_boxes else 0.0
            rows.append({"imgsz": sz, "conf": conf, "infer_ms": round(avg_ms, 1),
                         "fps": round(fps, 1), "boxes_per_img": round(boxes_per_img, 2),
                         "avg_conf": round(avg_conf, 3), "model": args.model})
            print(f"{sz:>6} {conf:>6} {avg_ms:>10.1f} {fps:>7.1f} {boxes_per_img:>8.2f} {avg_conf:>10.3f}")

    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["imgsz", "conf", "infer_ms", "fps",
                                               "boxes_per_img", "avg_conf", "model"])
        writer.writeheader()
        writer.writerows(rows)
    print("-" * 60)
    print(f"结果已保存: {os.path.abspath(args.out)}")
    print("解读: 选满足帧率需求的最大 imgsz；conf 从检出量/误报的平衡中挑选")


if __name__ == "__main__":
    main()
