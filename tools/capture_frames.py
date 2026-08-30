# -*- coding: utf-8 -*-
"""
采集真实场景帧，为建立黄金评估数据集做准备。

用法（在项目根目录运行）：
    ".venv\\Scripts\\python.exe" tools\\capture_frames.py --url http://192.168.1.100:8080/video --count 60 --interval 0.5

参数：
    --url       手机流地址（必填，如 http://192.168.1.100:8080/video）
    --count     采集帧数（建议 50~200，场景越丰富越好）
    --interval  相邻两帧的间隔秒数（移动手机拍摄不同场景时调大）
    --out       输出目录（默认 项目根/dataset）

采集建议：
    1. 拿着手机把平时要检测的场景都拍一遍（桌面、门口、客厅…）
    2. 同一场景换不同角度/距离多采几张
    3. 采完后按 dataset/README_标注流程.md 进行人工标注
"""
import os
import sys
import io
import time
import argparse

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import cv2
from config_classes import DETECTION_CLASSES
from config import BASE_DIR


def _read_existing_yaml(path):
    """解析已有 data.yaml 的 nc / names（简单解析，与 eval_dataset 同口径）"""
    nc, names = None, []
    in_names = False
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s.startswith("nc:"):
                try:
                    nc = int(s.split(":", 1)[1].strip())
                except ValueError:
                    nc = None
                continue
            if s.startswith("names:"):
                in_names = True
                continue
            if in_names:
                if ":" in s and s.split(":")[0].strip().isdigit():
                    names.append(s.split(":", 1)[1].strip())
                elif s and not s.startswith("#"):
                    break
    return nc, names


def _write_yaml(path, out_dir):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(out_dir)}\n")
        f.write("train: images\n")
        f.write("val: images\n")
        f.write(f"nc: {len(DETECTION_CLASSES)}\n")
        f.write("names:\n")
        for idx, c in enumerate(DETECTION_CLASSES):
            f.write(f"  {idx}: {c}\n")


def main():
    parser = argparse.ArgumentParser(description="采集真实场景帧建立评估数据集")
    parser.add_argument("--url", required=True,
                        help="必填：手机流地址，如 http://192.168.1.100:8080/video")
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--out", default=os.path.join(BASE_DIR, "dataset"))
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.url)
    if not cap.isOpened():
        print(f"[错误] 打不开视频流: {args.url}")
        print("请确认手机 App 服务器已开启、地址正确（带 /video）")
        sys.exit(1)

    img_dir = os.path.join(args.out, "images")
    lbl_dir = os.path.join(args.out, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    saved = 0
    print(f"开始采集 {args.count} 帧（间隔 {args.interval}s）→ {img_dir}")
    print("提示：采集期间缓慢移动手机，覆盖不同场景和角度")
    for i in range(args.count):
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"[警告] 第 {i} 帧读取失败，重试")
            time.sleep(0.5)
            continue
        name = time.strftime("%Y%m%d_%H%M%S") + f"_{saved:04d}"
        cv2.imwrite(os.path.join(img_dir, name + ".jpg"), frame)
        saved += 1
        print(f"  已保存 {saved}/{args.count}: {name}.jpg")
        time.sleep(args.interval)
    cap.release()

    # 生成 data.yaml（ultralytics 数据集描述），names 先用全量词汇表，
    # 标注完成后按实际标注的类别修改。
    # 保护已有 data.yaml：若已存在且 nc/names 与当前词汇表不一致（说明用户
    # 手工改过），不覆盖，改写 data.yaml.new 并提示人工确认。
    yaml_path = os.path.join(args.out, "data.yaml")
    if os.path.exists(yaml_path):
        old_nc, old_names = _read_existing_yaml(yaml_path)
        if old_nc != len(DETECTION_CLASSES) or old_names != DETECTION_CLASSES:
            new_path = yaml_path + ".new"
            print(f"[警告] 已存在的 {yaml_path} 的 nc/names 与当前类别表不一致"
                  f"（已有 nc={old_nc}，当前 {len(DETECTION_CLASSES)}），为防覆盖手工修改，跳过覆盖。")
            _write_yaml(new_path, args.out)
            print(f"[提示] 新版已写入 {new_path}，确认无误后可手工替换 data.yaml。")
        else:
            _write_yaml(yaml_path, args.out)
            print(f"data.yaml 已更新（nc/names 与类别表一致）: {yaml_path}")
    else:
        _write_yaml(yaml_path, args.out)

    # 标注指引：仅在不存在时生成，不覆盖用户已有的修改
    readme = os.path.join(args.out, "README_标注流程.md")
    if os.path.exists(readme):
        print(f"README 已存在，保持不变: {readme}")
    else:
        with open(readme, "w", encoding="utf-8") as f:
            f.write("""# 黄金数据集标注流程

## 目录结构（YOLO 格式）
```
dataset/
├── images/      ← 已采集的图片
├── labels/      ← 标注结果（每张图片一个同名 .txt）
└── data.yaml    ← 数据集描述（已自动生成）
```

## labels 文件格式
每个 .txt 每行一个框：`类别索引 中心x 中心y 宽 高`（全部归一化到 0~1）
类别索引必须与 data.yaml 中 names 的顺序一致。

## 推荐标注工具（任选其一）
1. **X-AnyLabeling**（免费桌面软件，支持 YOLO 导出）
   https://github.com/CVHub520/X-AnyLabeling
2. **Roboflow**（免费在线，标注后可直接导出 YOLOv8 格式）
   https://roboflow.com  → 新建项目 → 上传图片 → 标注 → Export → YOLOv8
3. **labelImg**（经典轻量，pip install labelImg）

## 标注要点
- 只标 data.yaml 里列出的类别；想加新类别先在 config_classes.py 和 data.yaml 里加
- 框要贴紧物体边缘，宁紧勿松
- 每张图标完检查一遍漏标/错标
- 50~100 张认真标注的图就能得到有统计意义的 mAP

## 标注完成后评估
    ".venv\\Scripts\\python.exe" tools\\eval_dataset.py --data dataset\\data.yaml
""")

    print("=" * 50)
    print(f"采集完成：{saved} 张图片 → {os.path.abspath(img_dir)}")
    print(f"下一步：按 {readme} 进行人工标注，然后用 tools/eval_dataset.py 评估")


if __name__ == "__main__":
    main()
