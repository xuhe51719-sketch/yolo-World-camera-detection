"""
摄像头黑屏诊断脚本
对每个摄像头索引 × 后端组合：打开、预热、读多帧，
计算像素均值/标准差，判断返回的是真实画面还是黑图，
并把样本帧存到 cam_test/ 便于肉眼检查。

像素均值 mean：全黑≈0，正常画面一般 30~200
标准差 std：纯黑/纯色≈0，真实画面一般 >5
"""
import os
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import BASE_DIR

# 输出目录锚定项目根，不依赖运行时工作目录
OUT_DIR = os.path.join(BASE_DIR, "cam_test")
os.makedirs(OUT_DIR, exist_ok=True)

BACKENDS = [
    (None, "default"),
    (cv2.CAP_DSHOW, "DirectShow"),
    (cv2.CAP_MSMF, "MSMF"),
]


def test_camera(idx, backend, backend_name):
    cap = cv2.VideoCapture(idx, backend) if backend is not None else cv2.VideoCapture(idx)
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 预热：有的摄像头前几帧是黑的，丢弃
    for _ in range(8):
        cap.read()
        time.sleep(0.03)

    means, stds = [], []
    last_frame = None
    for _ in range(5):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            means.append(float(frame.mean()))
            stds.append(float(frame.std()))
            last_frame = frame
        time.sleep(0.05)

    cap.release()

    if not means:
        return None
    return {
        "mean": sum(means) / len(means),
        "std": sum(stds) / len(stds),
        "frame": last_frame,
    }


def verdict(mean, std):
    # 平均亮度是最可靠的判据：纯黑画面 mean≈0（标准差只是黑底噪声）
    if mean < 10:
        return "黑图/无画面"
    if std < 3:
        return "疑似纯色/遮挡"
    return "有真实画面"


print("=" * 70)
print(f"{'index':>5} {'backend':<12} {'mean':>8} {'std':>8}  判定")
print("=" * 70)

working = []
for idx in range(3):
    for backend, name in BACKENDS:
        try:
            r = test_camera(idx, backend, name)
        except Exception as e:
            print(f"{idx:>5} {name:<12}  异常: {e}")
            continue
        if r is None:
            print(f"{idx:>5} {name:<12}      --      --  打不开/无帧")
            continue

        mean, std = r["mean"], r["std"]
        print(f"{idx:>5} {name:<12} {mean:>8.1f} {std:>8.1f}  {verdict(mean, std)}")

        # 保存样本帧
        if r["frame"] is not None:
            fname = os.path.join(OUT_DIR, f"diag_cam{idx}_{name}.jpg")
            cv2.imwrite(fname, r["frame"])

        if mean >= 10 and std >= 3:
            working.append((idx, name, mean, std))

print("=" * 70)
if working:
    print("可用摄像头（返回真实画面）：")
    for idx, name, mean, std in working:
        print(f"  index={idx}  backend={name}  mean={mean:.1f} std={std:.1f}")
else:
    print("没有任何一个本地摄像头返回真实画面——全部是黑图或打不开。")
print(f"样本帧已保存到 {OUT_DIR}，可直接查看。")
