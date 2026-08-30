# -*- coding: utf-8 -*-
# diag_world_bench.py - benchmark DEFAULT model (yolov8x-worldv2.pt) with model.track imgsz=960
import argparse
import time, os
import numpy as np
import cv2

_parser = argparse.ArgumentParser(description="默认模型（yolov8x-worldv2）track 推理基准")
_parser.add_argument("--url", required=True,
                     help="必填：手机流地址，如 http://192.168.1.100:8080/video（打不开时自动回退本地图像测）")
_args = _parser.parse_args()

for k in ("HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy"):
    os.environ.pop(k, None)
from ultralytics import YOLO
from config_classes import DETECTION_CLASSES
print("loading model ...", flush=True)
m = YOLO("yolov8x-worldv2.pt")
m.set_classes(DETECTION_CLASSES)
cap = cv2.VideoCapture(_args.url)
frame = None
if cap.isOpened():
    ret, frame = cap.read()
    cap.release()
if frame is None:
    frame = cv2.imread("dataset/images/20260811_173701_0000.jpg")
    if frame is None: frame = np.random.randint(0,255,(1080,1920,3),dtype=np.uint8)
print(f"frame shape={frame.shape}", flush=True)
import torch
half = torch.cuda.is_available()
print(f"device cuda={torch.cuda.is_available()} half={half}", flush=True)
# warmup
m.track(frame, persist=True, conf=0.2, imgsz=960, tracker="bytetrack.yaml", half=half, verbose=False)
ts = []
for i in range(10):
    t1 = time.time()
    m.track(frame, persist=True, conf=0.2, imgsz=960, tracker="bytetrack.yaml", half=half, verbose=False)
    ts.append(time.time()-t1)
a = np.array(ts)
print(f"[yolov8x-worldv2 track imgsz=960] ms: mean={a.mean()*1000:.0f} med={np.median(a)*1000:.0f} min={a.min()*1000:.0f} max={a.max()*1000:.0f}", flush=True)
t1=time.time(); [float(frame.mean()) for _ in range(10)]; print(f"[frame.mean()] {((time.time()-t1)/10)*1000:.1f} ms/frame", flush=True)
t1=time.time(); [cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY,85]) for _ in range(10)]; print(f"[imencode q85] {((time.time()-t1)/10)*1000:.1f} ms/frame", flush=True)
ts640 = []
for i in range(5):
    t1 = time.time()
    m.track(frame, persist=True, conf=0.2, imgsz=640, tracker="bytetrack.yaml", half=half, verbose=False)
    ts640.append(time.time()-t1)
b = np.array(ts640)
print(f"[yolov8x-worldv2 track imgsz=640] ms: mean={b.mean()*1000:.0f} min={b.min()*1000:.0f}", flush=True)
print("BENCH DONE", flush=True)
