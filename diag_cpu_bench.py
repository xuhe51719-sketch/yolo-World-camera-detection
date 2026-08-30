# -*- coding: utf-8 -*-
import time, os
import numpy as np, cv2
for k in ("HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy"):
    os.environ.pop(k, None)
os.environ["CUDA_VISIBLE_DEVICES"]=""  # force CPU
import torch
from ultralytics import YOLO
from config_classes import DETECTION_CLASSES
print("torch cuda avail(after mask):", torch.cuda.is_available(), flush=True)
m = YOLO("yolov8x-worldv2.pt"); m.set_classes(DETECTION_CLASSES)
frame = cv2.imread("dataset/images/20260811_173701_0000.jpg")
if frame is None: frame = np.random.randint(0,255,(1080,1920,3),dtype=np.uint8)
frame = cv2.resize(frame,(1920,1080))
print("warmup...", flush=True)
m.track(frame, persist=True, conf=0.2, imgsz=960, tracker="bytetrack.yaml", half=False, verbose=False)
ts=[]
for i in range(3):
    t1=time.time(); m.track(frame, persist=True, conf=0.2, imgsz=960, tracker="bytetrack.yaml", half=False, verbose=False); ts.append(time.time()-t1)
    print(f"  cpu iter{i}: {ts[-1]*1000:.0f} ms", flush=True)
a=np.array(ts)
print(f"[CPU yolov8x-worldv2 imgsz=960] mean={a.mean()*1000:.0f} ms/frame -> {1/a.mean():.2f} FPS", flush=True)
print("CPU BENCH DONE", flush=True)
