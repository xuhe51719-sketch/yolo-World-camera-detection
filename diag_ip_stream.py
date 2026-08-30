# -*- coding: utf-8 -*-
# diag_ip_stream.py - one-shot diagnostics: IP camera connectivity / stream rate / FFmpeg buffer backlog / YOLO inference benchmark
import argparse
import socket, ssl, time, os, urllib.request
import numpy as np
import cv2

HOST = None
PORT = None
URLS = []

def p(m): print(m, flush=True)

def tcp_probe():
    s = socket.socket(); s.settimeout(3)
    t0 = time.time()
    try:
        s.connect((HOST, PORT)); s.close()
        p(f"[TCP] {HOST}:{PORT} connect OK, took {time.time()-t0:.2f}s")
        return True
    except Exception as e:
        p(f"[TCP] {HOST}:{PORT} connect FAILED ({time.time()-t0:.2f}s): {type(e).__name__}: {e}")
        return False

def http_probe(url, timeout=6):
    ctx = ssl._create_unverified_context() if url.startswith("https") else None
    req = urllib.request.Request(url, headers={"User-Agent": "diag"})
    t0 = time.time()
    try:
        r = urllib.request.urlopen(req, timeout=timeout, context=ctx)
        ct = r.headers.get("Content-Type", "")
        data = r.read(200000)
        p(f"[HTTP] {url} -> {r.status}, Content-Type={ct}, got {len(data)}B in {time.time()-t0:.2f}s")
        if "multipart" in ct.lower():
            p(f"[HTTP] MJPEG multipart stream detected, head: {data[:100]!r}")
        try: r.close()
        except Exception: pass
        return True
    except Exception as e:
        p(f"[HTTP] {url} FAILED ({time.time()-t0:.2f}s): {type(e).__name__}: {e}")
        return False

def stream_test(url):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        p(f"[CV2] {url} open FAILED"); return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    p(f"[CV2] opened OK resolution={w}x{h} meta_fps={cap.get(cv2.CAP_PROP_FPS)}")
    dts = []; last = None; t0 = time.time()
    while time.time() - t0 < 10:
        t1 = time.time(); ret, fr = cap.read(); dt = time.time() - t1
        if not ret: p("[CV2] read interrupted"); break
        dts.append(dt); last = fr
    dur = time.time() - t0; a = np.array(dts)
    if a.size:
        p(f"[Phase1] read {a.size} frames in {dur:.1f}s -> consume rate = {a.size/dur:.2f} FPS")
        p(f"[Phase1] read/decode ms: mean={a.mean()*1000:.1f} med={np.median(a)*1000:.1f} p95={np.percentile(a,95)*1000:.1f} max={a.max()*1000:.1f}")
    p("[Phase2] pause reading 3s (simulate inference stall) ...")
    time.sleep(3)
    dts2 = []; t0 = time.time()
    while time.time() - t0 < 5:
        t1 = time.time(); ret, fr = cap.read(); dt = time.time() - t1
        if not ret: break
        dts2.append(dt); last = fr
    b = np.array(dts2)
    if b.size:
        burst = int((b < 0.01).sum())
        p(f"[Phase2] {b.size} frames in 5s ({b.size/5:.2f} FPS), backlog frames(dt<10ms)={burst}")
        p(f"[Phase2] first-10 mean read={b[:10].mean()*1000:.1f}ms (well below steady-state => buffered backlog exists, latency grows)")
    cap.release()
    return last

def yolo_bench(frame):
    from ultralytics import YOLO
    m = YOLO("yolov8n.pt")
    m.predict(frame, verbose=False)
    ts = []
    for _ in range(15):
        t1 = time.time(); m.predict(frame, verbose=False); ts.append(time.time()-t1)
    a = np.array(ts)
    p(f"[YOLO yolov8n] infer ms: mean={a.mean()*1000:.1f} med={np.median(a)*1000:.1f} min={a.min()*1000:.1f} (device auto)")

def main():
    global HOST, PORT, URLS
    parser = argparse.ArgumentParser(description="IP 摄像头流诊断（连通性/流帧率/缓冲积压/推理基准）")
    parser.add_argument("--host", required=True,
                        help="必填：手机摄像头 IP，如 192.168.1.100")
    parser.add_argument("--port", type=int, default=8080,
                        help="手机推流端口（默认 8080）")
    args = parser.parse_args()
    HOST, PORT = args.host, args.port
    URLS = [
        f"http://{HOST}:{PORT}/",
        f"http://{HOST}:{PORT}/video",
        f"https://{HOST}:{PORT}/",
        f"https://{HOST}:{PORT}/video",
    ]
    for k in ("HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy"):
        os.environ.pop(k, None)
    p("=== 1. TCP connectivity ===")
    ok = tcp_probe()
    p("=== 2. HTTP/HTTPS endpoint probes ===")
    good = None
    for u in URLS:
        if http_probe(u) and u.startswith("http://") and good is None:
            good = u
    if ok and good is None:
        good = URLS[1]
    p("=== 3. cv2 stream measurement ===")
    frame = None
    if good: frame = stream_test(good)
    if frame is None:
        frame = cv2.imread("dataset/images/20260811_173701_0000.jpg")
        if frame is None: frame = np.random.randint(0,255,(720,1280,3),dtype=np.uint8)
        p("[CV2] stream unavailable, benchmarking on local image instead")
    p("=== 4. YOLO inference benchmark (yolov8n.pt) ===")
    yolo_bench(frame)
    p("=== DIAG DONE ===")

if __name__ == "__main__":
    main()

