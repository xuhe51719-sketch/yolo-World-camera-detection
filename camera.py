# -*- coding: utf-8 -*-
"""
摄像头模块：手机流 / 本地摄像头的打开、重连、地址纠错与画面方向校正。
从原 app.py 拆分而来，行为保持一致；本模块导入无任何副作用。
"""

import logging
import os
import threading
import time

import cv2

import state
from config import settings

logger = logging.getLogger(__name__)

# FFmpeg 会读取的代理环境变量键（大小写都要处理）
PROXY_ENV_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                  "http_proxy", "https_proxy", "all_proxy")
FFMPEG_OPTS_KEY = "OPENCV_FFMPEG_CAPTURE_OPTIONS"


def _snapshot_proxy_env():
    """快照并移除代理环境变量，返回 {小写键: (实际键, 原值)}。
    Windows 环境变量不区分大小写（HTTP_PROXY 与 http_proxy 同一变量），
    必须按小写分组去重，否则同一变量会被处理两次、还原时报 KeyError。"""
    snapshot = {}
    for actual_key, value in list(os.environ.items()):
        if actual_key.lower() in {k.lower() for k in PROXY_ENV_KEYS}:
            snapshot.setdefault(actual_key.lower(), (actual_key, value))
    for _, (actual_key, _) in snapshot.items():
        os.environ.pop(actual_key, None)
    return snapshot


def _restore_proxy_env(snapshot):
    """逐键精确还原：原本存在的恢复原值，原本不存在的确保删除（不留空串）"""
    for lower_key, (actual_key, value) in snapshot.items():
        os.environ[actual_key] = value
    for actual_key in list(os.environ):
        if actual_key.lower() in {k.lower() for k in PROXY_ENV_KEYS} \
                and actual_key.lower() not in snapshot:
            os.environ.pop(actual_key, None)


# ============================================================
# 推流地址处理
# ============================================================
def normalize_camera_url(url):
    """清理推流地址：全角字符转半角（中文输入法常见错误），自动补全协议头"""
    url = (url or '').strip()
    fullwidth_map = {
        '：': ':', '／': '/', '＼': '\\', '；': ';', '＠': '@',
        '？': '?', '＆': '&', '＝': '=', '％': '%', '．': '.',
        '０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
        '５': '5', '６': '6', '７': '7', '８': '8', '９': '9',
    }
    for full, half in fullwidth_map.items():
        url = url.replace(full, half)
    url = url.strip()
    if url and '://' not in url:
        url = 'http://' + url
    return url


def candidate_urls(url):
    """生成候选地址列表（按尝试优先级排序）：
    - HTTP 地址若没带路径，补一个 /video 候选并优先尝试（IP Webcam 等 App 主界面
      显示的是 http://IP:8080，视频流实际在 /video），避免先浪费时间试主界面地址；
    - http 优先于 https（手机推流 App 一般只提供 HTTP 服务）"""
    base_urls = []
    if url.lower().startswith('https://'):
        base_urls.append('http://' + url.split('://', 1)[1])   # 误填 https 时优先试对应 http 地址
    base_urls.append(url)

    urls = []
    for base in base_urls:
        candidates = [base]
        low = base.lower()
        if low.startswith('http://') or low.startswith('https://'):
            rest = base.split('://', 1)[1]
            host_path = rest.split('?', 1)[0]           # 先剥离查询串再判断路径，避免把 /video 拼进查询串
            host, _, path = host_path.partition('/')
            if host and (path == '' or path == '/'):    # 仅根路径（含尾斜杠）且 host 非空时才补 /video
                candidates.insert(0, base.split('?', 1)[0].rstrip('/') + '/video')
        for c in candidates:
            if c not in urls:
                urls.append(c)
    return urls


# ============================================================
# 网络流"只取最新帧"包装器
# ============================================================
class LatestFrameCap:
    """网络流"只取最新帧"包装器。

    OpenCV/FFmpeg 读取 HTTP 流时内部有缓冲：一旦推理速度低于推流帧率，
    每次 read() 拿到的都是积压的旧帧，延迟会越滚越大。
    这里用后台线程持续读流、只保留最新一帧，read() 永远返回最新画面，
    把延迟压到"一帧采集 + 一次推理"的量级；若长时间没有新帧则判定断流。
    读流线程用 grab()（不解码，远快于 read()）连续排空 FFmpeg 内部缓冲，
    只对最后一次成功 grab 的帧做 retrieve() 解码，确保始终只拿"最新到达帧"。
    """

    # 每轮"排空窗口"（秒）：窗口内用 grab 快速跳过积压旧帧，只解码最后一帧；
    # 窗口长度决定拿到帧的新鲜度上限，也决定断流时的退出响应速度。
    # 连续 grab/retrieve 失败阈值：瞬时丢包不应触发全量重连，
    # 只有连续失败达到阈值（约 10 轮 × 0.05s ≈ 0.5s）才判定断流退出。
    DRAIN_WINDOW = 0.05
    FAIL_STREAK_MAX = 10

    def __init__(self, cap, stale_timeout=5.0, max_frame_age=None):
        if max_frame_age is None:
            max_frame_age = settings.frame_max_age
        self.cap = cap
        self.stale_timeout = stale_timeout
        self.max_frame_age = max_frame_age
        self._lock = threading.Lock()
        self._frame = None
        self._last_new = time.time()
        self._alive = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        fail_streak = 0   # 连续失败计数：任何一次成功即清零，达阈值才退出重连
        while self._alive:
            grabbed = False
            t0 = time.time()
            try:
                # grab() 只跳帧边界、不解码，远快于 read()，可快速排空 FFmpeg 内部
                # 积压的旧帧；只对最后一次成功 grab 的帧 retrieve() 解码，
                # 确保始终只保留"最新到达帧"，延迟不再随积压线性增长。
                while self._alive:
                    if not self.cap.grab():
                        break   # grab 失败（流结束/断开），按连续失败计数处理
                    grabbed = True
                    if time.time() - t0 >= self.DRAIN_WINDOW:
                        break
            except Exception:
                pass
            if not self._alive:
                break
            if not grabbed:
                fail_streak += 1
                if fail_streak >= self.FAIL_STREAK_MAX:
                    self._alive = False   # 连续失败达阈值，判定断流，外层触发重连
                    break
                time.sleep(0.05)
                continue
            try:
                ret, frame = self.cap.retrieve()
            except Exception:
                ret, frame = False, None
            if not ret or frame is None:
                fail_streak += 1
                if fail_streak >= self.FAIL_STREAK_MAX:
                    self._alive = False   # 连续解码失败达阈值，外层触发重连
                    break
                time.sleep(0.05)
                continue
            fail_streak = 0
            with self._lock:
                self._frame = frame
                self._last_new = time.time()
        try:
            self.cap.release()
        except Exception:
            pass

    def read(self):
        """返回最新帧；首帧最多等 1.5s。
        帧龄超过 max_frame_age 视为过期：不返回旧帧，改为等待新帧到达，
        避免展示秒级延迟的旧画面；长时间无新帧按断流处理"""
        t_end = time.time() + 1.5
        fallback = None
        while True:
            with self._lock:
                frame = self._frame
                last_new = self._last_new
            now = time.time()
            if frame is not None:
                if (now - last_new) > self.stale_timeout:
                    return False, None     # 长时间无新帧，按断流处理
                if (now - last_new) <= self.max_frame_age:
                    return True, frame     # 新鲜帧，直接返回（时间戳即 _last_new）
                fallback = frame           # 帧已过期：记住兜底，等新帧到达
            if now > t_end or not self._alive:
                if not self._alive:
                    return False, None     # 读流线程已退出，直接按断流处理
                if fallback is not None and (now - last_new) <= settings.fallback_max_age:
                    return True, fallback  # 兜底帧帧龄仍在可接受范围内，尽力返回
                return False, None         # 兜底帧也过于陈旧，按断流处理触发重连
            time.sleep(0.02)

    def isOpened(self):
        return self._alive and self.cap.isOpened()

    def get(self, prop):
        return self.cap.get(prop)

    def release(self):
        self._alive = False
        self._thread.join(timeout=2.0)   # 实际释放由读流线程完成，避免跨线程操作 VideoCapture


# ============================================================
# 画面方向校正
# ============================================================
def apply_orientation(frame):
    """按当前画面方向设置旋转/镜像（手机流方向固定，需在电脑端校正）"""
    rot = state.phone_orientation.get("rotate", 0)
    if rot == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rot == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif rot == 270:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    mirror = state.phone_orientation.get("mirror", "")
    if mirror == "h":
        frame = cv2.flip(frame, 1)
    elif mirror == "v":
        frame = cv2.flip(frame, 0)
    return frame


# ============================================================
# 手机 IP 摄像头打开
# ============================================================
def try_open_ip_camera(url):
    """尝试以网络流方式打开手机摄像头（HTTP MJPEG / RTSP）
    依次尝试候选地址；每个地址先默认后端，失败再尝试 FFMPEG 后端。
    注意：opencv-python 5.0 下不能用带 params 的 VideoCapture 重载
    （会触发 IMAGES 后端 "unsupported parameters... Bailout" 直接失败），
    必须用无参形式；连接重试由 detection_loop 的循环负责。

    另注意：FFmpeg 会读取 HTTP_PROXY/HTTPS_PROXY 环境变量，把局域网的
    手机流也路由到代理（如 Clash）导致打不开。摄像头流永远是局域网直连，
    打开期间临时移除代理变量，打开后恢复。"""
    # 入口完整快照 6 个代理键 + OPENCV_FFMPEG_CAPTURE_OPTIONS：
    # 记录"原本存在与否 + 原值"，finally 中逐一精确还原（大小写不敏感，兼容 Windows）
    saved_proxy = _snapshot_proxy_env()
    if saved_proxy:
        logger.info("检测到代理环境变量 %s，连接摄像头时已临时移除（局域网直连）",
                    sorted(saved_proxy))
    saved_ffmpeg_opts = os.environ.get(FFMPEG_OPTS_KEY)
    # FFmpeg 网络流打开/读取超时（微秒）：防止网络"半死"时 open/grab 无限阻塞、
    # 旧连接不释放导致重连持续失败；已有手工配置则不覆盖。
    os.environ.setdefault(FFMPEG_OPTS_KEY, "timeout;5000000|stimeout;5000000")
    try:
        for candidate in candidate_urls(url):
            for backend in (None, cv2.CAP_FFMPEG):
                try:
                    cap = cv2.VideoCapture(candidate, backend) if backend else cv2.VideoCapture(candidate)
                except Exception as e:
                    logger.warning("打开手机流异常(url=%s, backend=%s): %s", candidate, backend, e)
                    continue
                if not cap.isOpened():
                    cap.release()
                    continue
                # 将流缓冲限制为 1 帧：配合 LatestFrameCap 的 grab 排空，
                # 从根源上避免 FFmpeg 内部旧帧积压（延迟线性增长的结构性根因）
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                # 打开/读取超时（毫秒，尽力而为）：进一步防止半死网络下 grab/read 无限阻塞
                try:
                    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
                except Exception:
                    pass
                # 读一帧确认有真实画面
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    state.set_camera_info(index=candidate, backend_name="ip-camera",
                                          width=w, height=h)
                    if candidate != url:
                        logger.info("原始地址打不开，已自动改用: %s", candidate)
                    logger.info("已连接手机摄像头: %s 分辨率=%dx%d", candidate, w, h)
                    return LatestFrameCap(cap)   # 后台持续读流，只保留最新帧
                cap.release()
        return None
    finally:
        # 逐键还原：原本存在的恢复原值，原本不存在的删除（不能留空串）
        _restore_proxy_env(saved_proxy)
        if saved_ffmpeg_opts is None:
            os.environ.pop(FFMPEG_OPTS_KEY, None)
        else:
            os.environ[FFMPEG_OPTS_KEY] = saved_ffmpeg_opts


# ============================================================
# 摄像头自动选择
# ============================================================
def open_camera():
    """
    优先尝试手机 IP 摄像头（若已配置地址），
    否则自动尝试多个本地摄像头索引和后端，选择读取速度最快的可用摄像头。
    """
    # 优先尝试手机 IP 摄像头
    url = state.phone_url_override if state.phone_url_override else settings.phone_camera_url
    if url:
        cap = try_open_ip_camera(url)
        if cap is not None:
            return cap
        if not settings.phone_fallback_to_local:
            logger.warning("手机流无法打开；已禁用本地回退，将持续重试手机流（避免被本地黑屏误导）...")
            return None
        logger.warning("手机流无法打开，回退到本地摄像头...")

    candidates = []
    # 先尝试所有索引的默认后端，再尝试 DirectShow
    for idx in range(3):
        candidates.append((idx, None, "default"))
    for idx in range(3):
        candidates.append((idx, cv2.CAP_DSHOW, "DirectShow"))

    best_cap = None
    best_info = None
    best_avg = float('inf')

    for idx, backend, backend_name in candidates:
        # 单个候选的构造与探测整体隔离：某个候选抛异常（驱动崩溃等）
        # 只跳过该候选，不中止整个扫描
        try:
            cap = cv2.VideoCapture(idx, backend) if backend is not None else cv2.VideoCapture(idx)
            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.frame_height)
            cap.set(cv2.CAP_PROP_FPS, settings.target_fps)

            # 先快速读一帧，如果本身就很慢，直接跳过这个候选
            t0 = time.time()
            ret, frame = cap.read()
            first_dt = time.time() - t0
            if not ret or frame is None or frame.size == 0 or first_dt > 0.5:
                logger.info("跳过摄像头: index=%s, backend=%s, 首帧=%.3fs", idx, backend_name, first_dt)
                cap.release()
                continue

            # 再读 2 帧计算平均速度
            times = [first_dt]
            for _ in range(2):
                t0 = time.time()
                ret, frame = cap.read()
                t1 = time.time()
                if ret and frame is not None and frame.size > 0:
                    times.append(t1 - t0)

            avg_dt = sum(times) / len(times)
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        except Exception as e:
            logger.warning("测试摄像头异常，跳过该候选: index=%s, backend=%s, %s", idx, backend_name, e)
            continue
        logger.info("测试摄像头: index=%s, backend=%s, 分辨率=%dx%d, 平均读帧=%.3fs",
                    idx, backend_name, actual_w, actual_h, avg_dt)

        # 如果找到一个读帧很快（<0.2s）的摄像头，直接采用
        if avg_dt < 0.2:
            state.set_camera_info(index=idx, backend_name=backend_name,
                                  width=actual_w, height=actual_h)
            # 释放之前备选 slower 摄像头
            if best_cap is not None:
                best_cap.release()
            return cap

        # 否则作为备选，保留最快的一个
        if avg_dt < best_avg:
            if best_cap is not None:
                best_cap.release()
            best_avg = avg_dt
            best_cap = cap
            best_info = (idx, backend_name, actual_w, actual_h)
        else:
            cap.release()

    if best_cap is not None:
        idx, backend_name, actual_w, actual_h = best_info
        state.set_camera_info(index=idx, backend_name=backend_name,
                              width=actual_w, height=actual_h)
        logger.info("摄像头已连接: index=%s, backend=%s, 分辨率=%dx%d", idx, backend_name, actual_w, actual_h)
        return best_cap

    return None
