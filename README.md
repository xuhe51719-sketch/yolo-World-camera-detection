# YOLO-World 摄像头实时目标检测系统

基于 YOLOv8 / YOLO-World 的实时目标检测系统：从**本地摄像头**或**手机 IP 摄像头**读取画面，用神经网络实时识别并框出画面中的物体，通过浏览器网页展示检测结果（检测框、类别、置信度、性能指标），并内置**多目标跟踪、滚动录像回放、多边形区域入侵报警**等完整功能。

后端为 Python + Flask，前端为一个自包含的 HTML 页面（MJPEG 实时视频流），无需 Node 构建，克隆即用。

## 功能特性

- **124 类开放词汇检测**：默认使用 YOLO-World 开放词汇模型（`yolov8x-worldv2.pt`），在 COCO 80 类之外可识别饮水机、垃圾桶、插线板、显示器等 44 种日常物品；类别表集中在 `config_classes.py`，增删后重启生效
- **双图像源**：本地摄像头自动选优（索引 × 后端试帧测速）；手机装 IP Webcam / DroidCam 等 App 即可当摄像头用，网页上随时填地址切换，支持地址自动纠错与断线重连
- **多目标跟踪**：内置 ByteTrack，每个物体有稳定跟踪 ID，检测框不逐帧闪烁
- **滚动录制与倍速回放**（默认开启）：每段 60 秒、滚动保留约 1 小时，磁盘剩余 <2GB 自动停录；网页选段回放，支持 0.5× / 1× / 1.5× / 2× 倍速与单段删除
- **多边形区域入侵报警**：在画面上手绘警戒区域，人员与 10 类动物进入即触发（进入沿触发、30 秒冷却），双端报警音（网页 + 服务端电脑）并记录事件，区域持久化、重启不丢
- **模型热切换**：网页上随时在 `worldv2 / m / n` 权重间切换，画面不中断
- **性能指标**：管线帧率、推理耗时（均值 / P95）、ID 切换次数、GPU 显存、趋势曲线
- **画面方向校正**：旋转 0/90/180/270°、水平 / 垂直镜像，即时生效
- **数据集与评估工具链**：采集场景帧 → 人工标注 → mAP 精度评估 → imgsz × conf 参数扫描基准

## 快速开始

环境要求：Windows / Linux，Python 3.8+（建议 3.10+）。有 NVIDIA 显卡时强烈建议安装 GPU 版 PyTorch。

```bash
# 1. 克隆并进入项目
git clone https://github.com/xuhe51719-sketch/yolo-World-camera-detection.git
cd yolo-World-camera-detection

# 2. 创建并激活虚拟环境（Windows PowerShell）
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. 安装依赖
pip install -r requirements.txt

# 3.5 （可选，推荐）GPU 版 PyTorch，按本机 CUDA 版本选择
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 4. 启动
python app.py
```

启动后打开浏览器访问 <http://localhost:5000>。首次运行会自动下载模型权重（需联网）；按 `Ctrl+C` 优雅退出并释放摄像头。

> Linux / macOS 用户把激活命令换成 `source .venv/bin/activate`。

### 命令行参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--url` | 无（用本地摄像头） | 手机 IP 摄像头流地址，如 `http://192.168.1.100:8080/video` |
| `--host` | `127.0.0.1` | 监听地址，**默认仅本机**；局域网访问需显式传 `0.0.0.0` |
| `--port` | `5000` | Web 服务端口 |

### 常用环境变量

完整列表见 [使用指南.md](./使用指南.md) 第 2.3 节。

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `YOLO_MODEL` | `yolov8x-worldv2.pt` | 模型权重，可换 `yolov8m.pt` / `yolov8n.pt`（固定 80 类，更快） |
| `YOLO_IMGSZ` | `640` | 检测输入分辨率，调大更准但更慢 |
| `YOLO_HALF` | `1` | FP16 半精度（仅 CUDA 可用时生效） |
| `DETECTION_INTERVAL` | `2` | 隔帧推理间隔，增大提速、减小更实时 |
| `PHONE_CAMERA_URL` | 空 | 手机流地址，效果等同 `--url` |
| `RECORD_ENABLED` | `1` | 滚动录制开关，设 `0` 关闭 |

## 效果预览

<!-- 欢迎补充运行截图：放在仓库 screenshots/ 目录后在此引用 -->
<!-- ![主界面](screenshots/main.png) -->
<!-- ![区域入侵报警](screenshots/alarm.png) -->

## Web API

共 16 个端点（`/health` 为轻量健康检查，供看门狗使用）：

| 端点 | 方法 | 用途 |
|---|---|---|
| `/` | GET | 网页主界面 |
| `/video_feed` | GET | MJPEG 实时视频流 |
| `/detections` | GET | 最新检测结果（类别 / 置信度 / 跟踪 ID / 坐标） |
| `/model_info` | GET | 模型信息与类别表 |
| `/source` | GET | 图像源状态（地址 / 连接状态 / 黑屏警告 / 方向） |
| `/metrics` | GET | 性能指标（帧率 / 推理耗时 / 显存等） |
| `/set_source` | POST | 切换图像源（手机流 ↔ 本地摄像头），地址校验拒绝非法输入 |
| `/set_orientation` | POST | 设置画面旋转 / 镜像 |
| `/zones` | GET / POST | 警戒区域读取与整体替换（归一化坐标，持久化到 `zones.json`） |
| `/events` | GET | 入侵事件列表与报警状态 |
| `/recordings/status` | GET | 录制状态（段数 / 大小 / 磁盘剩余 / 停录原因） |
| `/recordings/clips` | GET | 已关闭录制段列表 |
| `/recordings/stream` | GET | 回放指定段（`?clip=xx.mp4&speed=1`，倍速与文件名白名单校验） |
| `/alert/test` | POST | 手动触发一次测试报警 |
| `/health` | GET | 健康检查（是否运行 / 最近是否有新帧 / 帧龄） |

## 数据集与精度评估工具

```bash
# 从手机流采集场景帧（--url 必填）
python tools/capture_frames.py --url http://192.168.1.100:8080/video --count 60

# 人工标注后，用 mAP 客观评估精度
python tools/eval_dataset.py --data dataset/data.yaml

# imgsz × conf 参数扫描，量化"速度—检出"权衡
python tools/sweep.py --images dataset/images
```

`datasets/world-monitoring-v4-121/` 为随仓库附带的黄金评估数据集（Roboflow 导出格式）。

## 运行测试

```bash
pip install -r requirements-dev.txt
python -m pytest
```

共 245 个单元测试，全部使用桩 / 假模型，不依赖真实摄像头与 GPU，几秒内跑完。

## 项目结构

```
├── app.py                 # 程序入口：参数解析、后台线程启动、Flask 服务
├── config.py              # 集中配置：环境变量读取、路径锚定
├── config_classes.py      # 124 类检测类别表 + 报警类别（单一来源）
├── camera.py              # 摄像头自动选优 / 手机流重连 / 地址纠错 / 画面方向
├── detector.py            # 模型惰性单例与热切换 / 检测循环 / 检测框绘制
├── recorder.py            # 滚动录制 / 回放与删除路由
├── zones.py               # 多边形区域入侵报警 / 事件 / 报警音 / 持久化
├── routes.py              # Flask 路由
├── templates/index.html   # 前端页面（自包含，无构建步骤）
├── tools/                 # 数据采集 / mAP 评估 / 参数扫描等工具
├── tests/                 # 245 个单元测试
├── datasets/              # 黄金评估数据集
└── 使用指南.md             # 详尽使用文档（环境变量全表 / 操作详解 / 排障手册）
```

各模块导入时均无副作用（不加载模型、不连摄像头、不起线程），便于二次开发与测试。

## 常见问题

摄像头黑屏、手机流连不上、帧率低、显存不足、端口占用等问题的排查手册见 [使用指南.md](./使用指南.md) 第 9 节「常见问题排障」。

最常见的两个坑：

1. **画面全黑**：先检查笔记本摄像头的物理隐私滑盖是否打开（最常见原因）；
2. **局域网设备打不开网页**：默认只监听本机，需 `python app.py --host 0.0.0.0` 并放行防火墙。

## 文档

- [使用指南.md](./使用指南.md) —— 完整使用文档：环境变量全表、前端操作详解、API 契约、工具链用法、排障手册
- [开发记录.md](./开发记录.md) —— 开发过程与设计决策记录
- [eval_reports/](./eval_reports/) —— 精度诊断与评估报告

## 许可证

本项目基于 [MIT License](./LICENSE) 开源。
