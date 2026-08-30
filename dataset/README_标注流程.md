# 黄金数据集标注流程

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
    ".venv\Scripts\python.exe" tools\eval_dataset.py --data dataset\data.yaml
