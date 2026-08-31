# -*- coding: utf-8 -*-
"""
开放词汇检测类别表（YOLO-World 专用）
app.py 与 tools/ 下的评估工具共享此文件，保证类别定义单一来源。
增删类别后重启服务生效。类别名必须用英文（CLIP 文本编码器要求）。
"""

DETECTION_CLASSES = [
    # --- COCO 80 类 ---
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
    # --- 补充的日常物品（开放词汇能力）---
    "water dispenser", "trash can", "power strip", "wall outlet", "monitor",
    "door", "window", "curtain", "toy", "charger", "cable", "earphones",
    "headphones", "glasses", "pen", "notebook", "paper", "shoe", "tissue box",
    "kettle", "pot", "pan", "fruit", "router", "fan", "air conditioner",
    "pillow", "blanket", "towel", "soap", "mirror", "wallet", "keys",
    "tablet", "smart speaker", "camera", "desk lamp", "calculator", "stapler",
    "mug", "thermos", "water bottle", "slippers", "plastic bag",
]

# 区域入侵报警关注的类别（人员与动物）：zones.py 检查线程仅对这些类别判定入侵。
# 取 COCO 类别中的行人/动物子集；frozenset 保证查找 O(1) 且不可篡改。
ALERT_CLASSES = frozenset({
    "person", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe",
})
