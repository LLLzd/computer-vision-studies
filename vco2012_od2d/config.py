"""配置文件

集中管理所有配置参数，包括数据集路径、模型参数、训练参数等。
"""

import os

# 数据集路径
DATA_DIR = 'data'
JPEGIMAGES_DIR = os.path.join(DATA_DIR, 'VOC2012', 'JPEGImages')
ANNOTATIONS_DIR = os.path.join(DATA_DIR, 'VOC2012', 'Annotations')
OUTPUT_DIR = 'outputs'

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# VOC 2012 类别
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# 类别到索引的映射
CLS_TO_IDX = {cls: i for i, cls in enumerate(VOC_CLASSES)}
NUM_CLASSES = len(VOC_CLASSES)

# 图像尺寸配置
IMAGE_SIZE = (512, 512)  # 输入图像尺寸
HEATMAP_SIZE = (128, 128)  # Heatmap尺寸 (1/4 缩放)

# 模型配置
INPUT_CHANNELS = 3  # 输入通道数
HIDDEN_CHANNELS = 64  # 隐藏通道数

# Box平均尺寸（在512x512尺度下，用于初始化wh预测头）
# 这些值是通过统计VOC2012数据集计算得到的
AVG_BOX_WIDTH = 162.34  # 平均宽度
AVG_BOX_HEIGHT = 177.43  # 平均高度

# 训练配置
BATCH_SIZE = 8  # 批量大小（降低到2，适应内存限制）
NUM_EPOCHS = 50  # 训练轮次
LEARNING_RATE = 1e-4  # 初始学习率
NUM_WORKERS = 2  # 数据加载线程数（降低到2，减少内存占用）

# 学习率调度配置
WARMUP_EPOCHS = 5  # 预热epoch数量
WARMUP_LR = 1e-6  # 预热起始学习率
MIN_LR = 1e-6  # 最小学习率
LR_SCHEDULER = 'cosine'  # 学习率调度器类型: 'cosine', 'step', 'plateau'

# 损失函数权重
HEATMAP_LOSS_WEIGHT = 1.0  # heatmap损失权重
OFFSET_LOSS_WEIGHT = 0.1  # offset损失权重
WH_LOSS_WEIGHT = 0.2  # wh损失权重

# 推理配置
THRESHOLD = 0.6  # 检测阈值
IOU_THRESHOLD = 0.3  # NMS IoU阈值
MAX_DETECTIONS = 10  # 最大检测数量

# 模型保存路径
WEIGHTS_DIR = os.path.join(OUTPUT_DIR, 'weights')
os.makedirs(WEIGHTS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(WEIGHTS_DIR, 'anchor_free_detector.pth')
