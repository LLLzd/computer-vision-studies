"""
配置文件
定义路径、超参数和数据预处理
"""

# 导入必要的库
import os  # 用于文件路径操作
import torch  # PyTorch 核心库
import numpy as np  # 用于数组操作
from torchvision import transforms  # 用于图像预处理

# 项目根目录 - 获取当前文件所在目录的绝对路径
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据目录 - 存放数据集的路径
DATA_DIR = os.path.join(ROOT_DIR, 'data')

# 模型保存目录 - 存放训练好的模型
MODEL_DIR = os.path.join(ROOT_DIR, 'models')

# 输出目录 - 存放可视化结果和推理结果
OUTPUT_DIR = os.path.join(ROOT_DIR, 'outputs')

# 确保目录存在 - 如果目录不存在则创建
os.makedirs(DATA_DIR, exist_ok=True)  # exist_ok=True 表示如果目录已存在则不报错
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 超参数配置
BATCH_SIZE = 8  # 批次大小，每次训练使用的样本数量（完全体UNet需要更大内存，M1芯片建议8）
LEARNING_RATE = 1e-4  # 学习率，控制参数更新的步长（完全体UNet使用更小学习率）
NUM_EPOCHS = 30  # 训练轮数，完整遍历数据集的次数（完全体UNet需要更多轮数）

# 设备配置 - 优先使用 GPU，如果没有则使用 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 图像尺寸 - 输入模型的图像大小
IMAGE_SIZE = 160  # 增加图像尺寸以保留更多细节信息

# 类别名称 - VOC2012 分割数据集的类别
CLASS_NAMES = [
    'background',  # 0 - 背景
    'aeroplane',   # 1 - 飞机
    'bicycle',     # 2 - 自行车
    'bird',        # 3 - 鸟
    'boat',        # 4 - 船
    'bottle',      # 5 - 瓶子
    'bus',         # 6 - 公交车
    'car',         # 7 - 汽车
    'cat',         # 8 - 猫
    'chair',       # 9 - 椅子
    'cow',         # 10 - 牛
    'diningtable', # 11 - 餐桌
    'dog',         # 12 - 狗
    'horse',       # 13 - 马
    'motorbike',   # 14 - 摩托车
    'person',      # 15 - 人
    'pottedplant', # 16 - 盆栽植物
    'sheep',       # 17 - 羊
    'sofa',        # 18 - 沙发
    'train',       # 19 - 火车
    'tvmonitor'    # 20 - 电视监视器
]

# 类别数量 - 包含背景
NUM_CLASSES = len(CLASS_NAMES)  # 21 个类别

# 数据预处理转换 - 定义图像预处理的步骤
transform = transforms.Compose([
    # 调整图像大小为指定尺寸
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # 输入: PIL图像 (H, W, C) → 输出: PIL图像 (IMAGE_SIZE, IMAGE_SIZE, C)
    # 数据增强
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    transforms.RandomVerticalFlip(p=0.2),  # 随机垂直翻转
    transforms.RandomRotation(10),  # 随机旋转±10度
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 随机调整亮度、对比度、饱和度
    # 将图像转换为张量，并自动将值范围从0-255归一化到0-1
    transforms.ToTensor(),  # 输入: PIL图像 (IMAGE_SIZE, IMAGE_SIZE, C) → 输出: 张量 (C, IMAGE_SIZE, IMAGE_SIZE)，值范围 [0, 1]
    # 归一化，使用 ImageNet 的均值和标准差
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # RGB 通道的均值
        std=[0.229, 0.224, 0.225]     # RGB 通道的标准差
    )  # 输入: 张量 (C, IMAGE_SIZE, IMAGE_SIZE) → 输出: 张量 (C, IMAGE_SIZE, IMAGE_SIZE)，归一化后的值
])

# 标签转换 - 定义标签预处理的步骤
def label_to_tensor(x):
    """
    将标签转换为长整型张量
    处理 Pascal VOC 2012 数据集中的边界标记（255），将其转换为背景（0）
    
    Args:
        x: PIL图像
    
    Returns:
        长整型张量，值为 0-20
    """
    label = np.array(x)
    # 将边界标记 255 转换为背景 0
    label[label == 255] = 0
    return torch.tensor(label, dtype=torch.long)

label_transform = transforms.Compose([
    # 调整标签大小为 128x128，使用最近邻插值
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE), 
        interpolation=transforms.InterpolationMode.NEAREST  # 最近邻插值，保持标签值不变
    ),  # 输入: PIL图像 (H, W) → 输出: PIL图像 (128, 128)
    # 将标签转换为长整型张量
    transforms.Lambda(label_to_tensor)  # 输入: PIL图像 (128, 128) → 输出: 张量 (128, 128)，值为类别索引
])
