"""
配置文件
定义路径、超参数和数据预处理
"""

# 导入必要的库
import os  # 用于文件路径操作
import torch  # PyTorch 核心库
from torchvision import transforms  # 用于图像预处理

# 项目根目录 - 获取当前文件所在目录的绝对路径
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据目录 - 存放数据集的路径
# 复用 mnist_project 中的 MNIST 数据集
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
BATCH_SIZE = 16  # 批次大小，每次训练使用的样本数量
LEARNING_RATE = 0.001  # 学习率，控制参数更新的步长
NUM_EPOCHS = 10  # 训练轮数，完整遍历数据集的次数

# 设备配置 - 优先使用 GPU，如果没有则使用 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 图像尺寸 - 输入模型的图像大小
IMAGE_SIZE = 28

# 数据预处理转换 - 定义图像预处理的步骤
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # 调整图像大小为 28x28
    transforms.ToTensor(),  # 将图像转换为张量，并自动将值范围从0-255归一化到0-1
    transforms.Normalize(mean=[0.5],  # 归一化，使用 MNIST 的均值和标准差
                         std=[0.5])
])

# 标签转换 - 定义标签预处理的步骤
label_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # 调整标签大小为 28x28
    transforms.ToTensor()  # 将标签转换为张量
])
