"""
配置文件
包含所有路径、超参数和数据预处理配置
"""
# 导入操作系统模块，用于路径操作
import os
# 导入 torchvision 的 transforms 模块，用于数据预处理
import torchvision.transforms as transforms

# ========== 路径配置 ==========
# 获取当前文件所在目录的绝对路径
BASE = os.path.dirname(os.path.abspath(__file__))
# 数据集保存路径
DATA_DIR = os.path.join(BASE, "data")
# 模型保存目录
MODEL_DIR = os.path.join(BASE, "models")
# 输出文件保存目录
OUTPUT_DIR = os.path.join(BASE, "outputs")

# 模型文件保存路径
MODEL_PATH = os.path.join(MODEL_DIR, "mnist_net.pth")
# 训练曲线保存路径
PLOT_PATH = os.path.join(OUTPUT_DIR, "loss_acc_curve.png")
# 推理结果保存路径
INFER_PATH = os.path.join(OUTPUT_DIR, "infer_result.png")
# 可视化结果保存路径
VISUAL_PATH = os.path.join(OUTPUT_DIR, "visualize.png")

# 确保必要的目录存在
# exist_ok=True 表示如果目录已存在则不报错
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 超参数配置 ==========
# 批次大小：每次训练的样本数量
BATCH_SIZE = 64
# 学习率：控制参数更新的步长
LEARNING_RATE = 0.001
# 训练轮数：整个数据集训练的次数
EPOCHS = 10
# 类别数量：MNIST 数据集的类别数（0-9）
NUM_CLASSES = 10

# ========== 数据预处理 ==========
# 定义数据转换流程
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),  # 将 PIL 图像或 numpy 数组转换为张量
    transforms.Normalize((0.1307,), (0.3081,))  # 归一化，使用 MNIST 数据集的均值和标准差
])

# ========== 类别名称 ==========
# MNIST 数据集的类别名称元组
CLASSES = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
