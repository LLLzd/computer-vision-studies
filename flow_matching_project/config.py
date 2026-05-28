"""
Flow Matching 项目配置文件
==========================

本配置文件定义了 Flow Matching 模型的所有超参数和路径设置。

Flow Matching 简介：
-------------------
Flow Matching 是一种生成模型技术，通过学习一个连续的流（flow）来将简单分布（如高斯噪声）
转换为复杂的数据分布。与扩散模型不同，Flow Matching 直接参数化向量场，而不是噪声预测。

核心思想：
1. 定义一个时间相关的路径 x(t)，从噪声 x(0) 到数据 x(1)
2. 学习这个路径的向量场 v(x, t)，使得 dx/dt = v(x, t)
3. 通过求解 ODE 来生成新样本

作者：教学项目
日期：2026-05
"""

# ============================================
# 导入必要的库
# ============================================

import os  # 操作系统接口，用于路径操作
import torch  # PyTorch 深度学习框架

# ============================================
# 项目路径配置
# ============================================

# 获取当前文件所在的目录路径
# __file__ 是当前文件的路径，os.path.abspath 转换为绝对路径
# os.path.dirname 获取文件所在的目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 输出目录：用于保存训练好的模型、生成的样本等
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')

# 模型保存目录：保存训练好的模型权重
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')

# 样本保存目录：保存生成的样本图像
SAMPLE_DIR = os.path.join(OUTPUT_DIR, 'samples')

# 日志保存目录：保存训练日志和可视化结果
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')

# 创建所有必要的目录
# exist_ok=True 表示如果目录已存在则不报错
os.makedirs(OUTPUT_DIR, exist_ok=True)  # 创建输出目录
os.makedirs(MODEL_DIR, exist_ok=True)   # 创建模型目录
os.makedirs(SAMPLE_DIR, exist_ok=True)  # 创建样本目录
os.makedirs(LOG_DIR, exist_ok=True)     # 创建日志目录

# ============================================
# 设备配置
# ============================================

# 检测可用的计算设备
# torch.cuda.is_available() 检查 CUDA GPU 是否可用
# torch.backends.mps.is_available() 检查 Apple Silicon GPU 是否可用
if torch.cuda.is_available():
    # 如果有 NVIDIA GPU，使用 CUDA
    DEVICE = torch.device('cuda')
    print(f"使用设备: CUDA GPU ({torch.cuda.get_device_name(0)})")
elif torch.backends.mps.is_available():
    # 如果有 Apple Silicon (M1/M2/M3)，使用 MPS
    DEVICE = torch.device('mps')
    print("使用设备: Apple Silicon GPU (MPS)")
else:
    # 否则使用 CPU
    DEVICE = torch.device('cpu')
    print("使用设备: CPU")

# ============================================
# 数据配置
# ============================================

# 数据集名称：我们使用 MNIST 作为教学示例
# MNIST 是手写数字数据集，包含 60000 张训练图像和 10000 张测试图像
# 每张图像是 28x28 的灰度图像
DATASET_NAME = 'mnist'

# 图像尺寸：MNIST 原始尺寸是 28x28
# 我们保持原始尺寸以简化处理
IMAGE_SIZE = 28

# 图像通道数：MNIST 是灰度图像，所以通道数为 1
# 如果是彩色图像（如 CIFAR-10），通道数为 3
NUM_CHANNELS = 1

# 数据维度：展平后的图像维度
# 28 * 28 = 784，这是 Flow Matching 模型的输入/输出维度
DATA_DIM = IMAGE_SIZE * IMAGE_SIZE  # 784

# ============================================
# Flow Matching 核心参数
# ============================================

# 时间步数：用于数值积分的步数
# Flow Matching 通过 ODE 求解器生成样本，需要离散化时间
# 更多步数 = 更精确的积分 = 更好的生成质量，但速度更慢
NUM_TIMESTEPS = 100  # 生成时使用的积分步数

# 积分方法：ODE 求解器的类型
# 'euler': 欧拉方法，最简单但精度较低
# 'rk4': 四阶龙格-库塔方法，精度更高但计算量更大
INTEGRATION_METHOD = 'euler'

# ============================================
# 网络架构参数
# ============================================

# 向量场网络的隐藏层维度
# 这个网络预测向量场 v(x, t)，即数据点在每个时间步的移动方向
# 更大的隐藏层 = 更强的表达能力 = 更好的生成质量，但训练更慢
HIDDEN_DIM = 256  # 隐藏层神经元数量

# 隐藏层数量：网络的深度
# 更多层 = 更深的网络 = 可以学习更复杂的模式
NUM_HIDDEN_LAYERS = 4  # 隐藏层数量

# 时间嵌入维度：将时间 t 编码为高维向量
# 时间嵌入帮助网络理解当前处于流的哪个阶段
TIME_EMBED_DIM = 64  # 时间嵌入的维度

# 激活函数：神经网络中使用的非线性激活
# 'relu': ReLU(x) = max(0, x)，简单高效
# 'silu': SiLU(x) = x * sigmoid(x)，更平滑
# 'tanh': tanh(x)，输出范围 [-1, 1]
ACTIVATION = 'silu'  # 使用 SiLU 激活函数

# ============================================
# 训练参数
# ============================================

# 批次大小：每次训练迭代使用的样本数量
# 更大的批次 = 更稳定的梯度 = 更快的训练，但需要更多内存
BATCH_SIZE = 128  # 每批 128 张图像

# 学习率：优化器更新权重的步长
# 太大：训练不稳定，可能发散
# 太小：训练太慢，可能陷入局部最优
LEARNING_RATE = 1e-3  # 0.001，常用的起始学习率

# 训练轮数：完整遍历数据集的次数
# 更多轮数 = 更充分的训练，但可能过拟合
NUM_EPOCHS = 50  # 训练 50 个 epoch

# 优化器类型
# 'adam': 自适应学习率优化器，最常用
# 'adamw': Adam 的改进版，更好的权重衰减
OPTIMIZER = 'adamw'

# 权重衰减：L2 正则化系数，防止过拟合
# 通过惩罚大的权重值来提高泛化能力
WEIGHT_DECAY = 1e-5  # 0.00001

# 学习率调度器类型
# 'constant': 保持学习率不变
# 'cosine': 余弦退火，学习率从初始值逐渐降到 0
# 'linear': 线性衰减
LR_SCHEDULER = 'cosine'

# 预热步数：学习率从 0 逐渐增加到初始值的步数
# 预热可以帮助训练开始时更稳定
WARMUP_STEPS = 500  # 前 500 步进行预热

# ============================================
# 采样和生成参数
# ============================================

# 生成样本数量：每次生成的图像数量
NUM_SAMPLES = 64  # 生成 64 张图像（8x8 网格）

# 噪声尺度：初始噪声的标准差
# Flow Matching 从高斯噪声开始，通过流变换到数据分布
NOISE_SCALE = 1.0  # 标准正态分布

# ============================================
# 保存和日志参数
# ============================================

# 保存间隔：每多少步保存一次模型检查点
SAVE_INTERVAL = 5  # 每 5 个 epoch 保存一次

# 采样间隔：每多少步生成一次样本用于可视化
SAMPLE_INTERVAL = 1  # 每 1 个 epoch 采样一次

# 日志间隔：每多少步打印一次训练日志
LOG_INTERVAL = 100  # 每 100 步打印一次

# ============================================
# 可视化参数
# ============================================

# 可视化网格大小：生成样本的网格布局
VIS_GRID_SIZE = 8  # 8x8 = 64 张图像

# 图像保存格式
IMAGE_FORMAT = 'png'  # 使用 PNG 格式保存图像

# ============================================
# Flow Matching 数学原理说明
# ============================================

"""
Flow Matching 的数学基础：
=========================

1. 概率路径 (Probability Path)：
   定义一个时间相关的概率分布 p_t(x)，其中 t ∈ [0, 1]
   - t = 0: p_0(x) 是简单分布（如高斯噪声）
   - t = 1: p_1(x) 是目标数据分布

2. 条件向量场 (Conditional Vector Field)：
   对于每个数据样本 x_1，定义一条从噪声到该样本的路径：
   x_t = (1 - t) * x_0 + t * x_1
   
   其中：
   - x_0 ~ N(0, I) 是随机噪声
   - x_1 是真实数据样本
   - t 是时间参数

3. 向量场 (Vector Field)：
   路径的导数给出向量场：
   v(x_t, t) = dx_t / dt = x_1 - x_0
   
   这个向量场告诉我们每个点在每个时刻应该往哪个方向移动。

4. 训练目标：
   学习一个神经网络 v_θ(x, t) 来近似真实的向量场
   损失函数：L = E_{t, x_0, x_1} [||v_θ(x_t, t) - (x_1 - x_0)||^2]
   
   其中 x_t = (1-t)*x_0 + t*x_1 是插值路径上的点。

5. 生成过程：
   从噪声 x_0 ~ N(0, I) 开始，求解 ODE：
   dx/dt = v_θ(x, t),  x(0) = x_0
   
   最终得到的 x(1) 就是从数据分布中采样的结果。

与扩散模型的对比：
==================

扩散模型：
- 前向过程：逐步添加噪声 x_t = √(α_t) * x_0 + √(1-α_t) * ε
- 反向过程：学习去噪，预测噪声 ε
- 生成：需要多步迭代去噪

Flow Matching：
- 直接定义从噪声到数据的线性路径
- 学习向量场（速度场）
- 生成：求解 ODE，可以灵活选择积分步数

优势：
- 更直观的几何解释
- 更灵活的生成速度（可以调整积分步数）
- 训练目标更简单（直接回归向量场）
"""

# ============================================
# 打印配置信息
# ============================================

def print_config():
    """
    打印当前配置信息，方便调试和记录
    
    这个函数会在训练开始时调用，显示所有重要的配置参数
    """
    print("\n" + "=" * 60)
    print("Flow Matching 配置信息")
    print("=" * 60)
    print(f"设备: {DEVICE}")
    print(f"数据集: {DATASET_NAME}")
    print(f"数据维度: {DATA_DIM} ({IMAGE_SIZE}x{IMAGE_SIZE})")
    print(f"隐藏层维度: {HIDDEN_DIM}")
    print(f"隐藏层数量: {NUM_HIDDEN_LAYERS}")
    print(f"时间嵌入维度: {TIME_EMBED_DIM}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"学习率: {LEARNING_RATE}")
    print(f"训练轮数: {NUM_EPOCHS}")
    print(f"时间步数: {NUM_TIMESTEPS}")
    print(f"积分方法: {INTEGRATION_METHOD}")
    print("=" * 60 + "\n")

# 当这个文件被导入时，打印配置信息
print_config()
