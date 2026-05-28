"""
Flow Matching 网络模型
=====================

本文件定义了 Flow Matching 的核心神经网络组件：
1. 时间嵌入层：将时间 t 编码为高维向量
2. 向量场网络：预测数据点在每个时间步的移动方向
3. Flow Matching 模型：整合上述组件的完整模型

核心概念：
---------
在 Flow Matching 中，我们需要学习一个向量场 v(x, t)，它描述了：
- 对于任意数据点 x
- 在任意时间 t ∈ [0, 1]
- 该点应该往哪个方向移动（dx/dt = v(x, t)）

这个向量场将简单分布（噪声）变换为复杂数据分布。

作者：教学项目
日期：2026-05
"""

# ============================================
# 导入必要的库
# ============================================

import math  # 数学函数，用于计算位置编码
import torch  # PyTorch 深度学习框架
import torch.nn as nn  # 神经网络模块
from config import (  # 从配置文件导入参数
    DATA_DIM,          # 数据维度（784 for MNIST）
    HIDDEN_DIM,        # 隐藏层维度
    NUM_HIDDEN_LAYERS, # 隐藏层数量
    TIME_EMBED_DIM,    # 时间嵌入维度
    ACTIVATION,        # 激活函数类型
)


# ============================================
# 时间嵌入模块
# ============================================

class SinusoidalTimeEmbedding(nn.Module):
    """
    正弦时间嵌入层
    
    将标量时间 t ∈ [0, 1] 编码为高维向量。
    这类似于 Transformer 中的位置编码，让网络能够理解"时间"的概念。
    
    数学原理：
    ---------
    对于时间 t，我们计算：
    embed[2i]   = sin(t * 10000^(2i/d))
    embed[2i+1] = cos(t * 10000^(2i/d))
    
    其中 d 是嵌入维度，i 是维度索引。
    
    这种编码的优势：
    1. 连续性：相邻时间点的嵌入向量也相邻
    2. 唯一性：不同时间点有不同的嵌入
    3. 可学习：网络可以从嵌入中提取时间信息
    
    例子：
    -----
    t = 0.5, embed_dim = 64
    输出: [sin(0.5*w_0), cos(0.5*w_0), sin(0.5*w_1), cos(0.5*w_1), ...]
    其中 w_i = 10000^(2i/64) 是频率
    """
    
    def __init__(self, embed_dim):
        """
        初始化时间嵌入层
        
        参数：
        -----
        embed_dim : int
            时间嵌入的维度
            例如：64 表示时间会被编码为 64 维向量
        """
        # 调用父类 nn.Module 的初始化方法
        # 这是 PyTorch 中定义自定义模块的标准做法
        super().__init__()
        
        # 保存嵌入维度，用于后续计算
        self.embed_dim = embed_dim
    
    def forward(self, t):
        """
        前向传播：计算时间嵌入
        
        参数：
        -----
        t : torch.Tensor
            时间值，形状为 (batch_size,) 或 (batch_size, 1)
            取值范围通常在 [0, 1]
            - t = 0 对应噪声分布
            - t = 1 对应数据分布
        
        返回：
        -----
        embed : torch.Tensor
            时间嵌入向量，形状为 (batch_size, embed_dim)
        
        计算过程：
        ---------
        1. 确保输入形状正确
        2. 计算频率因子
        3. 应用正弦和余弦函数
        4. 拼接成最终嵌入
        """
        # 确保时间 t 的形状为 (batch_size, 1)
        # 如果 t 是一维的 (batch_size,)，添加一个维度
        if t.dim() == 1:
            t = t.unsqueeze(1)  # (batch_size,) -> (batch_size, 1)
        
        # 计算半维度，因为我们要同时计算 sin 和 cos
        # 例如 embed_dim = 64，则 half_dim = 32
        half_dim = self.embed_dim // 2
        
        # 计算频率因子
        # 数学公式：w_i = 10000^(-2i/embed_dim) = exp(-2i * log(10000) / embed_dim)
        # 这里使用对数形式计算，避免数值不稳定
        # 
        # 具体步骤：
        # 1. math.log(10000) = 9.2103... (自然对数)
        # 2. 创建序列 [0, 1, 2, ..., half_dim-1]
        # 3. 计算 -2 * i * log(10000) / embed_dim
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=t.device) / half_dim
        )
        # freqs 的形状：(half_dim,)
        # freqs[i] = 10000^(-i/half_dim)
        
        # 计算时间与频率的乘积
        # t 的形状：(batch_size, 1)
        # freqs 的形状：(half_dim,)
        # t * freqs 的形状：(batch_size, half_dim)（广播机制）
        args = t * freqs  # (batch_size, half_dim)
        
        # 应用正弦和余弦函数
        # torch.sin(args) 和 torch.cos(args) 的形状都是 (batch_size, half_dim)
        # torch.cat 将它们在最后一个维度上拼接
        # 结果形状：(batch_size, 2 * half_dim) = (batch_size, embed_dim)
        embed = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        return embed


# ============================================
# 时间感知的多层感知机 (MLP)
# ============================================

class TimeAwareMLP(nn.Module):
    """
    时间感知的多层感知机
    
    这是一个带时间条件的 MLP，它接受：
    - 数据点 x（当前状态）
    - 时间 t（当前时刻）
    
    并输出向量场 v(x, t)。
    
    网络结构：
    ---------
    输入: [x, time_embed(t)]  (拼接)
    -> Linear -> Activation -> ... (多层)
    -> Linear
    输出: v(x, t)  (向量场)
    
    为什么需要时间感知？
    -------------------
    在 Flow Matching 中，向量场是时间相关的：
    - t 接近 0：数据点在噪声分布中，需要大幅移动
    - t 接近 1：数据点接近目标，需要精细调整
    
    网络需要知道当前时间，才能正确预测移动方向。
    """
    
    def __init__(self, data_dim, hidden_dim, num_layers, time_embed_dim, activation='silu'):
        """
        初始化时间感知 MLP
        
        参数：
        -----
        data_dim : int
            数据维度（例如 MNIST 展平后是 784）
        hidden_dim : int
            隐藏层维度（例如 256）
        num_layers : int
            隐藏层数量（例如 4）
        time_embed_dim : int
            时间嵌入维度（例如 64）
        activation : str
            激活函数类型，可选 'relu', 'silu', 'tanh'
        """
        super().__init__()
        
        # 保存参数
        self.data_dim = data_dim  # 数据维度
        self.time_embed_dim = time_embed_dim  # 时间嵌入维度
        
        # 创建时间嵌入层
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        
        # 选择激活函数
        # nn.ReLU: ReLU(x) = max(0, x)，简单高效
        # nn.SiLU: SiLU(x) = x * sigmoid(x)，平滑且性能好
        # nn.Tanh: tanh(x)，输出范围 [-1, 1]
        if activation == 'relu':
            self.act = nn.ReLU()  # ReLU 激活函数
        elif activation == 'silu':
            self.act = nn.SiLU()  # SiLU 激活函数（推荐）
        elif activation == 'tanh':
            self.act = nn.Tanh()  # Tanh 激活函数
        else:
            # 默认使用 SiLU
            self.act = nn.SiLU()
        
        # 构建网络层
        # 使用 nn.ModuleList 来存储层，这样 PyTorch 可以正确注册参数
        
        # 第一层：输入维度 = data_dim + time_embed_dim
        # 因为我们要拼接数据 x 和时间嵌入
        layers = []
        
        # 输入层：将 (data_dim + time_embed_dim) 映射到 hidden_dim
        # 例如：(784 + 64) -> 256
        layers.append(nn.Linear(data_dim + time_embed_dim, hidden_dim))
        layers.append(self.act)  # 添加激活函数
        
        # 隐藏层：hidden_dim -> hidden_dim
        # 例如：256 -> 256，重复 num_layers - 1 次
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))  # 线性变换
            layers.append(self.act)  # 激活函数
        
        # 输出层：hidden_dim -> data_dim
        # 输出向量场 v(x, t)，维度与数据相同
        # 例如：256 -> 784
        layers.append(nn.Linear(hidden_dim, data_dim))
        
        # 将层列表转换为 nn.Sequential
        # nn.Sequential 会按顺序执行各层
        self.net = nn.Sequential(*layers)
    
    def forward(self, x, t):
        """
        前向传播：计算向量场 v(x, t)
        
        参数：
        -----
        x : torch.Tensor
            数据点，形状为 (batch_size, data_dim)
            例如：(128, 784) 表示 128 个展平的 MNIST 图像
        t : torch.Tensor
            时间值，形状为 (batch_size,) 或 (batch_size, 1)
            取值范围 [0, 1]
        
        返回：
        -----
        v : torch.Tensor
            向量场，形状为 (batch_size, data_dim)
            表示每个数据点在当前时刻的移动方向
        
        计算过程：
        ---------
        1. 计算时间嵌入 embed(t)
        2. 拼接 [x, embed(t)]
        3. 通过 MLP 计算向量场
        """
        # 计算时间嵌入
        # t 的形状：(batch_size,) -> embed 的形状：(batch_size, time_embed_dim)
        t_embed = self.time_embed(t)
        
        # 拼接数据和时间嵌入
        # x 的形状：(batch_size, data_dim)
        # t_embed 的形状：(batch_size, time_embed_dim)
        # 拼接后形状：(batch_size, data_dim + time_embed_dim)
        x_t = torch.cat([x, t_embed], dim=-1)
        
        # 通过网络计算向量场
        # 输入形状：(batch_size, data_dim + time_embed_dim)
        # 输出形状：(batch_size, data_dim)
        v = self.net(x_t)
        
        return v


# ============================================
# Flow Matching 主模型
# ============================================

class FlowMatchingModel(nn.Module):
    """
    Flow Matching 主模型
    
    这是 Flow Matching 的核心模型，整合了：
    1. 时间嵌入
    2. 向量场网络
    3. 训练和采样方法
    
    训练阶段：
    ---------
    输入：数据样本 x_1
    过程：
    1. 采样噪声 x_0 ~ N(0, I)
    2. 采样时间 t ~ Uniform(0, 1)
    3. 计算插值点 x_t = (1-t) * x_0 + t * x_1
    4. 计算目标向量场 v_target = x_1 - x_0
    5. 预测向量场 v_pred = network(x_t, t)
    6. 计算损失 L = ||v_pred - v_target||^2
    
    采样阶段：
    ---------
    输入：噪声 x_0 ~ N(0, I)
    过程：求解 ODE dx/dt = v(x, t), x(0) = x_0
    输出：生成样本 x(1)
    """
    
    def __init__(self, data_dim=DATA_DIM, hidden_dim=HIDDEN_DIM, 
                 num_layers=NUM_HIDDEN_LAYERS, time_embed_dim=TIME_EMBED_DIM,
                 activation=ACTIVATION):
        """
        初始化 Flow Matching 模型
        
        参数：
        -----
        data_dim : int
            数据维度，默认从 config 导入
        hidden_dim : int
            隐藏层维度
        num_layers : int
            隐藏层数量
        time_embed_dim : int
            时间嵌入维度
        activation : str
            激活函数类型
        """
        super().__init__()
        
        # 保存数据维度
        self.data_dim = data_dim
        
        # 创建向量场网络
        # 这个网络预测 v(x, t)
        self.vector_field = TimeAwareMLP(
            data_dim=data_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            time_embed_dim=time_embed_dim,
            activation=activation
        )
    
    def forward(self, x, t):
        """
        前向传播：计算向量场
        
        参数：
        -----
        x : torch.Tensor
            数据点，形状 (batch_size, data_dim)
        t : torch.Tensor
            时间，形状 (batch_size,)
        
        返回：
        -----
        v : torch.Tensor
            向量场，形状 (batch_size, data_dim)
        """
        return self.vector_field(x, t)
    
    def compute_loss(self, x_1):
        """
        计算训练损失
        
        这是 Flow Matching 的核心训练目标。
        
        参数：
        -----
        x_1 : torch.Tensor
            真实数据样本，形状 (batch_size, data_dim)
            例如：(128, 784) 表示 128 个 MNIST 图像
        
        返回：
        -----
        loss : torch.Tensor
            标量损失值
        
        数学原理：
        ---------
        Flow Matching 的训练目标是学习向量场 v(x, t)，
        使得它能将噪声分布变换为数据分布。
        
        具体步骤：
        1. 对于每个数据样本 x_1，我们定义一条从噪声到该样本的路径
        2. 路径：x_t = (1-t) * x_0 + t * x_1，其中 x_0 ~ N(0, I)
        3. 路径的导数（目标向量场）：v_target = dx_t/dt = x_1 - x_0
        4. 训练目标：让网络预测的 v(x_t, t) 接近 v_target
        
        损失函数：
        L = E_{t, x_0, x_1} [||v_θ(x_t, t) - (x_1 - x_0)||^2]
        """
        # 获取批次大小
        batch_size = x_1.shape[0]
        
        # 获取设备（CPU 或 GPU）
        device = x_1.device
        
        # 步骤 1：采样噪声 x_0 ~ N(0, I)
        # torch.randn 生成标准正态分布的随机数
        # 形状：(batch_size, data_dim)
        x_0 = torch.randn_like(x_1)  # 与 x_1 相同形状的标准正态噪声
        
        # 步骤 2：采样时间 t ~ Uniform(0, 1)
        # torch.rand 生成 [0, 1) 均匀分布的随机数
        # 形状：(batch_size,)
        t = torch.rand(batch_size, device=device)
        
        # 步骤 3：计算插值点 x_t
        # 数学公式：x_t = (1 - t) * x_0 + t * x_1
        # 
        # t = 0: x_t = x_0（噪声）
        # t = 1: x_t = x_1（数据）
        # t = 0.5: x_t 是噪声和数据的中间点
        #
        # 需要调整 t 的形状以便广播
        # t 的形状：(batch_size,) -> (batch_size, 1)
        t_expanded = t.unsqueeze(1)  # (batch_size, 1)
        
        # 计算插值
        # (1 - t) * x_0: 向噪声方向加权
        # t * x_1: 向数据方向加权
        x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
        
        # 步骤 4：计算目标向量场
        # 数学公式：v_target = x_1 - x_0
        # 这是从噪声到数据的方向
        v_target = x_1 - x_0
        
        # 步骤 5：预测向量场
        # 使用神经网络预测 v(x_t, t)
        v_pred = self.vector_field(x_t, t)
        
        # 步骤 6：计算损失
        # 使用均方误差（MSE）
        # ||v_pred - v_target||^2 的均值
        loss = torch.mean((v_pred - v_target) ** 2)
        
        return loss
    
    def sample(self, num_samples, num_steps=100, device='cpu', method='euler'):
        """
        从模型生成样本
        
        通过求解 ODE 从噪声生成数据样本。
        
        参数：
        -----
        num_samples : int
            要生成的样本数量
        num_steps : int
            ODE 求解的步数（更多步数 = 更精确）
        device : str 或 torch.device
            计算设备
        method : str
            积分方法，'euler' 或 'rk4'
        
        返回：
        -----
        samples : torch.Tensor
            生成的样本，形状 (num_samples, data_dim)
        
        数学原理：
        ---------
        生成过程是求解以下 ODE：
        dx/dt = v(x, t),  x(0) = x_0 ~ N(0, I)
        
        欧拉方法（最简单）：
        x(t + dt) = x(t) + dt * v(x(t), t)
        
        四阶龙格-库塔（更精确）：
        k1 = v(x, t)
        k2 = v(x + 0.5*dt*k1, t + 0.5*dt)
        k3 = v(x + 0.5*dt*k2, t + 0.5*dt)
        k4 = v(x + dt*k3, t + dt)
        x(t + dt) = x(t) + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        """
        # 步骤 1：采样初始噪声 x_0 ~ N(0, I)
        x = torch.randn(num_samples, self.data_dim, device=device)
        
        # 步骤 2：计算时间步长
        # 从 t=0 到 t=1，分成 num_steps 步
        dt = 1.0 / num_steps  # 时间步长
        
        # 步骤 3：数值积分求解 ODE
        if method == 'euler':
            # 欧拉方法：简单但精度较低
            for step in range(num_steps):
                # 当前时间
                t = step / num_steps  # 从 0 到接近 1
                
                # 创建时间张量（所有样本使用相同时间）
                t_tensor = torch.full((num_samples,), t, device=device)
                
                # 计算向量场
                v = self.vector_field(x, t_tensor)
                
                # 欧拉更新：x = x + dt * v
                x = x + dt * v
        
        elif method == 'rk4':
            # 四阶龙格-库塔方法：更精确
            for step in range(num_steps):
                t = step / num_steps
                t_tensor = torch.full((num_samples,), t, device=device)
                
                # k1 = v(x, t)
                k1 = self.vector_field(x, t_tensor)
                
                # k2 = v(x + 0.5*dt*k1, t + 0.5*dt)
                t2_tensor = torch.full((num_samples,), t + 0.5 * dt, device=device)
                k2 = self.vector_field(x + 0.5 * dt * k1, t2_tensor)
                
                # k3 = v(x + 0.5*dt*k2, t + 0.5*dt)
                k3 = self.vector_field(x + 0.5 * dt * k2, t2_tensor)
                
                # k4 = v(x + dt*k3, t + dt)
                t3_tensor = torch.full((num_samples,), t + dt, device=device)
                k4 = self.vector_field(x + dt * k3, t3_tensor)
                
                # 更新：x = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
                x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        
        else:
            raise ValueError(f"未知的积分方法: {method}")
        
        return x


# ============================================
# 辅助函数：模型参数统计
# ============================================

def count_parameters(model):
    """
    计算模型的参数数量
    
    参数：
    -----
    model : nn.Module
        PyTorch 模型
    
    返回：
    -----
    total_params : int
        总参数数量
    trainable_params : int
        可训练参数数量
    """
    # 计算总参数数量
    total_params = sum(p.numel() for p in model.parameters())
    
    # 计算可训练参数数量（requires_grad=True）
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


# ============================================
# 测试代码
# ============================================

if __name__ == '__main__':
    # 这段代码在直接运行此文件时执行，用于测试
    
    print("\n" + "=" * 60)
    print("测试 Flow Matching 网络模型")
    print("=" * 60)
    
    # 创建模型
    model = FlowMatchingModel()
    
    # 统计参数
    total, trainable = count_parameters(model)
    print(f"\n模型参数统计:")
    print(f"  总参数数量: {total:,}")
    print(f"  可训练参数数量: {trainable:,}")
    
    # 测试前向传播
    batch_size = 4
    x = torch.randn(batch_size, DATA_DIM)  # 随机数据
    t = torch.rand(batch_size)  # 随机时间
    
    print(f"\n测试输入:")
    print(f"  x 形状: {x.shape}")
    print(f"  t 形状: {t.shape}")
    
    # 计算向量场
    v = model(x, t)
    print(f"\n向量场输出形状: {v.shape}")
    
    # 测试损失计算
    loss = model.compute_loss(x)
    print(f"损失值: {loss.item():.6f}")
    
    # 测试采样
    samples = model.sample(num_samples=2, num_steps=10)
    print(f"\n生成样本形状: {samples.shape}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60 + "\n")
