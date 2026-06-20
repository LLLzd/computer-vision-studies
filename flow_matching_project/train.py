"""
Flow Matching 训练脚本
=====================

本文件实现了 Flow Matching 模型的完整训练流程：
1. 数据加载：加载 MNIST 数据集
2. 模型初始化：创建 Flow Matching 模型
3. 优化器设置：配置 AdamW 优化器和学习率调度器
4. 训练循环：迭代训练并保存检查点
5. 可视化：定期生成样本用于监控训练进度

训练流程：
---------
对于每个训练批次：
1. 从数据集采样真实数据 x_1
2. 计算损失 L = E[||v_θ(x_t, t) - (x_1 - x_0)||^2]
3. 反向传播更新参数

作者：教学项目
日期：2026-05
"""

# ============================================
# 导入必要的库
# ============================================

import os  # 操作系统接口
import time  # 时间相关功能
import math  # 数学函数（用于学习率调度器）
import torch  # PyTorch 深度学习框架
import torch.nn as nn  # 神经网络模块
from torch.utils.data import DataLoader  # 数据加载器
from torchvision import datasets, transforms  # 数据集和变换
from tqdm import tqdm  # 进度条显示

# 导入项目模块
from config import (  # 配置参数
    DEVICE,              # 计算设备
    BATCH_SIZE,          # 批次大小
    LEARNING_RATE,       # 学习率
    NUM_EPOCHS,          # 训练轮数
    WEIGHT_DECAY,        # 权重衰减
    OPTIMIZER,           # 优化器类型
    LR_SCHEDULER,        # 学习率调度器
    WARMUP_STEPS,        # 预热步数
    SAVE_INTERVAL,       # 保存间隔
    SAMPLE_INTERVAL,     # 采样间隔
    LOG_INTERVAL,        # 日志间隔
    MODEL_DIR,           # 模型保存目录
    SAMPLE_DIR,          # 样本保存目录
    LOG_DIR,             # 日志保存目录
    NUM_SAMPLES,         # 生成样本数量
    NUM_TIMESTEPS,       # 时间步数
    INTEGRATION_METHOD,  # 积分方法
    IMAGE_SIZE,          # 图像尺寸
)
from net import FlowMatchingModel, count_parameters  # 模型和辅助函数


# ============================================
# 数据加载函数
# ============================================

def get_dataloader(batch_size, device='cpu'):
    """
    创建 MNIST 数据加载器
    
    参数：
    -----
    batch_size : int
        每批次的样本数量
    device : str
        计算设备（用于确定是否使用 GPU 预加载）
    
    返回：
    -----
    dataloader : DataLoader
        PyTorch 数据加载器
    
    数据预处理：
    -----------
    1. 转换为灰度图像（MNIST 已经是灰度）
    2. 调整大小为 28x28（MNIST 原始尺寸）
    3. 转换为张量
    4. 归一化到 [0, 1] 范围
    5. 展平为一维向量（784 维）
    """
    # 定义数据预处理变换
    # transforms.Compose 将多个变换组合在一起
    transform = transforms.Compose([
        # 转换为灰度图像
        # MNIST 已经是灰度，但这个变换确保格式正确
        transforms.Grayscale(num_output_channels=1),
        
        # 调整图像大小
        # MNIST 原始尺寸是 28x28，这里显式设置
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        
        # 转换为 PyTorch 张量
        # 将 PIL 图像转换为张量，并归一化到 [0, 1]
        transforms.ToTensor(),
        
        # 标准化为均值 0、标准差 1
        # 这非常重要！因为噪声 x_0 ~ N(0, 1)
        # 数据需要与噪声分布匹配才能让 Flow Matching 正常工作
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    
    # 下载并加载 MNIST 训练数据集
    # root='./data': 数据保存路径
    # train=True: 使用训练集
    # download=True: 如果数据不存在则下载
    # transform=transform: 应用预处理变换
    dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # 创建数据加载器
    # batch_size: 每批次的样本数量
    # shuffle=True: 每个 epoch 开始时打乱数据
    # num_workers=0: 数据加载的工作进程数（0 表示在主进程加载）
    # pin_memory: 如果使用 GPU，将数据固定在内存中加速传输
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == 'cuda')
    )
    
    print(f"数据集大小: {len(dataset)}")
    print(f"批次数量: {len(dataloader)}")
    
    return dataloader


# ============================================
# 展平数据的辅助函数
# ============================================

def flatten_batch(images):
    """
    将图像批次展平为一维向量
    
    参数：
    -----
    images : torch.Tensor
        图像批次，形状 (batch_size, 1, 28, 28)
    
    返回：
    -----
    flattened : torch.Tensor
        展平后的向量，形状 (batch_size, 784)
    
    说明：
    -----
    Flow Matching 需要将图像展平为一维向量。
    例如：28x28 的图像展平为 784 维向量。
    """
    # 获取批次大小
    batch_size = images.shape[0]
    
    # 展平图像
    # images.view(batch_size, -1) 将每张图像展平为一维向量
    # -1 表示自动计算该维度大小（28 * 28 = 784）
    flattened = images.view(batch_size, -1)
    
    return flattened


# ============================================
# 学习率调度器
# ============================================

def get_lr_scheduler(optimizer, num_training_steps, warmup_steps, scheduler_type='cosine'):
    """
    创建学习率调度器
    
    参数：
    -----
    optimizer : torch.optim.Optimizer
        优化器
    num_training_steps : int
        总训练步数
    warmup_steps : int
        预热步数
    scheduler_type : str
        调度器类型，'cosine' 或 'linear'
    
    返回：
    -----
    scheduler : 学习率调度器
    
    学习率调度策略：
    ---------------
    1. 预热阶段（前 warmup_steps 步）：
       学习率从 0 线性增加到初始学习率
       
    2. 主训练阶段：
       - cosine: 余弦退火，学习率平滑下降到 0
       - linear: 线性衰减，学习率线性下降到 0
    
    为什么需要预热？
    ---------------
    训练开始时，模型参数是随机初始化的。
    如果直接使用大的学习率，可能导致训练不稳定。
    预热让模型在开始时使用小学习率，逐渐适应后再增加。
    """
    
    def lr_lambda(current_step):
        """
        学习率乘数函数
        
        参数：
        -----
        current_step : int
            当前训练步数
        
        返回：
        -----
        multiplier : float
            学习率乘数（0 到 1）
        """
        # 预热阶段
        if current_step < warmup_steps:
            # 线性增加：从 0 到 1
            return float(current_step) / float(max(1, warmup_steps))
        
        # 主训练阶段
        progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
        
        if scheduler_type == 'cosine':
            # 余弦退火
            # 数学公式：lr = 0.5 * (1 + cos(π * progress))
            # progress 从 0 到 1，cos 从 1 到 -1，lr 从 1 到 0
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        
        elif scheduler_type == 'linear':
            # 线性衰减
            # 数学公式：lr = 1 - progress
            return max(0.0, 1.0 - progress)
        
        else:
            # 默认：保持学习率不变
            return 1.0
    
    # 创建 LambdaLR 调度器
    # lr_lambda 函数定义了学习率的变化规则
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return scheduler


# ============================================
# 训练一个 epoch 的函数
# ============================================

def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch, log_interval):
    """
    训练一个 epoch
    
    参数：
    -----
    model : FlowMatchingModel
        Flow Matching 模型
    dataloader : DataLoader
        数据加载器
    optimizer : Optimizer
        优化器
    scheduler : LRScheduler
        学习率调度器
    device : torch.device
        计算设备
    epoch : int
        当前 epoch 编号
    log_interval : int
        日志打印间隔
    
    返回：
    -----
    avg_loss : float
        平均损失值
    
    训练过程：
    ---------
    对于每个批次：
    1. 将数据移到设备（GPU/CPU）
    2. 展平图像为一维向量
    3. 计算损失
    4. 反向传播
    5. 更新参数
    6. 更新学习率
    """
    # 将模型设置为训练模式
    # 这会启用 dropout、batch norm 等的训练行为
    model.train()
    
    # 初始化损失累加器
    total_loss = 0.0
    num_batches = 0
    
    # 创建进度条
    # tqdm 显示训练进度、速度、剩余时间等信息
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    # 遍历数据加载器
    for batch_idx, (images, labels) in enumerate(pbar):
        # 注意：labels 是数字标签，Flow Matching 不需要它
        # Flow Matching 是无监督生成模型，只需要图像数据
        
        # 步骤 1：将数据移到设备
        # images 的形状：(batch_size, 1, 28, 28)
        images = images.to(device)
        
        # 步骤 2：展平图像
        # 展平后形状：(batch_size, 784)
        x_1 = flatten_batch(images)
        
        # 步骤 3：清零梯度
        # PyTorch 默认会累加梯度，所以每步都要清零
        optimizer.zero_grad()
        
        # 步骤 4：计算损失
        # 这会调用 model.compute_loss(x_1)
        # 内部会采样噪声、时间、计算插值、预测向量场、计算 MSE 损失
        loss = model.compute_loss(x_1)
        
        # 步骤 5：反向传播
        # loss.backward() 计算损失对所有参数的梯度
        loss.backward()
        
        # 步骤 6：梯度裁剪（可选，提高训练稳定性）
        # 将梯度范数裁剪到最大值 1.0
        # 防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 步骤 7：更新参数
        # optimizer.step() 根据梯度更新模型参数
        optimizer.step()
        
        # 步骤 8：更新学习率
        # scheduler.step() 根据调度策略更新学习率
        scheduler.step()
        
        # 累加损失
        total_loss += loss.item()
        num_batches += 1
        
        # 更新进度条显示
        # 显示当前损失和学习率
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{current_lr:.6f}'
        })
        
        # 定期打印详细日志
        if batch_idx % log_interval == 0:
            print(f'  Batch {batch_idx}/{len(dataloader)}, '
                  f'Loss: {loss.item():.4f}, '
                  f'LR: {current_lr:.6f}')
    
    # 计算平均损失
    avg_loss = total_loss / num_batches
    
    return avg_loss


# ============================================
# 生成样本的函数
# ============================================

def generate_samples(model, num_samples, num_steps, device, epoch, sample_dir):
    """
    生成样本并保存
    
    参数：
    -----
    model : FlowMatchingModel
        训练好的模型
    num_samples : int
        生成样本数量
    num_steps : int
        ODE 求解步数
    device : torch.device
        计算设备
    epoch : int
        当前 epoch（用于命名）
    sample_dir : str
        样本保存目录
    
    生成过程：
    ---------
    1. 将模型设置为评估模式
    2. 从模型采样生成样本
    3. 将样本重塑为图像形状
    4. 保存为图像文件
    """
    # 将模型设置为评估模式
    # 这会禁用 dropout、batch norm 等的训练行为
    model.eval()
    
    # 使用 torch.no_grad() 禁用梯度计算
    # 生成时不需要梯度，可以节省内存和加速计算
    with torch.no_grad():
        # 从模型采样
        # 这会求解 ODE，从噪声生成样本
        samples = model.sample(
            num_samples=num_samples,
            num_steps=num_steps,
            device=device,
            method=INTEGRATION_METHOD
        )
    
    # 将样本重塑为图像形状
    # samples 形状：(num_samples, 784)
    # 重塑为：(num_samples, 1, 28, 28)
    samples = samples.view(num_samples, 1, IMAGE_SIZE, IMAGE_SIZE)
    
    # 限制值范围到 [0, 1]
    # 生成的样本可能超出这个范围，需要裁剪
    samples = torch.clamp(samples, 0, 1)
    
    # 保存样本图像
    save_sample_image(samples, epoch, sample_dir)


def save_sample_image(samples, epoch, sample_dir):
    """
    保存样本图像
    
    参数：
    -----
    samples : torch.Tensor
        样本张量，形状 (num_samples, 1, 28, 28)
    epoch : int
        当前 epoch
    sample_dir : str
        保存目录
    """
    import matplotlib.pyplot as plt  # 绘图库
    
    # 计算网格大小
    num_samples = samples.shape[0]
    grid_size = int(num_samples ** 0.5)
    
    # 创建图像网格
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    
    # 遍历每个样本
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx < num_samples:
                # 获取图像
                img = samples[idx, 0].cpu().numpy()
                
                # 显示图像
                axes[i, j].imshow(img, cmap='gray', vmin=0, vmax=1)
                axes[i, j].axis('off')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(sample_dir, f'samples_epoch_{epoch:03d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"样本已保存到: {save_path}")


# ============================================
# 保存和加载模型检查点
# ============================================

def save_checkpoint(model, optimizer, scheduler, epoch, loss, save_dir):
    """
    保存训练检查点
    
    参数：
    -----
    model : nn.Module
        模型
    optimizer : Optimizer
        优化器
    scheduler : LRScheduler
        学习率调度器
    epoch : int
        当前 epoch
    loss : float
        当前损失
    save_dir : str
        保存目录
    
    检查点内容：
    -----------
    - 模型参数
    - 优化器状态
    - 调度器状态
    - epoch 编号
    - 损失值
    
    这些信息允许从检查点恢复训练。
    """
    # 构建检查点字典
    checkpoint = {
        'epoch': epoch,  # 当前 epoch
        'model_state_dict': model.state_dict(),  # 模型参数
        'optimizer_state_dict': optimizer.state_dict(),  # 优化器状态
        'scheduler_state_dict': scheduler.state_dict(),  # 调度器状态
        'loss': loss,  # 当前损失
    }
    
    # 保存路径
    save_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch:03d}.pt')
    
    # 保存检查点
    # torch.save 将对象序列化到文件
    torch.save(checkpoint, save_path)
    
    print(f"检查点已保存到: {save_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """
    加载训练检查点
    
    参数：
    -----
    model : nn.Module
        模型
    optimizer : Optimizer
        优化器
    scheduler : LRScheduler
        学习率调度器
    checkpoint_path : str
        检查点文件路径
    device : torch.device
        计算设备
    
    返回：
    -----
    start_epoch : int
        起始 epoch（从检查点恢复）
    """
    # 加载检查点
    # map_location 确保张量加载到正确的设备
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 恢复模型参数
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 恢复优化器状态
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 恢复调度器状态
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # 获取 epoch 和损失
    start_epoch = checkpoint['epoch'] + 1  # 从下一个 epoch 开始
    loss = checkpoint['loss']
    
    print(f"已加载检查点: {checkpoint_path}")
    print(f"  恢复训练从 epoch {start_epoch}, 损失 {loss:.4f}")
    
    return start_epoch


# ============================================
# 主训练函数
# ============================================

def train(resume_from=None):
    """
    主训练函数
    
    参数：
    -----
    resume_from : str or None
        检查点路径，如果提供则从该检查点恢复训练
    
    训练流程：
    ---------
    1. 创建数据加载器
    2. 创建模型
    3. 创建优化器和调度器
    4. 如果需要，加载检查点
    5. 训练循环
    6. 定期保存检查点和生成样本
    """
    import math  # 数学函数（用于学习率调度器）
    
    print("\n" + "=" * 60)
    print("开始训练 Flow Matching 模型")
    print("=" * 60)
    
    # 步骤 1：创建数据加载器
    print("\n加载数据集...")
    dataloader = get_dataloader(BATCH_SIZE, DEVICE)
    
    # 步骤 2：创建模型
    print("\n创建模型...")
    model = FlowMatchingModel().to(DEVICE)
    
    # 统计模型参数
    total_params, trainable_params = count_parameters(model)
    print(f"模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")
    
    # 步骤 3：创建优化器
    print(f"\n创建优化器: {OPTIMIZER}")
    if OPTIMIZER == 'adam':
        # Adam 优化器
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
    elif OPTIMIZER == 'adamw':
        # AdamW 优化器（推荐）
        # AdamW 对权重衰减的处理更好
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
    else:
        raise ValueError(f"未知的优化器: {OPTIMIZER}")
    
    # 步骤 4：创建学习率调度器
    # 计算总训练步数
    num_training_steps = NUM_EPOCHS * len(dataloader)
    print(f"\n总训练步数: {num_training_steps}")
    print(f"预热步数: {WARMUP_STEPS}")
    
    # 创建调度器
    scheduler = get_lr_scheduler(
        optimizer,
        num_training_steps=num_training_steps,
        warmup_steps=WARMUP_STEPS,
        scheduler_type=LR_SCHEDULER
    )
    
    # 步骤 5：加载检查点（如果需要）
    start_epoch = 0
    if resume_from is not None and os.path.exists(resume_from):
        print(f"\n从检查点恢复: {resume_from}")
        start_epoch = load_checkpoint(model, optimizer, scheduler, resume_from, DEVICE)
    
    # 步骤 6：训练循环
    print("\n开始训练循环...")
    print(f"训练 {NUM_EPOCHS} 个 epoch，从 epoch {start_epoch} 开始")
    
    # 记录训练开始时间
    start_time = time.time()
    
    # 训练每个 epoch
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"{'='*60}")
        
        # 训练一个 epoch
        avg_loss = train_one_epoch(
            model, dataloader, optimizer, scheduler, 
            DEVICE, epoch + 1, LOG_INTERVAL
        )
        
        print(f"\nEpoch {epoch + 1} 完成, 平均损失: {avg_loss:.4f}")
        
        # 定期保存检查点
        if (epoch + 1) % SAVE_INTERVAL == 0:
            save_checkpoint(model, optimizer, scheduler, epoch + 1, avg_loss, MODEL_DIR)
        
        # 定期生成样本
        if (epoch + 1) % SAMPLE_INTERVAL == 0:
            print("\n生成样本...")
            generate_samples(
                model, NUM_SAMPLES, NUM_TIMESTEPS, 
                DEVICE, epoch + 1, SAMPLE_DIR
            )
    
    # 训练完成
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"总训练时间: {total_time / 60:.2f} 分钟")
    print("=" * 60)
    
    # 保存最终模型
    final_path = os.path.join(MODEL_DIR, 'model_final.pt')
    torch.save(model.state_dict(), final_path)
    print(f"\n最终模型已保存到: {final_path}")


# ============================================
# 程序入口
# ============================================

if __name__ == '__main__':
    # 解析命令行参数
    import argparse
    
    parser = argparse.ArgumentParser(description='训练 Flow Matching 模型')
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点恢复训练的路径')
    
    args = parser.parse_args()
    
    # 开始训练
    train(resume_from=args.resume)
