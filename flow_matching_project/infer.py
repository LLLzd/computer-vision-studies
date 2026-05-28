"""
Flow Matching 推理脚本
=====================

本文件实现了 Flow Matching 模型的推理功能：
1. 加载训练好的模型
2. 从噪声生成新样本
3. 可视化生成过程
4. 保存生成结果

推理过程：
---------
1. 从标准正态分布采样噪声 x_0 ~ N(0, I)
2. 求解 ODE: dx/dt = v(x, t), x(0) = x_0
3. 得到生成样本 x(1)

作者：教学项目
日期：2026-05
"""

# ============================================
# 导入必要的库
# ============================================

import os  # 操作系统接口
import torch  # PyTorch 深度学习框架
import matplotlib.pyplot as plt  # 绘图库
import numpy as np  # 数值计算库
from tqdm import tqdm  # 进度条显示

# 导入项目模块
from config import (  # 配置参数
    DEVICE,              # 计算设备
    MODEL_DIR,           # 模型保存目录
    SAMPLE_DIR,          # 样本保存目录
    NUM_SAMPLES,         # 生成样本数量
    NUM_TIMESTEPS,       # 时间步数
    INTEGRATION_METHOD,  # 积分方法
    IMAGE_SIZE,          # 图像尺寸
    DATA_DIM,            # 数据维度
)
from net import FlowMatchingModel  # Flow Matching 模型


# ============================================
# 加载模型函数
# ============================================

def load_model(model_path, device):
    """
    加载训练好的模型
    
    参数：
    -----
    model_path : str
        模型文件路径
        可以是完整检查点（.pt 包含 optimizer 等）
        或仅模型权重（.pt 仅包含 state_dict）
    device : torch.device
        计算设备
    
    返回：
    -----
    model : FlowMatchingModel
        加载了权重的模型
    
    说明：
    -----
    支持两种模型文件格式：
    1. 检查点格式：包含 model_state_dict, optimizer_state_dict 等
    2. 权重格式：直接是 state_dict
    """
    # 创建模型实例
    model = FlowMatchingModel()
    
    # 加载文件
    checkpoint = torch.load(model_path, map_location=device)
    
    # 判断文件格式
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # 检查点格式
        print("加载检查点格式模型...")
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 打印检查点信息
        if 'epoch' in checkpoint:
            print(f"  检查点 epoch: {checkpoint['epoch']}")
        if 'loss' in checkpoint:
            print(f"  检查点损失: {checkpoint['loss']:.4f}")
    else:
        # 权重格式
        print("加载权重格式模型...")
        model.load_state_dict(checkpoint)
    
    # 将模型移到设备
    model = model.to(device)
    
    # 设置为评估模式
    model.eval()
    
    print(f"模型已加载到设备: {device}")
    
    return model


# ============================================
# 生成样本函数
# ============================================

def generate_samples(model, num_samples, num_steps, device, method='euler'):
    """
    从模型生成样本
    
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
    method : str
        积分方法，'euler' 或 'rk4'
    
    返回：
    -----
    samples : torch.Tensor
        生成的样本，形状 (num_samples, 1, 28, 28)
    
    生成过程：
    ---------
    1. 采样初始噪声 x_0 ~ N(0, I)
    2. 求解 ODE 得到 x(1)
    3. 重塑为图像形状
    4. 裁剪到 [0, 1] 范围
    """
    print(f"\n生成 {num_samples} 个样本...")
    print(f"  ODE 步数: {num_steps}")
    print(f"  积分方法: {method}")
    
    # 使用 torch.no_grad() 禁用梯度计算
    # 推理时不需要梯度，可以节省内存和加速
    with torch.no_grad():
        # 从模型采样
        samples = model.sample(
            num_samples=num_samples,
            num_steps=num_steps,
            device=device,
            method=method
        )
    
    # 重塑为图像形状
    # samples 形状：(num_samples, 784) -> (num_samples, 1, 28, 28)
    samples = samples.view(num_samples, 1, IMAGE_SIZE, IMAGE_SIZE)
    
    # 裁剪到 [0, 1] 范围
    # 生成的样本可能超出图像值范围
    samples = torch.clamp(samples, 0, 1)
    
    print(f"  样本形状: {samples.shape}")
    print(f"  值范围: [{samples.min().item():.3f}, {samples.max().item():.3f}]")
    
    return samples


# ============================================
# 可视化生成过程函数
# ============================================

def visualize_generation_process(model, num_samples, num_steps, device, save_path):
    """
    可视化生成过程
    
    展示从噪声到生成样本的中间过程。
    
    参数：
    -----
    model : FlowMatchingModel
        训练好的模型
    num_samples : int
        可视化的样本数量（通常较小，如 4-8）
    num_steps : int
        ODE 求解步数
    device : torch.device
        计算设备
    save_path : str
        保存路径
    
    可视化内容：
    -----------
    展示多个时间点的中间状态：
    - t=0: 噪声
    - t=0.25: 早期形态
    - t=0.5: 中间形态
    - t=0.75: 接近完成
    - t=1: 最终生成结果
    
    这有助于理解 Flow Matching 的工作原理。
    """
    print("\n可视化生成过程...")
    
    # 定义要展示的时间点
    # 从 0 到 1，均匀采样 5 个时间点
    time_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    # 采样初始噪声
    x = torch.randn(num_samples, DATA_DIM, device=device)
    
    # 存储各时间点的状态
    states = [x.clone()]  # t=0 的状态（噪声）
    
    # 计算时间步长
    dt = 1.0 / num_steps
    
    # 求解 ODE 并记录中间状态
    with torch.no_grad():
        for step in range(num_steps):
            # 当前时间
            t = step / num_steps
            t_tensor = torch.full((num_samples,), t, device=device)
            
            # 计算向量场
            v = model(x, t_tensor)
            
            # 欧拉更新
            x = x + dt * v
            
            # 检查是否到达记录时间点
            current_time = (step + 1) / num_steps
            for tp in time_points[1:]:  # 跳过 t=0
                if abs(current_time - tp) < dt / 2:
                    states.append(x.clone())
                    break
    
    # 确保记录了所有时间点
    while len(states) < len(time_points):
        states.append(x.clone())
    
    # 创建可视化图像
    fig, axes = plt.subplots(num_samples, len(time_points), figsize=(15, 3 * num_samples))
    
    # 如果只有一行，确保 axes 是二维数组
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    # 绘制每个样本在每个时间点的状态
    for i in range(num_samples):
        for j, (t, state) in enumerate(zip(time_points, states)):
            # 重塑为图像形状
            img = state[i].view(IMAGE_SIZE, IMAGE_SIZE).cpu().numpy()
            
            # 裁剪到 [0, 1]
            img = np.clip(img, 0, 1)
            
            # 显示图像
            axes[i, j].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[i, j].axis('off')
            
            # 第一行添加时间标签
            if i == 0:
                axes[i, j].set_title(f't={t:.2f}', fontsize=12)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"生成过程可视化已保存到: {save_path}")


# ============================================
# 保存样本图像函数
# ============================================

def save_samples_as_image(samples, save_path, title="Generated Samples"):
    """
    将样本保存为图像网格
    
    参数：
    -----
    samples : torch.Tensor
        样本张量，形状 (num_samples, 1, 28, 28)
    save_path : str
        保存路径
    title : str
        图像标题
    """
    # 获取样本数量
    num_samples = samples.shape[0]
    
    # 计算网格大小
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    # 创建图像网格
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    
    # 如果只有一张图，确保 axes 是数组
    if grid_size == 1:
        axes = np.array([[axes]])
    
    # 遍历网格位置
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            
            if idx < num_samples:
                # 获取图像
                img = samples[idx, 0].cpu().numpy()
                
                # 显示图像
                axes[i, j].imshow(img, cmap='gray', vmin=0, vmax=1)
            
            # 隐藏坐标轴
            axes[i, j].axis('off')
    
    # 设置标题
    fig.suptitle(title, fontsize=16)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"样本图像已保存到: {save_path}")


# ============================================
# 保存样本为单独文件函数
# ============================================

def save_individual_samples(samples, save_dir, prefix='sample'):
    """
    将每个样本保存为单独的图像文件
    
    参数：
    -----
    samples : torch.Tensor
        样本张量，形状 (num_samples, 1, 28, 28)
    save_dir : str
        保存目录
    prefix : str
        文件名前缀
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 遍历每个样本
    for i, sample in enumerate(samples):
        # 获取图像
        img = sample[0].cpu().numpy()
        
        # 创建图像
        plt.figure(figsize=(3, 3))
        plt.imshow(img, cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        
        # 保存
        save_path = os.path.join(save_dir, f'{prefix}_{i:04d}.png')
        plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close()
    
    print(f"已保存 {len(samples)} 个单独样本到: {save_dir}")


# ============================================
# 计算样本统计信息函数
# ============================================

def compute_sample_statistics(samples):
    """
    计算生成样本的统计信息
    
    参数：
    -----
    samples : torch.Tensor
        样本张量，形状 (num_samples, 1, 28, 28)
    
    返回：
    -----
    stats : dict
        统计信息字典
    
    统计内容：
    ---------
    - 均值：所有像素的平均值
    - 标准差：像素值的标准差
    - 最小值：最小像素值
    - 最大值：最大像素值
    - 中位数：像素值的中位数
    """
    # 转换为 numpy 数组
    samples_np = samples.cpu().numpy()
    
    # 计算统计信息
    stats = {
        'mean': float(np.mean(samples_np)),  # 均值
        'std': float(np.std(samples_np)),    # 标准差
        'min': float(np.min(samples_np)),    # 最小值
        'max': float(np.max(samples_np)),    # 最大值
        'median': float(np.median(samples_np)),  # 中位数
    }
    
    return stats


# ============================================
# 主推理函数
# ============================================

def infer(model_path, num_samples=None, num_steps=None, method=None, 
          visualize_process=False, save_individual=False):
    """
    主推理函数
    
    参数：
    -----
    model_path : str
        模型文件路径
    num_samples : int or None
        生成样本数量，None 则使用配置默认值
    num_steps : int or None
        ODE 求解步数，None 则使用配置默认值
    method : str or None
        积分方法，None 则使用配置默认值
    visualize_process : bool
        是否可视化生成过程
    save_individual : bool
        是否保存单独的样本文件
    
    推理流程：
    ---------
    1. 加载模型
    2. 生成样本
    3. 计算统计信息
    4. 保存结果
    5. 可视化（可选）
    """
    # 使用默认值
    if num_samples is None:
        num_samples = NUM_SAMPLES
    if num_steps is None:
        num_steps = NUM_TIMESTEPS
    if method is None:
        method = INTEGRATION_METHOD
    
    print("\n" + "=" * 60)
    print("Flow Matching 推理")
    print("=" * 60)
    
    # 步骤 1：加载模型
    print(f"\n加载模型: {model_path}")
    model = load_model(model_path, DEVICE)
    
    # 步骤 2：生成样本
    samples = generate_samples(model, num_samples, num_steps, DEVICE, method)
    
    # 步骤 3：计算统计信息
    print("\n计算样本统计信息...")
    stats = compute_sample_statistics(samples)
    print(f"  均值: {stats['mean']:.4f}")
    print(f"  标准差: {stats['std']:.4f}")
    print(f"  最小值: {stats['min']:.4f}")
    print(f"  最大值: {stats['max']:.4f}")
    print(f"  中位数: {stats['median']:.4f}")
    
    # 步骤 4：保存结果
    print("\n保存结果...")
    
    # 保存为网格图像
    grid_path = os.path.join(SAMPLE_DIR, 'generated_samples.png')
    save_samples_as_image(samples, grid_path, 
                         title=f"Generated Samples (steps={num_steps}, method={method})")
    
    # 保存单独文件（可选）
    if save_individual:
        individual_dir = os.path.join(SAMPLE_DIR, 'individual')
        save_individual_samples(samples, individual_dir)
    
    # 步骤 5：可视化生成过程（可选）
    if visualize_process:
        process_path = os.path.join(SAMPLE_DIR, 'generation_process.png')
        visualize_generation_process(model, num_samples=4, num_steps=num_steps,
                                    device=DEVICE, save_path=process_path)
    
    print("\n" + "=" * 60)
    print("推理完成！")
    print("=" * 60)
    
    return samples


# ============================================
# 程序入口
# ============================================

if __name__ == '__main__':
    # 解析命令行参数
    import argparse
    
    parser = argparse.ArgumentParser(description='Flow Matching 推理')
    parser.add_argument('--model', type=str, required=True,
                        help='模型文件路径')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='生成样本数量')
    parser.add_argument('--num-steps', type=int, default=None,
                        help='ODE 求解步数')
    parser.add_argument('--method', type=str, choices=['euler', 'rk4'], 
                        default=None, help='积分方法')
    parser.add_argument('--visualize-process', action='store_true',
                        help='可视化生成过程')
    parser.add_argument('--save-individual', action='store_true',
                        help='保存单独的样本文件')
    
    args = parser.parse_args()
    
    # 执行推理
    infer(
        model_path=args.model,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        method=args.method,
        visualize_process=args.visualize_process,
        save_individual=args.save_individual
    )
