"""
Flow Matching 可视化脚本
=======================

本文件提供了多种可视化功能：
1. 向量场可视化：展示学习到的向量场
2. 生成轨迹可视化：展示从噪声到数据的路径
3. 训练损失曲线：展示训练过程中的损失变化
4. 样本对比：对比不同时间步数/方法的生成结果

这些可视化有助于理解 Flow Matching 的工作原理和调试模型。

作者：教学项目
日期：2026-05
"""

# ============================================
# 导入必要的库
# ============================================

import os  # 操作系统接口
import torch  # PyTorch 深度学习框架
import numpy as np  # 数值计算库
import matplotlib.pyplot as plt  # 绘图库
from matplotlib.patches import FancyArrowPatch  # 绘制箭头
from mpl_toolkits.mplot3d import proj3d  # 3D 投影

# 导入项目模块
from config import (  # 配置参数
    DEVICE,              # 计算设备
    LOG_DIR,             # 日志保存目录
    SAMPLE_DIR,          # 样本保存目录
    IMAGE_SIZE,          # 图像尺寸
    DATA_DIM,            # 数据维度
)
from net import FlowMatchingModel  # Flow Matching 模型


# ============================================
# 向量场可视化函数
# ============================================

def visualize_vector_field_2d(model, device, save_path=None):
    """
    可视化 2D 向量场
    
    由于 MNIST 数据是 784 维的，无法直接可视化。
    这里我们使用 PCA 将数据降维到 2D，然后可视化向量场。
    
    参数：
    -----
    model : FlowMatchingModel
        训练好的模型
    device : torch.device
        计算设备
    save_path : str or None
        保存路径，None 则不保存
    
    说明：
    -----
    向量场 v(x, t) 描述了每个点在每个时刻的移动方向。
    通过可视化，我们可以看到：
    - 在噪声区域（t 接近 0）：向量场较大，指向数据区域
    - 在数据区域（t 接近 1）：向量场较小，精细调整
    """
    print("\n可视化 2D 向量场...")
    
    # 定义可视化范围
    # 假设数据在 [-3, 3] x [-3, 3] 范围内
    x_range = np.linspace(-3, 3, 20)
    y_range = np.linspace(-3, 3, 20)
    
    # 创建网格
    X, Y = np.meshgrid(x_range, y_range)
    
    # 定义要可视化的时间点
    time_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    # 创建图像
    fig, axes = plt.subplots(1, len(time_points), figsize=(20, 4))
    
    model.eval()
    with torch.no_grad():
        for idx, t in enumerate(time_points):
            # 计算每个网格点的向量场
            U = np.zeros_like(X)  # x 方向分量
            V = np.zeros_like(Y)  # y 方向分量
            
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    # 创建一个 784 维的点
                    # 这里简单地将 2D 坐标复制到所有维度
                    # 这只是一个示意，真实的可视化需要 PCA
                    point_2d = np.array([X[i, j], Y[i, j]])
                    
                    # 扩展到 784 维
                    # 使用简单的复制策略
                    point_784 = np.zeros(DATA_DIM)
                    point_784[0] = point_2d[0]
                    point_784[1] = point_2d[1]
                    
                    # 转换为张量
                    x_tensor = torch.tensor(point_784, dtype=torch.float32).unsqueeze(0).to(device)
                    t_tensor = torch.tensor([t], dtype=torch.float32).to(device)
                    
                    # 计算向量场
                    v = model(x_tensor, t_tensor)
                    
                    # 提取前两个分量作为 2D 向量
                    U[i, j] = v[0, 0].item()
                    V[i, j] = v[0, 1].item()
            
            # 绘制向量场
            # 使用 quiver 绘制箭头
            axes[idx].quiver(X, Y, U, V, np.sqrt(U**2 + V**2), 
                           cmap='viridis', scale=50)
            axes[idx].set_title(f't = {t:.2f}')
            axes[idx].set_xlabel('x')
            axes[idx].set_ylabel('y')
            axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"向量场可视化已保存到: {save_path}")
    
    plt.show()
    plt.close()


# ============================================
# 生成轨迹可视化函数
# ============================================

def visualize_generation_trajectory(model, num_samples, num_steps, device, save_path=None):
    """
    可视化生成轨迹
    
    展示从噪声到生成样本的完整路径。
    
    参数：
    -----
    model : FlowMatchingModel
        训练好的模型
    num_samples : int
        可视化的样本数量
    num_steps : int
        ODE 求解步数
    device : torch.device
        计算设备
    save_path : str or None
        保存路径
    
    可视化内容：
    -----------
    对于每个样本，展示：
    1. 起始点（噪声）
    2. 中间轨迹点
    3. 终点（生成样本）
    
    这有助于理解 Flow Matching 如何将噪声变换为数据。
    """
    print("\n可视化生成轨迹...")
    
    # 采样初始噪声
    x = torch.randn(num_samples, DATA_DIM, device=device)
    
    # 存储轨迹
    trajectories = [x.clone().cpu().numpy()]
    
    # 求解 ODE
    dt = 1.0 / num_steps
    times = [0.0]
    
    model.eval()
    with torch.no_grad():
        for step in range(num_steps):
            t = step / num_steps
            t_tensor = torch.full((num_samples,), t, device=device)
            
            v = model(x, t_tensor)
            x = x + dt * v
            
            # 记录轨迹（每隔一定步数）
            if step % (num_steps // 10) == 0:
                trajectories.append(x.clone().cpu().numpy())
                times.append((step + 1) / num_steps)
    
    # 确保记录最后一点
    if times[-1] < 1.0:
        trajectories.append(x.clone().cpu().numpy())
        times.append(1.0)
    
    # 可视化轨迹
    # 由于数据是 784 维，我们只可视化前两个维度
    fig, axes = plt.subplots(1, num_samples, figsize=(4 * num_samples, 4))
    
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        # 提取第 i 个样本的轨迹
        traj = np.array([t[i] for t in trajectories])
        
        # 绘制轨迹
        axes[i].plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.5, linewidth=2)
        
        # 标记起点和终点
        axes[i].scatter(traj[0, 0], traj[0, 1], c='red', s=100, 
                       marker='o', label='Start (noise)', zorder=5)
        axes[i].scatter(traj[-1, 0], traj[-1, 1], c='green', s=100, 
                       marker='*', label='End (generated)', zorder=5)
        
        # 标记中间点
        for j in range(1, len(traj) - 1):
            axes[i].scatter(traj[j, 0], traj[j, 1], c='blue', s=30, 
                           marker='.', alpha=0.5)
        
        axes[i].set_xlabel('Dimension 1')
        axes[i].set_ylabel('Dimension 2')
        axes[i].set_title(f'Sample {i + 1}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"生成轨迹可视化已保存到: {save_path}")
    
    plt.show()
    plt.close()


# ============================================
# 训练损失曲线可视化函数
# ============================================

def visualize_training_loss(loss_history, save_path=None):
    """
    可视化训练损失曲线
    
    参数：
    -----
    loss_history : list of float
        每个 epoch 的平均损失
    save_path : str or None
        保存路径
    
    可视化内容：
    -----------
    - 损失随 epoch 的变化曲线
    - 平滑后的损失曲线
    - 最小损失标记
    """
    print("\n可视化训练损失曲线...")
    
    if not loss_history:
        print("警告：损失历史为空，无法可视化")
        return
    
    # 创建图像
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制原始损失曲线
    epochs = range(1, len(loss_history) + 1)
    ax.plot(epochs, loss_history, 'b-', alpha=0.5, linewidth=1, label='原始损失')
    
    # 计算平滑后的损失（移动平均）
    window_size = min(5, len(loss_history))
    if window_size > 1:
        smoothed = np.convolve(loss_history, np.ones(window_size)/window_size, mode='valid')
        ax.plot(range(window_size, len(loss_history) + 1), smoothed, 
               'r-', linewidth=2, label=f'平滑损失 (窗口={window_size})')
    
    # 标记最小损失
    min_idx = np.argmin(loss_history)
    min_loss = loss_history[min_idx]
    ax.scatter(min_idx + 1, min_loss, c='green', s=100, marker='*', 
              label=f'最小损失: {min_loss:.4f} (epoch {min_idx + 1})', zorder=5)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('训练损失曲线', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"损失曲线已保存到: {save_path}")
    
    plt.show()
    plt.close()


# ============================================
# 不同步数对比可视化函数
# ============================================

def compare_num_steps(model, num_samples, step_options, device, save_path=None):
    """
    对比不同 ODE 步数的生成结果
    
    参数：
    -----
    model : FlowMatchingModel
        训练好的模型
    num_samples : int
        每种配置生成的样本数量
    step_options : list of int
        要对比的步数列表，如 [10, 50, 100, 500]
    device : torch.device
        计算设备
    save_path : str or None
        保存路径
    
    说明：
    -----
    更多步数 = 更精确的 ODE 求解 = 更好的生成质量
    但也意味着更慢的生成速度
    这个可视化帮助选择合适的步数
    """
    print(f"\n对比不同 ODE 步数: {step_options}")
    
    # 为每种步数生成样本
    all_samples = []
    
    model.eval()
    for num_steps in step_options:
        print(f"  生成步数={num_steps} 的样本...")
        with torch.no_grad():
            samples = model.sample(
                num_samples=num_samples,
                num_steps=num_steps,
                device=device,
                method='euler'
            )
        # 重塑并裁剪
        samples = samples.view(num_samples, 1, IMAGE_SIZE, IMAGE_SIZE)
        samples = torch.clamp(samples, 0, 1)
        all_samples.append(samples.cpu().numpy())
    
    # 创建对比图像
    fig, axes = plt.subplots(num_samples, len(step_options), 
                            figsize=(3 * len(step_options), 3 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        for j, num_steps in enumerate(step_options):
            img = all_samples[j][i, 0]
            axes[i, j].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[i, j].axis('off')
            
            if i == 0:
                axes[i, j].set_title(f'steps={num_steps}', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"步数对比已保存到: {save_path}")
    
    plt.show()
    plt.close()


# ============================================
# 不同积分方法对比可视化函数
# ============================================

def compare_methods(model, num_samples, num_steps, device, save_path=None):
    """
    对比不同积分方法的生成结果
    
    参数：
    -----
    model : FlowMatchingModel
        训练好的模型
    num_samples : int
        每种方法生成的样本数量
    num_steps : int
        ODE 求解步数
    device : torch.device
        计算设备
    save_path : str or None
        保存路径
    
    积分方法：
    ---------
    - Euler（欧拉方法）：简单快速，精度较低
    - RK4（四阶龙格-库塔）：更精确，但计算量大
    """
    print(f"\n对比不同积分方法...")
    
    methods = ['euler', 'rk4']
    all_samples = []
    
    model.eval()
    for method in methods:
        print(f"  使用 {method} 方法生成样本...")
        with torch.no_grad():
            samples = model.sample(
                num_samples=num_samples,
                num_steps=num_steps,
                device=device,
                method=method
            )
        samples = samples.view(num_samples, 1, IMAGE_SIZE, IMAGE_SIZE)
        samples = torch.clamp(samples, 0, 1)
        all_samples.append(samples.cpu().numpy())
    
    # 创建对比图像
    fig, axes = plt.subplots(num_samples, len(methods), 
                            figsize=(3 * len(methods), 3 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        for j, method in enumerate(methods):
            img = all_samples[j][i, 0]
            axes[i, j].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[i, j].axis('off')
            
            if i == 0:
                axes[i, j].set_title(f'{method.upper()}', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"方法对比已保存到: {save_path}")
    
    plt.show()
    plt.close()


# ============================================
# 时间嵌入可视化函数
# ============================================

def visualize_time_embedding(embed_dim=64, save_path=None):
    """
    可视化时间嵌入
    
    展示不同时间点的嵌入向量。
    
    参数：
    -----
    embed_dim : int
        时间嵌入维度
    save_path : str or None
        保存路径
    
    说明：
    -----
    时间嵌入将标量时间 t 编码为高维向量。
    通过可视化，我们可以看到：
    - 不同时间的嵌入是不同的
    - 相邻时间的嵌入是相似的
    - 嵌入向量具有周期性结构
    """
    print("\n可视化时间嵌入...")
    
    from net import SinusoidalTimeEmbedding
    
    # 创建时间嵌入层
    time_embed = SinusoidalTimeEmbedding(embed_dim)
    
    # 创建时间点
    times = torch.linspace(0, 1, 100)
    
    # 计算嵌入
    with torch.no_grad():
        embeddings = time_embed(times).numpy()
    
    # 创建可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：嵌入矩阵热力图
    im = axes[0].imshow(embeddings.T, aspect='auto', cmap='RdBu', 
                        extent=[0, 1, embed_dim, 0])
    axes[0].set_xlabel('Time t', fontsize=12)
    axes[0].set_ylabel('Embedding Dimension', fontsize=12)
    axes[0].set_title('时间嵌入矩阵', fontsize=14)
    plt.colorbar(im, ax=axes[0], label='Value')
    
    # 右图：部分维度的嵌入曲线
    dims_to_plot = [0, 10, 20, 30, 40, 50]
    for dim in dims_to_plot:
        if dim < embed_dim:
            axes[1].plot(times.numpy(), embeddings[:, dim], 
                        label=f'dim {dim}', alpha=0.7)
    
    axes[1].set_xlabel('Time t', fontsize=12)
    axes[1].set_ylabel('Embedding Value', fontsize=12)
    axes[1].set_title('部分维度的嵌入曲线', fontsize=14)
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"时间嵌入可视化已保存到: {save_path}")
    
    plt.show()
    plt.close()


# ============================================
# 主可视化函数
# ============================================

def visualize_all(model_path, device):
    """
    执行所有可视化
    
    参数：
    -----
    model_path : str
        模型文件路径
    device : torch.device
        计算设备
    """
    print("\n" + "=" * 60)
    print("Flow Matching 可视化")
    print("=" * 60)
    
    # 加载模型
    print(f"\n加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model = FlowMatchingModel()
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # 创建保存目录
    vis_dir = os.path.join(LOG_DIR, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 1. 向量场可视化
    visualize_vector_field_2d(
        model, device, 
        save_path=os.path.join(vis_dir, 'vector_field.png')
    )
    
    # 2. 生成轨迹可视化
    visualize_generation_trajectory(
        model, num_samples=4, num_steps=100, device=device,
        save_path=os.path.join(vis_dir, 'generation_trajectory.png')
    )
    
    # 3. 不同步数对比
    compare_num_steps(
        model, num_samples=4, step_options=[10, 50, 100, 200],
        device=device,
        save_path=os.path.join(vis_dir, 'compare_steps.png')
    )
    
    # 4. 不同方法对比
    compare_methods(
        model, num_samples=4, num_steps=100, device=device,
        save_path=os.path.join(vis_dir, 'compare_methods.png')
    )
    
    # 5. 时间嵌入可视化
    visualize_time_embedding(
        embed_dim=64,
        save_path=os.path.join(vis_dir, 'time_embedding.png')
    )
    
    print("\n" + "=" * 60)
    print("可视化完成！")
    print("=" * 60)


# ============================================
# 程序入口
# ============================================

if __name__ == '__main__':
    # 解析命令行参数
    import argparse
    
    parser = argparse.ArgumentParser(description='Flow Matching 可视化')
    parser.add_argument('--model', type=str, required=True,
                        help='模型文件路径')
    parser.add_argument('--type', type=str, 
                        choices=['all', 'vector_field', 'trajectory', 
                                'compare_steps', 'compare_methods', 'time_embedding'],
                        default='all', help='可视化类型')
    
    args = parser.parse_args()
    
    if args.type == 'all':
        visualize_all(args.model, DEVICE)
    elif args.type == 'time_embedding':
        visualize_time_embedding(save_path=os.path.join(LOG_DIR, 'time_embedding.png'))
    else:
        # 加载模型
        checkpoint = torch.load(args.model, map_location=DEVICE)
        model = FlowMatchingModel()
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(DEVICE)
        model.eval()
        
        vis_dir = os.path.join(LOG_DIR, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        if args.type == 'vector_field':
            visualize_vector_field_2d(model, DEVICE, 
                                     os.path.join(vis_dir, 'vector_field.png'))
        elif args.type == 'trajectory':
            visualize_generation_trajectory(model, 4, 100, DEVICE,
                                          os.path.join(vis_dir, 'generation_trajectory.png'))
        elif args.type == 'compare_steps':
            compare_num_steps(model, 4, [10, 50, 100, 200], DEVICE,
                            os.path.join(vis_dir, 'compare_steps.png'))
        elif args.type == 'compare_methods':
            compare_methods(model, 4, 100, DEVICE,
                          os.path.join(vis_dir, 'compare_methods.png'))
