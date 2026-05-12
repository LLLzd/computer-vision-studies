"""
3D Gaussian Splatting - 训练与优化模块（简化版）

原理：
    3DGS的训练过程是一个优化问题：
    1. 初始化大量3D高斯（从点云或随机初始化）
    2. 通过梯度下降优化高斯参数，使渲染结果与真实图像匹配
    3. 使用可微渲染技术计算损失的梯度

损失函数：
    L = ||rendered_image - target_image||^2

优化参数：
    - 每个高斯的位置 (x, y, z)
    - 每个高斯的协方差矩阵（通过缩放和旋转参数化）
    - 每个高斯的颜色 (r, g, b)
    - 每个高斯的不透明度 (alpha)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from utils.core import Gaussian3D, Camera, GaussianRenderer


class SimpleGaussianOptimizer:
    """
    简化版高斯参数优化器
    
    使用梯度下降优化高斯参数（位置和颜色）
    """
    
    def __init__(self, gaussians, learning_rate=0.1):
        """
        初始化优化器
        
        参数：
            gaussians: list - Gaussian3D对象列表
            learning_rate: float - 学习率
        """
        self.gaussians = gaussians
        self.lr = learning_rate
    
    def compute_loss(self, rendered_image, target_image):
        """
        计算损失函数
        
        参数：
            rendered_image: np.array (H, W, 3) - 渲染图像
            target_image: np.array (H, W, 3) - 目标图像
        
        返回：
            loss: float - 损失值
        """
        return np.mean((rendered_image - target_image)**2)
    
    def optimize_position(self, renderer, camera, target_image, num_steps=5):
        """
        优化高斯位置（简化版：直接根据图像梯度调整位置）
        
        参数：
            renderer: GaussianRenderer - 渲染器
            camera: Camera - 相机
            target_image: np.array (H, W, 3) - 目标图像
            num_steps: int - 优化步数
        """
        losses = []
        
        for step in range(num_steps):
            # 渲染当前图像
            image = renderer.render(self.gaussians, camera)
            loss = self.compute_loss(image, target_image)
            losses.append(loss)
            
            # 计算图像差异
            diff = target_image - image  # (H, W, 3)
            
            # 简单的位置更新策略：让高斯向误差大的区域移动
            for g in self.gaussians:
                # 获取高斯的投影位置
                proj_matrix = camera.get_full_projection_matrix()
                proj = g.project_to_2d(proj_matrix)
                cx, cy = proj['center']
                
                # 检查投影位置是否在图像范围内
                if 0 <= cx < renderer.width and 0 <= cy < renderer.height:
                    # 获取该像素的误差
                    pixel_error = diff[int(cy), int(cx)]
                    
                    # 根据误差调整位置
                    # 将图像空间的误差转换到3D空间
                    dx = pixel_error[0] * 0.01
                    dy = pixel_error[1] * 0.01
                    
                    # 更新位置（向误差方向移动）
                    g.position[0] += dx * self.lr
                    g.position[1] += dy * self.lr
            
            if (step + 1) % 2 == 0:
                print(f"  Step {step+1}/{num_steps}, Loss: {loss:.4f}")
        
        # 最后一次渲染
        final_image = renderer.render(self.gaussians, camera)
        final_loss = self.compute_loss(final_image, target_image)
        losses.append(final_loss)
        
        return losses


def create_synthetic_target(image_size):
    """
    创建一个合成的目标图像（用于测试训练）
    
    参数：
        image_size: tuple - 图像尺寸 (width, height)
    
    返回：
        target_image: np.array (height, width, 3) - 目标图像
    """
    width, height = image_size
    target_image = np.zeros((height, width, 3))
    
    # 绘制一个红色圆形
    cx, cy = width // 2, height // 2
    radius = 40
    yy, xx = np.mgrid[:height, :width]
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    mask = dist < radius
    target_image[mask] = [1.0, 0.2, 0.2]
    
    # 绘制一个蓝色正方形
    square_size = 30
    sx, sy = width // 4, height // 4
    target_image[sy:sy+square_size, sx:sx+square_size] = [0.2, 0.2, 1.0]
    
    return target_image


def training_demo():
    """演示训练过程（简化版）"""
    print("=" * 60)
    print("3D Gaussian Splatting 训练演示（简化版）")
    print("=" * 60)
    
    # 参数设置
    image_size = (320, 240)  # (width, height)
    num_gaussians = 20  # 减少高斯数量加快演示
    num_steps = 10
    
    print(f"\n参数设置:")
    print(f"  图像尺寸: {image_size[0]} x {image_size[1]}")
    print(f"  高斯数量: {num_gaussians}")
    print(f"  优化步数: {num_steps}")
    
    # 创建目标图像
    print("\n创建目标图像...")
    target_image = create_synthetic_target(image_size)
    
    # 创建相机
    camera = Camera(
        position=[0, 0, 0],
        look_at=[0, 0, 3],
        up_vector=[0, 1, 0],
        fov=60,
        image_size=image_size
    )
    
    # 创建初始高斯（集中在中心区域）
    print("初始化高斯...")
    gaussians = []
    np.random.seed(42)
    
    for _ in range(num_gaussians):
        x = np.random.uniform(-0.5, 0.5)
        y = np.random.uniform(-0.5, 0.5)
        z = np.random.uniform(2.5, 3.5)
        
        scale = np.random.uniform(0.15, 0.3)
        
        # 使用SVD分解创建有效的旋转矩阵
        U, _, Vt = np.linalg.svd(np.random.randn(3, 3))
        rotation_matrix = U @ Vt
        if np.linalg.det(rotation_matrix) < 0:
            rotation_matrix[:, -1] *= -1
        
        sigma = np.eye(3) * scale**2
        covariance = rotation_matrix @ sigma @ rotation_matrix.T
        
        # 初始颜色为随机
        color = np.random.rand(3)
        opacity = np.random.uniform(0.6, 1.0)
        
        gaussians.append(Gaussian3D(
            position=[x, y, z],
            covariance=covariance,
            color=color,
            opacity=opacity
        ))
    
    # 创建渲染器
    renderer = GaussianRenderer(image_size)
    
    # 初始渲染
    initial_image = renderer.render(gaussians, camera)
    
    # 创建优化器并优化
    optimizer = SimpleGaussianOptimizer(gaussians, learning_rate=0.5)
    
    print("\n开始优化...")
    losses = optimizer.optimize_position(renderer, camera, target_image, num_steps)
    
    # 最终渲染
    final_image = renderer.render(gaussians, camera)
    
    # 绘制结果
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 目标图像
    axes[0].imshow(target_image)
    axes[0].set_title('Target Image')
    axes[0].axis('off')
    
    # 初始渲染
    axes[1].imshow(initial_image)
    axes[1].set_title('Initial Render')
    axes[1].axis('off')
    
    # 最终渲染
    axes[2].imshow(final_image)
    axes[2].set_title('Final Render')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 绘制损失曲线
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(losses, 'b-', linewidth=2, marker='o')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Optimization Loss Curve')
    ax.grid(True)
    plt.show()
    
    print("\n优化完成！")


if __name__ == "__main__":
    training_demo()
