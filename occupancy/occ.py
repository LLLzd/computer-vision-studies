"""
Occupancy Grid Estimation - 神经网络方法

原理：
    使用神经网络从原始传感器数据（如激光雷达扫描）中学习 occupancy grid。
    与贝叶斯方法不同，神经网络方法可以从数据中学习复杂的映射关系。

本示例实现：
    1. 简单的MLP网络：从极坐标激光扫描数据预测每个网格的占用概率
    2. 模拟数据生成：创建带有障碍物的简单环境
    3. 训练和推理演示

网络结构：
    Input: 激光扫描数据 (N rays, 每个包含距离和角度信息)
    Hidden: 全连接层 + ReLU
    Output: 网格占用概率图

学习目标：
    最小化交叉熵损失：L = -Σ [y*log(p) + (1-y)*log(1-p)]
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class SimpleOccupancyNet:
    """
    简单的神经网络用于 Occupancy Grid Estimation
    
    该网络接受激光扫描数据，输出 occupancy grid
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, grid_size):
        """
        初始化网络
        
        参数：
            input_dim: 输入维度（激光射线数 * 特征数）
            hidden_dim: 隐藏层维度
            output_dim: 输出维度（网格数）
            grid_size: 网格地图大小
        """
        self.grid_size = grid_size
        
        # 初始化网络权重（简化版：无隐藏层，直接线性映射）
        # 实际应用中应该使用多层网络
        np.random.seed(42)
        
        # Xavier初始化
        scale = np.sqrt(2.0 / (input_dim + output_dim))
        self.weights = np.random.randn(input_dim, output_dim) * scale
        self.bias = np.zeros(output_dim)
        
        # 学习率
        self.lr = 0.01
        
    def relu(self, x):
        """ReLU激活函数"""
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        """Sigmoid激活函数"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def softmax(self, x):
        """Softmax激活函数"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / (exp_x.sum() + 1e-10)
    
    def forward(self, x):
        """
        前向传播
        
        参数：
            x: 输入数据 (batch_size, input_dim)
        
        返回：
            输出占用概率 (batch_size, grid_size * grid_size)
        """
        # 线性层
        self.linear_output = np.dot(x, self.weights) + self.bias
        
        # 添加一些非线性（模拟隐藏层效果）
        hidden = self.relu(self.linear_output)
        
        # 输出层
        output = self.sigmoid(hidden)
        
        return output
    
    def backward(self, x, y_true, y_pred):
        """
        反向传播（简化版）
        
        参数：
            x: 输入数据
            y_true: 真实标签
            y_pred: 预测输出
        """
        batch_size = x.shape[0]
        
        # 交叉熵损失的梯度
        # dL/dy = (y_pred - y_true) / (y_pred * (1 - y_pred))
        # 为了稳定性，使用均方误差
        error = y_pred - y_true
        
        # 梯度
        d_weights = np.dot(x.T, error) / batch_size
        d_bias = error.mean(axis=0)
        
        # 梯度裁剪，防止梯度爆炸
        d_weights = np.clip(d_weights, -1, 1)
        d_bias = np.clip(d_bias, -1, 1)
        
        # 更新权重
        self.weights -= self.lr * d_weights
        self.bias -= self.lr * d_bias
    
    def train_step(self, X, Y):
        """单步训练"""
        y_pred = self.forward(X)
        self.backward(X, Y, y_pred)
        return y_pred
    
    def predict(self, X):
        """预测"""
        return self.forward(X)


class LaserScanSimulator:
    """
    模拟激光扫描数据
    """
    
    def __init__(self, num_rays, max_range, fov):
        """
        初始化激光扫描模拟器
        
        参数：
            num_rays: 射线数量
            max_range: 最大测距范围
            fov: 视场角（弧度）
        """
        self.num_rays = num_rays
        self.max_range = max_range
        self.fov = fov
        
        # 射线的角度
        self.angles = np.linspace(-fov/2, fov/2, num_rays)
    
    def scan(self, sensor_pose, obstacles):
        """
        执行一次扫描
        
        参数：
            sensor_pose: 传感器位姿 (x, y, theta)
            obstacles: 障碍物列表 [(x, y, w, h), ...]
        
        返回：
            distances: 每个射线测量的距离
        """
        sx, sy, stheta = sensor_pose
        distances = np.ones(self.num_rays) * self.max_range
        
        for i, angle in enumerate(self.angles):
            ray_angle = stheta + angle
            
            # 找到与障碍物的交点
            min_dist = self.max_range
            
            for ox, oy, ow, oh in obstacles:
                dist = self.ray_box_intersection(
                    sx, sy, ray_angle, ox, oy, ow, oh
                )
                if dist is not None and dist < min_dist:
                    min_dist = dist
            
            distances[i] = min_dist
        
        return distances
    
    def ray_box_intersection(self, ox, oy, angle, bx, by, bw, bh):
        """
        计算射线与轴对齐矩形的交点
        
        参数：
            ox, oy: 射线起点
            angle: 射线角度
            bx, by, bw, bh: 矩形中心x,y 和 宽高
        """
        dx = np.cos(angle)
        dy = np.sin(angle)
        
        # 矩形的边界
        x_min = bx - bw / 2
        x_max = bx + bw / 2
        y_min = by - bh / 2
        y_max = by + bh / 2
        
        t_min = 0
        t_max = self.max_range
        
        # 检查x方向的交点
        if abs(dx) > 1e-10:
            t1 = (x_min - ox) / dx
            t2 = (x_max - ox) / dx
            if t1 > t2:
                t1, t2 = t2, t1
            if t1 > t_min:
                t_min = t1
            if t2 < t_max:
                t_max = t2
        
        # 检查y方向的交点
        if abs(dy) > 1e-10:
            t1 = (y_min - oy) / dy
            t2 = (y_max - oy) / dy
            if t1 > t2:
                t1, t2 = t2, t1
            if t1 > t_min:
                t_min = t1
            if t2 < t_max:
                t_max = t2
        
        if t_min < t_max and t_min > 0:
            return t_min
        return None


def create_training_data(num_samples, grid_size, num_rays, max_range, fov):
    """
    创建训练数据
    
    参数：
        num_samples: 样本数量
        grid_size: 网格大小
        num_rays: 射线数量
        max_range: 最大测距
        fov: 视场角
    
    返回：
        X: 输入数据 (num_samples, num_rays)
        Y: 标签数据 (num_samples, grid_size * grid_size)
    """
    scanner = LaserScanSimulator(num_rays, max_range, fov)
    
    X = []
    Y = []
    
    for _ in range(num_samples):
        # 随机放置传感器
        sensor_x = np.random.uniform(-4, 4)
        sensor_y = np.random.uniform(-4, 4)
        sensor_theta = np.random.uniform(-np.pi, np.pi)
        sensor_pose = (sensor_x, sensor_y, sensor_theta)
        
        # 随机生成障碍物（1-3个）
        num_obstacles = np.random.randint(1, 4)
        obstacles = []
        for _ in range(num_obstacles):
            ox = np.random.uniform(-3, 3)
            oy = np.random.uniform(-3, 3)
            ow = np.random.uniform(0.3, 1.0)
            oh = np.random.uniform(0.3, 1.0)
            obstacles.append((ox, oy, ow, oh))
        
        # 执行扫描
        distances = scanner.scan(sensor_pose, obstacles)
        X.append(distances)
        
        # 生成真值grid
        grid = np.zeros((grid_size, grid_size))
        resolution = 10.0 / grid_size  # 10m x 10m 区域
        
        for ox, oy, ow, oh in obstacles:
            # 找到障碍物占据的网格
            gx_min = int((ox - ow/2 + 5) / resolution)
            gx_max = int((ox + ow/2 + 5) / resolution)
            gy_min = int((oy - oh/2 + 5) / resolution)
            gy_max = int((oy + oh/2 + 5) / resolution)
            
            gx_min = np.clip(gx_min, 0, grid_size - 1)
            gx_max = np.clip(gx_max, 0, grid_size - 1)
            gy_min = np.clip(gy_min, 0, grid_size - 1)
            gy_max = np.clip(gy_max, 0, grid_size - 1)
            
            grid[gy_min:gy_max+1, gx_min:gx_max+1] = 1.0
        
        Y.append(grid.flatten())
    
    return np.array(X), np.array(Y)


def visualize_prediction(net, X, Y_true, grid_size):
    """可视化网络预测结果"""
    # 获取预测
    Y_pred = net.predict(X)
    
    # 重建grid
    pred_grid = Y_pred[0].reshape(grid_size, grid_size)
    true_grid = Y_true[0].reshape(grid_size, grid_size)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 输入激光扫描
    ax = axes[0]
    angles = np.linspace(-np.pi/4, np.pi/4, len(X[0]))
    ax.plot(angles * 180 / np.pi, X[0], 'b-', linewidth=2)
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Distance (m)')
    ax.set_title('Laser Scan Input')
    ax.grid(True)
    ax.set_ylim(0, 6)
    
    # 真值
    ax = axes[1]
    extent = [-5, 5, -5, 5]
    im1 = ax.imshow(true_grid, extent=extent, origin='lower', cmap='gray')
    ax.set_title('Ground Truth')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    plt.colorbar(im1, ax=ax)
    
    # 预测
    ax = axes[2]
    im2 = ax.imshow(pred_grid, extent=extent, origin='lower', cmap='gray')
    ax.set_title('Network Prediction')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    plt.colorbar(im2, ax=ax)
    
    plt.tight_layout()
    plt.show()


def demo():
    """演示神经网络occupancy estimation"""
    print("=" * 50)
    print("神经网络 Occupancy Grid Estimation 演示")
    print("=" * 50)
    
    # 参数设置
    grid_size = 20
    num_rays = 36
    max_range = 6.0
    fov = np.pi / 2  # 90度视场
    num_samples = 500
    
    print(f"\n参数设置:")
    print(f"  网格大小: {grid_size} x {grid_size}")
    print(f"  激光射线数: {num_rays}")
    print(f"  最大测距: {max_range}m")
    print(f"  视场角: {fov * 180 / np.pi}度")
    print(f"  训练样本数: {num_samples}")
    
    # 创建训练数据
    print("\n生成训练数据...")
    X_train, Y_train = create_training_data(
        num_samples, grid_size, num_rays, max_range, fov
    )
    
    # 创建网络
    input_dim = num_rays
    output_dim = grid_size * grid_size
    net = SimpleOccupancyNet(input_dim, 128, output_dim, grid_size)
    
    # 训练
    print("\n开始训练...")
    epochs = 100
    batch_size = 32
    
    losses = []
    for epoch in range(epochs):
        # 随机打乱
        indices = np.random.permutation(num_samples)
        total_loss = 0
        
        for i in range(0, num_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_train[batch_idx]
            Y_batch = Y_train[batch_idx]
            
            # 训练一步
            Y_pred = net.train_step(X_batch, Y_batch)
            
            # 计算损失（均方误差）
            loss = np.mean((Y_pred - Y_batch) ** 2)
            total_loss += loss
        
        avg_loss = total_loss / (num_samples / batch_size)
        losses.append(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # 绘制损失曲线
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(losses, 'b-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Training Loss')
    ax.grid(True)
    plt.tight_layout()
    plt.show()
    
    # 测试预测
    print("\n生成测试样本...")
    X_test, Y_test = create_training_data(1, grid_size, num_rays, max_range, fov)
    
    # 可视化结果
    visualize_prediction(net, X_test, Y_test, grid_size)
    
    # 计算预测精度
    Y_pred_test = net.predict(X_test)
    pred_grid = Y_pred_test[0].reshape(grid_size, grid_size)
    true_grid = Y_test[0].reshape(grid_size, grid_size)
    
    # 二值化预测
    pred_binary = (pred_grid > 0.5).astype(float)
    
    # 计算准确率
    accuracy = np.mean(pred_binary == true_grid)
    print(f"\n预测准确率: {accuracy:.2%}")
    
    # 计算IoU
    intersection = np.sum(pred_binary * true_grid)
    union = np.sum(pred_binary) + np.sum(true_grid) - intersection
    iou = intersection / (union + 1e-10)
    print(f"IoU: {iou:.2%}")


if __name__ == "__main__":
    demo()
