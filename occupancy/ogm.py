"""
Occupancy Grid Mapping (OGM) - 贝叶斯方法

原理：
    Occupancy Grid Mapping 将环境划分为网格，每个格子存储该位置被占用的概率。
    使用贝叶斯推理结合多次传感器观测来更新占用概率。

核心公式：
    P(occ | z) = P(z | occ) * P(occ) / P(z)
    
    -logodds形式（便于加法计算）：
    l(occ | z) = l(z | occ) + l(occ) - l(z)
    
其中：
    - P(occ): 先验概率（通常为0.5）
    - P(z | occ): 似然（传感器测量模型）
    - l(x) = logit(P(x)) = log(P(x) / (1 - P(x)))

传感器模型：
    - 命中（hit）：物体在传感器范围内，被检测到
    - 未命中（miss）：激光穿过空区域
    - 不确定：超出传感器范围
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class OccupancyGridMap:
    """
    使用贝叶斯方法构建二维占用网格地图
    """
    
    def __init__(self, size, resolution):
        """
        初始化占用网格地图
        
        参数：
            size: 地图尺寸（米）
            resolution: 网格分辨率（米/格）
        """
        self.size = size
        self.resolution = resolution
        self.grid_size = int(size / resolution)
        
        # 初始化网格为未知状态（0.5概率）
        # 使用log-odds形式存储，便于贝叶斯更新
        # logodds = log(p / (1-p))，当p=0.5时，logodds=0
        self.log_odds = np.zeros((self.grid_size, self.grid_size))
        
        # 先验概率 P(occ) = 0.5，对应 log-odds = 0
        self.prior = 0.5
        
        # 占用和空闲的log-odds似然
        # P(z=occupied | occ) 高，P(z=free | occ) 低
        self.l_occ = np.log(0.7 / 0.3)   # 命中时的log-odds
        self.l_free = np.log(0.3 / 0.7)  # 未命中时的log-odds
        
    def world_to_grid(self, x, y):
        """世界坐标转换为网格索引"""
        gx = int((x + self.size / 2) / self.resolution)
        gy = int((y + self.size / 2) / self.resolution)
        return gx, gy
    
    def grid_to_world(self, gx, gy):
        """网格索引转换为世界坐标"""
        x = gx * self.resolution - self.size / 2
        y = gy * self.resolution - self.size / 2
        return x, y
    
    def bresenham(self, x0, y0, x1, y1):
        """
        Bresenham算法：计算两点之间的网格单元
        用于确定激光射线经过的空闲区域
        """
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        
        return points
    
    def update(self, sensor_pose, measurement, max_range):
        """
        使用贝叶斯方法更新网格
        
        参数：
            sensor_pose: 传感器位置 (x, y, theta) - 世界坐标
            measurement: 测量距离（米）
            max_range: 传感器最大范围
        """
        sx, sy, stheta = sensor_pose
        
        # 计算命中点（物体位置）的世界坐标
        hit_x = sx + measurement * np.cos(stheta)
        hit_y = sy + measurement * np.sin(stheta)
        
        # 转换为网格坐标
        sx_g, sy_g = self.world_to_grid(sx, sy)
        hit_gx, hit_gy = self.world_to_grid(hit_x, hit_y)
        
        # 使用Bresenham算法计算射线经过的所有网格单元
        ray_cells = self.bresenham(sx_g, sy_g, hit_gx, hit_gy)
        
        # 更新每个网格单元
        for (gx, gy) in ray_cells:
            # 检查边界
            if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                if gx == hit_gx and gy == hit_gy:
                    # 命中点：增加占用概率
                    self.log_odds[gy, gx] += self.l_occ
                else:
                    # 未命中点（射线经过的空闲区域）：降低占用概率
                    self.log_odds[gy, gx] += self.l_free
    
    def get_probability(self):
        """将log-odds转换为概率"""
        # logodds = log(p / (1-p))
        # p = 1 / (1 + exp(-logodds))
        odds = np.exp(self.log_odds)
        return odds / (1 + odds)
    
    def visualize(self, sensor_poses=None, show=True):
        """可视化占用网格"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 获取占用概率
        prob = self.get_probability()
        
        # 绘制网格
        extent = [-self.size/2, self.size/2, -self.size/2, self.size/2]
        im = ax.imshow(prob, extent=extent, cmap='gray', origin='lower', vmin=0, vmax=1)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, label='Occupancy Probability')
        
        # 绘制传感器位置
        if sensor_poses is not None:
            for pose in sensor_poses:
                sx, sy, _ = pose
                ax.plot(sx, sy, 'bo', markersize=10, label='Sensor')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Occupancy Grid Map (Bayesian Method)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if show:
            plt.show()
        
        return fig, ax


def create_simple_map():
    """创建一个简单的测试场景"""
    size = 10.0  # 10米 x 10米
    resolution = 0.1  # 10cm分辨率
    max_range = 5.0  # 传感器最大范围5米
    
    # 创建OGM
    ogm = OccupancyGridMap(size, resolution)
    
    # 模拟传感器在中心 (0, 0) 位置
    sensor_pose = (0, 0, 0)  # x, y, theta (朝右)
    
    # 模拟一些障碍物
    obstacles = [
        (3, 0, 1, 1),   # (x, y, width, height)
        (-2, 3, 2, 0.5),
        (1, -2, 0.5, 2),
    ]
    
    # 从不同角度进行测量
    angles = np.linspace(-np.pi/2, np.pi/2, 19)  # -90度到+90度
    
    for angle in angles:
        for obs_x, obs_y, obs_w, obs_h in obstacles:
            # 计算障碍物边界到传感器的距离
            dx = obs_x - sensor_pose[0]
            dy = obs_y - sensor_pose[1]
            dist = np.sqrt(dx**2 + dy**2)
            
            # 简化的射线投射
            theta = angle
            hit_x = sensor_pose[0] + dist * np.cos(theta)
            hit_y = sensor_pose[1] + dist * np.sin(theta)
            
            # 更新OGM
            ogm.update(sensor_pose, dist, max_range)
    
    return ogm


def demo():
    """演示贝叶斯occupancy grid mapping"""
    print("=" * 50)
    print("贝叶斯 Occupancy Grid Mapping 演示")
    print("=" * 50)
    
    # 创建简单的测试地图
    ogm = create_simple_map()
    
    # 可视化结果
    ogm.visualize()
    
    # 打印统计信息
    prob = ogm.get_probability()
    print(f"\n网格大小: {ogm.grid_size} x {ogm.grid_size}")
    print(f"分辨率: {ogm.resolution} m/cell")
    print(f"占用格子数 (p > 0.6): {np.sum(prob > 0.6)}")
    print(f"空闲格子数 (p < 0.4): {np.sum(prob < 0.4)}")
    print(f"未知格子数: {np.sum((prob >= 0.4) & (prob <= 0.6))}")
    
    plt.show()


if __name__ == "__main__":
    demo()
