"""
3D Gaussian Splatting (3DGS) - 核心模块

原理：
    3D Gaussian Splatting 使用大量的 3D 高斯分布来表示 3D 场景。
    每个高斯由以下参数定义：
    - 位置 (x, y, z)
    - 协方差矩阵（决定形状和方向）
    - 颜色 (r, g, b)
    - 不透明度 (alpha)

渲染过程：
    1. 将3D高斯投影到2D图像平面
    2. 计算每个像素的贡献（使用累积渲染）
    3. 使用深度排序确保正确的遮挡关系

核心公式：
    1. 高斯投影：将3D高斯转换为2D椭圆
    2. 权重计算：使用高斯函数计算每个像素的贡献
    3. 累积渲染：alpha混合

参考：
    "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
    https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation


class Gaussian3D:
    """
    3D高斯类
    
    每个3D高斯由以下参数组成：
    - position: 中心位置 (x, y, z)
    - covariance: 3x3协方差矩阵
    - color: RGB颜色 (0-1)
    - opacity: 不透明度 (0-1)
    """
    
    def __init__(self, position, covariance, color, opacity=1.0):
        """
        初始化3D高斯
        
        参数：
            position: np.array (3,) - 中心位置
            covariance: np.array (3,3) - 协方差矩阵
            color: np.array (3,) - RGB颜色
            opacity: float - 不透明度
        """
        self.position = np.array(position, dtype=np.float32)
        self.covariance = np.array(covariance, dtype=np.float32)
        self.color = np.clip(np.array(color, dtype=np.float32), 0.0, 1.0)
        self.opacity = np.clip(opacity, 0.0, 1.0)
        
        # 计算特征值和特征向量（用于可视化）
        self.eigenvalues, self.eigenvectors = np.linalg.eig(covariance)
    
    def get_principal_axes(self):
        """获取主坐标轴的长度（3sigma）"""
        return np.sqrt(self.eigenvalues) * 3
    
    def project_to_2d(self, camera_matrix):
        """
        将3D高斯投影到2D图像平面
        
        参数：
            camera_matrix: np.array (3,4) - 相机投影矩阵
        
        返回：
            2D椭圆参数：中心、半长轴、半短轴、旋转角度
        """
        # 将3D点转换为齐次坐标
        point_homogeneous = np.append(self.position, 1.0)
        
        # 投影到图像平面
        projected_homogeneous = camera_matrix @ point_homogeneous
        projected_2d = projected_homogeneous[:2] / projected_homogeneous[2]
        
        # 计算2D协方差（投影后的椭圆形状）
        # 取相机矩阵的前3x3部分（旋转+内参）
        RK = camera_matrix[:, :3]
        cov_2d = RK @ self.covariance @ RK.T
        
        # 计算2D椭圆参数
        eigenvalues_2d, eigenvectors_2d = np.linalg.eig(cov_2d[:2, :2])
        
        # 半长轴和半短轴（3sigma）
        a = np.sqrt(eigenvalues_2d[0]) * 3
        b = np.sqrt(eigenvalues_2d[1]) * 3
        
        # 旋转角度（弧度）
        angle = np.arctan2(eigenvectors_2d[1, 0], eigenvectors_2d[0, 0])
        
        return {
            'center': projected_2d,
            'a': a,      # 半长轴
            'b': b,      # 半短轴
            'angle': angle,  # 旋转角度（弧度）
            'depth': projected_homogeneous[2]  # 深度
        }
    
    def evaluate(self, points):
        """
        在给定点集上计算高斯的值
        
        参数：
            points: np.array (N, 3) - 3D点集
        
        返回：
            values: np.array (N,) - 每个点的高斯值
        """
        diff = points - self.position[np.newaxis, :]  # (N, 3)
        
        # 计算马氏距离的平方
        # (x - mu)^T * Sigma^{-1} * (x - mu)
        cov_inv = np.linalg.inv(self.covariance)
        mahalanobis_sq = np.sum(diff @ cov_inv * diff, axis=1)
        
        # 高斯函数值
        det = np.linalg.det(self.covariance)
        normalization = 1.0 / np.sqrt((2 * np.pi)**3 * det)
        
        return normalization * np.exp(-0.5 * mahalanobis_sq)
    
    def plot_3d(self, ax):
        """在3D坐标轴上绘制高斯"""
        # 绘制中心点
        ax.scatter(*self.position, color=self.color, s=50, alpha=self.opacity)
        
        # 绘制主轴
        axes = self.get_principal_axes()
        for i in range(3):
            axis_dir = self.eigenvectors[:, i] * axes[i]
            ax.plot(
                [self.position[0] - axis_dir[0], self.position[0] + axis_dir[0]],
                [self.position[1] - axis_dir[1], self.position[1] + axis_dir[1]],
                [self.position[2] - axis_dir[2], self.position[2] + axis_dir[2]],
                color=self.color, alpha=0.5
            )


class Camera:
    """
    简单的相机类
    
    定义相机的位置、朝向和内参
    """
    
    def __init__(self, position, look_at, up_vector, fov=60, image_size=(640, 480)):
        """
        初始化相机
        
        参数：
            position: np.array (3,) - 相机位置
            look_at: np.array (3,) - 注视点
            up_vector: np.array (3,) - 上方向
            fov: float - 视场角（度）
            image_size: tuple - 图像尺寸 (width, height)
        """
        self.position = np.array(position, dtype=np.float32)
        self.look_at = np.array(look_at, dtype=np.float32)
        self.up_vector = np.array(up_vector, dtype=np.float32)
        self.fov = fov
        self.image_size = image_size
        
        # 计算相机坐标系
        self.update_view_matrix()
        
        # 计算投影矩阵
        self.update_projection_matrix()
    
    def update_view_matrix(self):
        """计算视图矩阵（相机坐标系到世界坐标系）"""
        # 前向向量（指向注视点）
        forward = self.look_at - self.position
        forward = forward / np.linalg.norm(forward)
        
        # 右向量
        right = np.cross(forward, self.up_vector)
        right = right / np.linalg.norm(right)
        
        # 上向量（重新正交化）
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        # 视图矩阵（相机坐标系到世界坐标系的变换）
        # [R | t] 其中 R = [right, up, -forward]^T
        self.view_matrix = np.eye(4)
        self.view_matrix[:3, :3] = np.column_stack([right, up, -forward])
        self.view_matrix[:3, 3] = -self.view_matrix[:3, :3] @ self.position
    
    def update_projection_matrix(self):
        """计算投影矩阵"""
        width, height = self.image_size
        aspect_ratio = width / height
        
        # 焦距（从fov计算）
        fov_rad = np.deg2rad(self.fov)
        f = height / (2 * np.tan(fov_rad / 2))
        
        # 简单的针孔相机投影矩阵（3x4）
        # [f, 0, cx, 0]
        # [0, f, cy, 0]
        # [0, 0, 1, 0]
        self.projection_matrix = np.array([
            [f, 0, width / 2, 0],
            [0, f, height / 2, 0],
            [0, 0, 1, 0]
        ], dtype=np.float32)
    
    def get_full_projection_matrix(self):
        """获取完整的投影矩阵（投影矩阵 @ 视图矩阵）"""
        # 投影矩阵是3x4，视图矩阵是4x4，直接相乘
        return self.projection_matrix @ self.view_matrix


class GaussianRenderer:
    """
    高斯渲染器
    
    将3D高斯渲染到2D图像
    """
    
    def __init__(self, image_size):
        """
        初始化渲染器
        
        参数：
            image_size: tuple - 图像尺寸 (width, height)
        """
        self.width, self.height = image_size
    
    def render(self, gaussians, camera, show_depth=False):
        """
        渲染3D高斯到2D图像
        
        参数：
            gaussians: list - Gaussian3D对象列表
            camera: Camera对象
            show_depth: bool - 是否显示深度图
        
        返回：
            image: np.array (height, width, 3) - 渲染后的图像
        """
        # 初始化图像
        image = np.zeros((self.height, self.width, 3), dtype=np.float32)
        depth_buffer = np.ones((self.height, self.width)) * np.inf
        
        # 获取投影矩阵
        proj_matrix = camera.get_full_projection_matrix()
        
        # 投影所有高斯并按深度排序
        projected_gaussians = []
        for g in gaussians:
            proj = g.project_to_2d(proj_matrix)
            projected_gaussians.append({
                'gaussian': g,
                'projected': proj
            })
        
        # 按深度排序（从后到前）
        projected_gaussians.sort(key=lambda x: x['projected']['depth'], reverse=True)
        
        # 渲染每个高斯
        for item in projected_gaussians:
            g = item['gaussian']
            proj = item['projected']
            
            cx, cy = proj['center']
            a, b = proj['a'], proj['b']
            angle = np.rad2deg(proj['angle'])
            depth = proj['depth']
            
            # 计算影响范围（考虑椭圆大小）
            margin = max(a, b) + 5
            x_min = max(0, int(cx - margin))
            x_max = min(self.width - 1, int(cx + margin))
            y_min = max(0, int(cy - margin))
            y_max = min(self.height - 1, int(cy + margin))
            
            if x_min > x_max or y_min > y_max:
                continue
            
            # 生成像素坐标网格
            yy, xx = np.mgrid[y_min:y_max+1, x_min:x_max+1]
            pixels = np.stack([xx, yy], axis=-1)  # (H, W, 2)
            
            # 计算像素到椭圆中心的距离
            dx = pixels[..., 0] - cx
            dy = pixels[..., 1] - cy
            
            # 旋转坐标系
            cos_theta = np.cos(np.deg2rad(angle))
            sin_theta = np.sin(np.deg2rad(angle))
            
            dx_rot = dx * cos_theta + dy * sin_theta
            dy_rot = -dx * sin_theta + dy * cos_theta
            
            # 计算椭圆方程的值
            # (x/a)^2 + (y/b)^2
            ellipse_val = (dx_rot / a)**2 + (dy_rot / b)**2
            
            # 计算高斯权重（使用2D高斯近似）
            weight = np.exp(-0.5 * ellipse_val) * g.opacity
            
            # 累积到图像（alpha混合）
            for y in range(y_min, y_max + 1):
                for x in range(x_min, x_max + 1):
                    idx_y = y - y_min
                    idx_x = x - x_min
                    w = weight[idx_y, idx_x]
                    
                    # 深度测试
                    if depth < depth_buffer[y, x]:
                        image[y, x] = (1 - w) * image[y, x] + w * g.color
                        depth_buffer[y, x] = depth
        
        # 裁剪到[0, 1]范围
        image = np.clip(image, 0.0, 1.0)
        
        if show_depth:
            # 归一化深度图
            depth_normalized = (depth_buffer - depth_buffer.min()) / (depth_buffer.max() - depth_buffer.min())
            return image, depth_normalized
        
        return image


def create_test_scene(num_gaussians=100):
    """
    创建一个测试场景
    
    参数：
        num_gaussians: int - 高斯数量
    
    返回：
        gaussians: list - Gaussian3D对象列表
    """
    gaussians = []
    
    np.random.seed(42)
    
    for i in range(num_gaussians):
        # 随机位置（在一个立方体空间内）
        x = np.random.uniform(-2, 2)
        y = np.random.uniform(-2, 2)
        z = np.random.uniform(2, 6)  # 在相机前方2-6米
        
        # 随机缩放
        scale = np.random.uniform(0.1, 0.5)
        
        # 随机旋转
        rot = Rotation.random()
        rotation_matrix = rot.as_matrix()
        
        # 协方差矩阵（缩放后的球体）
        sigma = np.eye(3) * scale**2
        covariance = rotation_matrix @ sigma @ rotation_matrix.T
        
        # 随机颜色
        color = np.random.rand(3)
        
        # 随机不透明度
        opacity = np.random.uniform(0.3, 1.0)
        
        gaussian = Gaussian3D(
            position=[x, y, z],
            covariance=covariance,
            color=color,
            opacity=opacity
        )
        
        gaussians.append(gaussian)
    
    return gaussians


def visualize_3d_scene(gaussians):
    """可视化3D场景"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for g in gaussians:
        g.plot_3d(ax)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Gaussian Splatting - Scene Visualization')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(0, 8)
    
    plt.show()


def demo():
    """演示3D Gaussian Splatting"""
    print("=" * 60)
    print("3D Gaussian Splatting 演示")
    print("=" * 60)
    
    # 参数设置
    image_size = (320, 240)
    num_gaussians = 150
    
    print(f"\n参数设置:")
    print(f"  图像尺寸: {image_size[0]} x {image_size[1]}")
    print(f"  高斯数量: {num_gaussians}")
    
    # 创建测试场景
    print("\n创建3D高斯场景...")
    gaussians = create_test_scene(num_gaussians)
    
    # 创建相机
    camera = Camera(
        position=[0, 0, 0],      # 相机位置
        look_at=[0, 0, 4],       # 注视点（场景中心）
        up_vector=[0, 1, 0],     # 上方向
        fov=60,                  # 视场角
        image_size=image_size    # 图像尺寸
    )
    
    # 创建渲染器
    renderer = GaussianRenderer(image_size)
    
    # 渲染图像
    print("渲染图像...")
    image = renderer.render(gaussians, camera)
    
    # 可视化3D场景
    print("\n显示3D场景可视化...")
    visualize_3d_scene(gaussians)
    
    # 显示渲染结果
    print("\n显示渲染结果...")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image)
    ax.set_title('3D Gaussian Splatting - Rendered Image')
    ax.axis('off')
    plt.show()
    
    print("\n渲染完成！")


if __name__ == "__main__":
    demo()
