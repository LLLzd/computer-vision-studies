"""
合成数据生成脚本

生成BEV 2D Box真值数据：
- 范围: 200米 x 200米 (左下角(0,0)为原点)
- 帧数: 200帧 (20秒 x 10帧/秒)
- 目标: 4辆相同大小的车（4.5m x 1.8m）
- 复杂轨迹：曲线、变加速、圆弧运动
"""

import json
import numpy as np
import os
from typing import List, Dict


VEHICLE_CONFIG = {
    "vehicle": {
        "name": "Vehicle",
        "length": 4.5,
        "width": 1.8,
        "color": (100, 150, 255),
        "max_speed": 10.0,
        "acceleration": 1.5,
    },
}


def compute_corners(center_x: float, center_y: float, length: float,
                    width: float, yaw: float) -> List[float]:
    """计算四点坐标（米坐标系）"""
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    half_l = length / 2
    half_w = width / 2

    corners = np.array([
        [-half_l, -half_w],
        [half_l, -half_w],
        [half_l, half_w],
        [-half_l, half_w],
    ])

    rotated = np.zeros_like(corners)
    rotated[:, 0] = cos_yaw * corners[:, 0] - sin_yaw * corners[:, 1]
    rotated[:, 1] = sin_yaw * corners[:, 0] + cos_yaw * corners[:, 1]

    rotated[:, 0] += center_x
    rotated[:, 1] += center_y

    return rotated.flatten().tolist()


class Vehicle:
    """车辆基类"""
    def __init__(self, vehicle_id: str, start_pos: tuple, start_speed: float):
        self.vehicle_id = vehicle_id
        self.x = start_pos[0]
        self.y = start_pos[1]
        self.yaw = 0.0
        self.speed = start_speed
        self.length = 4.5
        self.width = 1.8

    def update(self, frame_id: int, total_frames: int, dt: float):
        """更新车辆状态（需要子类实现）"""
        pass

    def get_box(self, frame_id: int) -> Dict:
        """获取当前帧的Box信息（米坐标系）"""
        vx = np.cos(self.yaw) * self.speed
        vy = np.sin(self.yaw) * self.speed

        corners = compute_corners(self.x, self.y, self.length, self.width, self.yaw)

        return {
            "frame_id": frame_id,
            "is_key_frame": frame_id % 10 == 0,
            "category": "vehicle",
            "track_id": self.vehicle_id,
            "center": [self.x, self.y],
            "velocity": [vx, vy],
            "speed": self.speed,
            "yaw": self.yaw,
            "dimensions": [self.width, self.length],
            "bbox_bev_2d": corners,
            "score": 1.0,
            "category_name": "Vehicle",
            "color": [100, 150, 255],
        }


class Vehicle1Curved(Vehicle):
    """id1: 贝塞尔曲线运动"""
    def __init__(self, vehicle_id: str):
        super().__init__(vehicle_id, (100.0, 10.0), 9.0)
        self.start_pos = np.array([100.0, 10.0])
        self.end_pos = np.array([100.0, 190.0])
        # 控制点（让轨迹向右弯曲）
        self.control1 = np.array([140.0, 70.0])
        self.control2 = np.array([140.0, 130.0])

    def update(self, frame_id: int, total_frames: int, dt: float):
        t = frame_id / (total_frames - 1)
        
        # 三次贝塞尔曲线
        p = (1-t)**3 * self.start_pos + \
            3*(1-t)**2 * t * self.control1 + \
            3*(1-t)*t**2 * self.control2 + \
            t**3 * self.end_pos
        
        self.x, self.y = p[0], p[1]
        
        # 计算yaw（基于切线方向）
        if frame_id < total_frames - 1:
            t_next = (frame_id + 1) / (total_frames - 1)
            p_next = (1-t_next)**3 * self.start_pos + \
                     3*(1-t_next)**2 * t_next * self.control1 + \
                     3*(1-t_next)*t_next**2 * self.control2 + \
                     t_next**3 * self.end_pos
            dx = p_next[0] - p[0]
            dy = p_next[1] - p[1]
            self.yaw = np.arctan2(dy, dx)


class Vehicle2VariableAccel(Vehicle):
    """id2: 变加速运动（剧烈波动）"""
    def __init__(self, vehicle_id: str):
        super().__init__(vehicle_id, (100.0, 190.0), 0.0)
        self.start_pos = np.array([100.0, 190.0])
        self.end_pos = np.array([100.0, 10.0])
        self.yaw = -np.pi / 2
        self.base_speed = 8.0
        self.path_positions = []
        self.path_initialized = False

    def _generate_path(self, total_frames):
        """生成完整路径点，考虑速度变化"""
        x = self.start_pos[0]
        y = self.start_pos[1]
        self.path_positions = []
        
        for frame_id in range(total_frames):
            self.path_positions.append((x, y))
            
            t = frame_id / (total_frames - 1)
            # 剧烈变化的速度：多个正弦叠加 + 零速段
            speed = self._calculate_speed(t)
            
            # 根据速度和方向移动
            dx = np.cos(self.yaw) * speed * 0.1
            dy = np.sin(self.yaw) * speed * 0.1
            
            # 限制在终点范围内
            new_y = y + dy
            new_y = max(self.end_pos[1], min(self.start_pos[1], new_y))
            
            y = new_y

    def _calculate_speed(self, t):
        """计算剧烈变化的速度"""
        # 主趋势：整体先快后慢
        base = 8.0 * np.sin(t * np.pi)
        
        # 高频波动：快速变化
        high_freq = 3.0 * np.sin(t * 6 * np.pi)
        
        # 突停段：特定时间点速度降为0
        stop = 1.0
        if 0.25 < t < 0.35:
            stop = max(0, 1.0 - abs((t - 0.3) / 0.05))
        if 0.7 < t < 0.8:
            stop = max(0, 1.0 - abs((t - 0.75) / 0.05))
        
        speed = (base + high_freq) * stop
        return max(0, speed)

    def update(self, frame_id: int, total_frames: int, dt: float):
        if not self.path_initialized:
            self._generate_path(total_frames)
            self.path_initialized = True
            
        t = frame_id / (total_frames - 1)
        
        self.speed = self._calculate_speed(t)
        
        # 使用预计算的路径位置
        if frame_id < len(self.path_positions):
            self.x, self.y = self.path_positions[frame_id]


class Vehicle3Arc(Vehicle):
    """id3: 圆弧运动"""
    def __init__(self, vehicle_id: str):
        super().__init__(vehicle_id, (100.0, 10.0), 9.0)
        self.center = np.array([100.0, 100.0])  # 圆弧中心
        self.radius = 90.0
        self.start_angle = -np.pi / 2
        self.end_angle = 0.0

    def update(self, frame_id: int, total_frames: int, dt: float):
        t = frame_id / (total_frames - 1)
        angle = self.start_angle + (self.end_angle - self.start_angle) * t
        
        self.x = self.center[0] + self.radius * np.cos(angle)
        self.y = self.center[1] + self.radius * np.sin(angle)
        
        # yaw是切线方向（垂直于半径，逆时针旋转90度）
        self.yaw = angle + np.pi / 2


class Vehicle4CurvedVariableAccel(Vehicle):
    """id4: 曲线+变加速运动（剧烈波动）"""
    def __init__(self, vehicle_id: str):
        super().__init__(vehicle_id, (100.0, 10.0), 5.0)
        self.start_pos = np.array([100.0, 10.0])
        self.end_pos = np.array([10.0, 100.0])
        # 控制点（向左弯曲）
        self.control1 = np.array([70.0, 40.0])
        self.control2 = np.array([40.0, 70.0])
        self.min_speed = 3.0
        self.max_speed = 15.0
        self.path_positions = []
        self.path_initialized = False

    def _generate_path(self, total_frames):
        """生成完整路径点"""
        for frame_id in range(total_frames):
            t = frame_id / (total_frames - 1)
            p = (1-t)**2 * self.start_pos + \
                2*(1-t)*t * self.control1 + \
                t**2 * self.end_pos
            self.path_positions.append((p[0], p[1]))

    def _calculate_speed(self, t):
        """计算剧烈变化的速度"""
        # 主趋势
        base = self.min_speed + (self.max_speed - self.min_speed) * np.sin(t * np.pi)
        
        # 高频波动
        high_freq = 4.0 * np.sin(t * 8 * np.pi)
        
        # 突停/急加速
        stop = 1.0
        if 0.15 < t < 0.2:
            stop = max(0, 1.0 - abs((t - 0.175)/0.025))
        if 0.5 < t < 0.55:
            stop = max(0, 1.0 - abs((t - 0.525)/0.025))
        if 0.85 < t < 0.9:
            stop = max(0, 1.0 - abs((t - 0.875)/0.025))
        
        # 偶尔急加速
        boost = 1.0
        if 0.35 < t < 0.4:
            boost = 1.5 + 0.5 * np.sin(((t - 0.35)/0.05) * np.pi)
        if 0.65 < t < 0.7:
            boost = 1.5 + 0.5 * np.sin(((t - 0.65)/0.05) * np.pi)
        
        speed = (base + high_freq) * stop * boost
        return max(0, speed)

    def update(self, frame_id: int, total_frames: int, dt: float):
        if not self.path_initialized:
            self._generate_path(total_frames)
            self.path_initialized = True
        
        t = frame_id / (total_frames - 1)
        
        self.speed = self._calculate_speed(t)
        
        if frame_id < len(self.path_positions):
            self.x, self.y = self.path_positions[frame_id]
        
        # 计算yaw
        if frame_id < total_frames - 1 and frame_id < len(self.path_positions)-1:
            p_next = self.path_positions[frame_id + 1]
            dx = p_next[0] - self.x
            dy = p_next[1] - self.y
            self.yaw = np.arctan2(dy, dx)


def clamp_position(x: float, y: float, margin: float = 5.0, bounds: float = 200.0) -> tuple:
    """限制位置在边界内"""
    x = np.clip(x, margin, bounds - margin)
    y = np.clip(y, margin, bounds - margin)
    return x, y


def clear_file(file_path: str):
    """清空文件内容"""
    if os.path.exists(file_path):
        with open(file_path, 'w') as f:
            pass


def generate_synthetic_data(output_dir: str = "."):
    """生成合成数据"""
    print("🚀 开始生成合成数据...")

    if output_dir != ".":
        os.makedirs(output_dir, exist_ok=True)

    # 清空旧json
    key_frame_path = os.path.join(output_dir, "key_frame_boxes.json")
    full_gt_path = os.path.join(output_dir, "full_gt_boxes.json")
    metadata_path = os.path.join(output_dir, "data_metadata.json")
    clear_file(key_frame_path)
    clear_file(full_gt_path)
    clear_file(metadata_path)

    num_frames = 200
    dt = 0.1

    # 四辆车的轨迹配置
    vehicles = [
        Vehicle1Curved("1"),
        Vehicle2VariableAccel("2"),
        Vehicle3Arc("3"),
        Vehicle4CurvedVariableAccel("4"),
    ]

    all_frames = []

    for frame_id in range(num_frames):
        for vehicle in vehicles:
            vehicle.update(frame_id, num_frames, dt)
            box = vehicle.get_box(frame_id)
            all_frames.append(box)

    all_frames.sort(key=lambda x: (x["frame_id"], x["track_id"]))

    key_frames = [f for f in all_frames if f["is_key_frame"]]
    full_gt = all_frames

    with open(key_frame_path, 'w') as f:
        json.dump(key_frames, f, indent=2)
    print(f"✅ 关键帧数据已保存: {key_frame_path} ({len(key_frames)} 帧)")

    with open(full_gt_path, 'w') as f:
        json.dump(full_gt, f, indent=2)
    print(f"✅ 全帧真值已保存: {full_gt_path} ({len(full_gt)} 帧)")

    print("\n📊 数据统计:")
    print(f"  - 范围: 200米 x 200米 (左下角(0,0)为原点)")
    print(f"  - 总帧数: {num_frames}")
    print(f"  - 时间跨度: {num_frames * dt} 秒")
    print(f"  - 帧率: {1/dt} fps")
    print(f"  - 目标数量: {len(vehicles)}")
    print(f"  - 关键帧间隔: 10帧 (1秒)")
    print(f"  - 车辆大小: {VEHICLE_CONFIG['vehicle']['length']}m x {VEHICLE_CONFIG['vehicle']['width']}m")
    print(f"  - 轨迹: id1(贝塞尔曲线), id2(变加速直线), id3(圆弧), id4(曲线+变加速)")

    metadata = {
        "range_meters": [200, 200],
        "origin": "左下角(0,0)",
        "num_frames": num_frames,
        "time_duration_s": num_frames * dt,
        "frame_rate_fps": 1 / dt,
        "key_frame_interval": 10,
        "vehicle_config": VEHICLE_CONFIG["vehicle"],
        "trajectories": [
            {"track_id": "1", "description": "贝塞尔曲线运动"},
            {"track_id": "2", "description": "变加速直线运动"},
            {"track_id": "3", "description": "圆弧运动"},
            {"track_id": "4", "description": "曲线+变加速运动"},
        ],
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ 数据元信息已保存: {metadata_path}")

    print("\n🎉 合成数据生成完成!")

    return metadata


if __name__ == "__main__":
    generate_synthetic_data()
