"""
BEV 2D Box 数据格式定义

统一BEV 2D Car Box标准字段：
{
  "frame_id": 帧序号,
  "is_key_frame": 是否关键帧(true/false),
  "category": "car",  # 类别如 car, truck, van, motorcycle, tricycle
  "track_id": "001",  # 目标跟踪ID，同辆车连续帧唯一
  "center": [x, y],   # 中心点坐标
  "velocity": [vx, vy],  # 速度向量 (m/s)
  "speed": float,     # 速度标量 (m/s)
  "yaw": float,       # 朝向角 (弧度，x轴正方向为0，逆时针为正)
  "dimensions": [w, l], # [宽度, 长度] (m)
  "bbox_bev_2d": [x1, y1, x2, y2, x3, y3, x4, y4],  # BEV四个角点坐标
  "score": 1.0,        # 置信度
}
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class BEVBox2D:
    """BEV 2D Box数据结构"""
    frame_id: int
    is_key_frame: bool
    category: str
    track_id: str
    center: List[float]  # [x, y]
    velocity: List[float]  # [vx, vy]
    speed: float  # 速度标量 (m/s)
    yaw: float  # 朝向角 (弧度)
    dimensions: List[float]  # [width, length]
    bbox_bev_2d: List[float] = field(default_factory=list)  # [x1,y1,x2,y2,x3,y3,x4,y4]
    score: float = 1.0
    category_name: Optional[str] = None

    @property
    def w(self) -> float:
        return self.dimensions[0]

    @property
    def l(self) -> float:
        return self.dimensions[1]

    @property
    def vx(self) -> float:
        return self.velocity[0]

    @property
    def vy(self) -> float:
        return self.velocity[1]

    @property
    def center_x(self) -> float:
        return self.center[0]

    @property
    def center_y(self) -> float:
        return self.center[1]

    @property
    def corners(self) -> List[Tuple[float, float]]:
        """获取四个角点"""
        if len(self.bbox_bev_2d) >= 8:
            return [
                (self.bbox_bev_2d[0], self.bbox_bev_2d[1]),
                (self.bbox_bev_2d[2], self.bbox_bev_2d[3]),
                (self.bbox_bev_2d[4], self.bbox_bev_2d[5]),
                (self.bbox_bev_2d[6], self.bbox_bev_2d[7]),
            ]
        return []

    @property
    def x1(self) -> float:
        if len(self.bbox_bev_2d) >= 8:
            return min(c[0] for c in self.corners)
        return 0.0

    @property
    def y1(self) -> float:
        if len(self.bbox_bev_2d) >= 8:
            return min(c[1] for c in self.corners)
        return 0.0

    @property
    def x2(self) -> float:
        if len(self.bbox_bev_2d) >= 8:
            return max(c[0] for c in self.corners)
        return 0.0

    @property
    def y2(self) -> float:
        if len(self.bbox_bev_2d) >= 8:
            return max(c[1] for c in self.corners)
        return 0.0

    @property
    def box_width(self) -> float:
        return abs(self.x2 - self.x1)

    @property
    def box_height(self) -> float:
        return abs(self.y2 - self.y1)

    @property
    def long_side(self) -> float:
        return max(self.box_width, self.box_height)

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "frame_id": self.frame_id,
            "is_key_frame": self.is_key_frame,
            "category": self.category,
            "track_id": self.track_id,
            "center": self.center,
            "velocity": self.velocity,
            "speed": self.speed,
            "yaw": self.yaw,
            "dimensions": self.dimensions,
            "bbox_bev_2d": self.bbox_bev_2d,
            "score": self.score,
            "category_name": self.category_name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BEVBox2D":
        """从字典创建对象"""
        return cls(
            frame_id=data["frame_id"],
            is_key_frame=data.get("is_key_frame", False),
            category=data["category"],
            track_id=data["track_id"],
            center=data["center"],
            velocity=data.get("velocity", [0.0, 0.0]),
            speed=data.get("speed", 0.0),
            yaw=data.get("yaw", 0.0),
            dimensions=data.get("dimensions", [1.0, 1.0]),
            bbox_bev_2d=data.get("bbox_bev_2d", []),
            score=data.get("score", 1.0),
            category_name=data.get("category_name"),
        )


@dataclass
class TrackSequence:
    """单目标跟踪序列"""
    track_id: str
    boxes: List[BEVBox2D] = field(default_factory=list)

    def get_key_frames(self) -> List[BEVBox2D]:
        return [box for box in self.boxes if box.is_key_frame]

    def get_frame_range(self) -> tuple:
        if not self.boxes:
            return (0, 0)
        frame_ids = [box.frame_id for box in self.boxes]
        return (min(frame_ids), max(frame_ids))

    def get_box_at_frame(self, frame_id: int) -> Optional[BEVBox2D]:
        for box in self.boxes:
            if box.frame_id == frame_id:
                return box
        return None


@dataclass
class EvaluationResult:
    """评测结果数据结构 - 专业版"""
    method_name: str
    
    # IOU类指标
    iou_mean: float
    iou_std: float
    iou_median: float
    iou_90_percentile: float
    
    # 中心误差类指标 (单位：米)
    center_error_mean: float
    center_error_std: float
    center_error_median: float
    center_error_max: float
    center_error_90_percentile: float
    
    # 尺寸误差类指标 (单位：米)
    width_error_mean: float
    width_error_std: float
    height_error_mean: float
    height_error_std: float
    
    # 角点误差类指标 (单位：米)
    corner_error_mean: float
    corner_error_std: float
    
    # 航向角误差类指标 (单位：弧度)
    yaw_error_mean: float
    yaw_error_std: float
    
    # 轨迹类指标
    ade: float  # Average Displacement Error (平均位移误差)
    fde: float  # Final Displacement Error (最终位移误差)
    trajectory_length_ratio: float  # 预测轨迹长度与真实轨迹长度比
    
    # 平滑度类指标
    speed_variance: float
    acceleration_variance: float
    jerk: float  # Jerk (加加速度，衡量平滑度)
    
    # 检测类指标
    precision: float
    recall: float
    mAP: dict

    def to_dict(self) -> dict:
        return {
            "method_name": self.method_name,
            "iou_mean": self.iou_mean,
            "iou_std": self.iou_std,
            "iou_median": self.iou_median,
            "iou_90_percentile": self.iou_90_percentile,
            "center_error_mean": self.center_error_mean,
            "center_error_std": self.center_error_std,
            "center_error_median": self.center_error_median,
            "center_error_max": self.center_error_max,
            "center_error_90_percentile": self.center_error_90_percentile,
            "width_error_mean": self.width_error_mean,
            "width_error_std": self.width_error_std,
            "height_error_mean": self.height_error_mean,
            "height_error_std": self.height_error_std,
            "corner_error_mean": self.corner_error_mean,
            "corner_error_std": self.corner_error_std,
            "yaw_error_mean": self.yaw_error_mean,
            "yaw_error_std": self.yaw_error_std,
            "ade": self.ade,
            "fde": self.fde,
            "trajectory_length_ratio": self.trajectory_length_ratio,
            "speed_variance": self.speed_variance,
            "acceleration_variance": self.acceleration_variance,
            "jerk": self.jerk,
            "precision": self.precision,
            "recall": self.recall,
            "mAP": self.mAP
        }


def compute_corners(center: List[float], dimensions: List[float],
                    yaw: float) -> List[float]:
    """根据中心点、尺寸和朝向角计算四点坐标

    参数:
        center: [x, y] 中心点坐标
        dimensions: [w, l] 宽度和长度
        yaw: 朝向角 (弧度)

    返回:
        [x1,y1,x2,y2,x3,y3,x4,y4] 四个角点坐标
    """
    w, l = dimensions
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    half_w = w / 2
    half_l = l / 2

    corners_local = np.array([
        [-half_l, -half_w],
        [half_l, -half_w],
        [half_l, half_w],
        [-half_l, half_w],
    ])

    rotated = np.zeros_like(corners_local)
    rotated[:, 0] = cos_yaw * corners_local[:, 0] - sin_yaw * corners_local[:, 1]
    rotated[:, 1] = sin_yaw * corners_local[:, 0] + cos_yaw * corners_local[:, 1]

    rotated[:, 0] += center[0]
    rotated[:, 1] += center[1]

    return rotated.flatten().tolist()


import numpy as np
