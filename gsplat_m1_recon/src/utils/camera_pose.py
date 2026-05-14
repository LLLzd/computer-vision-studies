"""相机位姿小工具：在 COLMAP 世界系下绕竖直轴旋转，得到新外参（用于新视角渲染对比）。"""

from __future__ import annotations

import math

import numpy as np
import torch


def world_yaw_matrix_4x4(angle_rad: float, *, dtype=torch.float32, device: torch.device | None = None) -> torch.Tensor:
    """
    世界坐标绕 +Y 轴旋转 angle_rad（右手系：从上往下看逆时针为正）。
    用于 p_cam = w2c @ (Ry @ p_world_homo)，即 w2c_new = w2c @ Ry。
    """
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    ry = torch.tensor(
        [[c, 0.0, s, 0.0], [0.0, 1.0, 0.0, 0.0], [-s, 0.0, c, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    )
    return ry


def compose_w2c_world_yaw(w2c: torch.Tensor, yaw_deg: float) -> torch.Tensor:
    """在保持 COLMAP 约定下，绕世界 Y 轴旋转场景等价于相机绕物体转一圈。"""
    rad = math.radians(yaw_deg)
    ry = world_yaw_matrix_4x4(rad, dtype=w2c.dtype, device=w2c.device)
    return w2c @ ry


def numpy_w2c_to_torch(w2c_np: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(w2c_np.astype(np.float32)).to(device)
