from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class CameraModel:
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray


@dataclass
class ImagePose:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """COLMAP 四元数 (qw,qx,qy,qz) 转旋转矩阵。"""
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float32,
    )


def _non_comment_lines(path: Path) -> List[str]:
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    return lines


def read_cameras_txt(path: str | Path) -> Dict[int, CameraModel]:
    cameras: Dict[int, CameraModel] = {}
    for line in _non_comment_lines(Path(path)):
        parts = line.split()
        camera_id = int(parts[0])
        model = parts[1]
        width = int(parts[2])
        height = int(parts[3])
        params = np.array([float(v) for v in parts[4:]], dtype=np.float32)
        cameras[camera_id] = CameraModel(camera_id, model, width, height, params)
    return cameras


def read_images_txt(path: str | Path) -> Dict[int, ImagePose]:
    images: Dict[int, ImagePose] = {}
    lines = _non_comment_lines(Path(path))
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        image_id = int(parts[0])
        qvec = np.array([float(v) for v in parts[1:5]], dtype=np.float32)
        tvec = np.array([float(v) for v in parts[5:8]], dtype=np.float32)
        camera_id = int(parts[8])
        name = parts[9]
        images[image_id] = ImagePose(image_id, qvec, tvec, camera_id, name)
        i += 2  # 下一行为 2D-3D 对应，可在本工程中忽略
    return images


def read_points3d_txt(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    xyz: List[List[float]] = []
    rgb: List[List[float]] = []
    for line in _non_comment_lines(Path(path)):
        parts = line.split()
        xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
        rgb.append([float(parts[4]) / 255.0, float(parts[5]) / 255.0, float(parts[6]) / 255.0])
    if not xyz:
        raise RuntimeError("points3D.txt 为空，COLMAP 重建失败或没有三角化点。")
    return np.asarray(xyz, dtype=np.float32), np.asarray(rgb, dtype=np.float32)
