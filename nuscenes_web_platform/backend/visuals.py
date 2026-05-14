"""Server-side renders: LiDAR BEV, camera with 3D box projection (MVP)."""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from PIL import Image, ImageDraw
from pyquaternion import Quaternion

if TYPE_CHECKING:
    from nuscenes.nuscenes import NuScenes

# demo_by_nuscenes on path at runtime
from utils import (  # type: ignore[import-untyped]
    get_ego_pose,
    get_sample_data_path,
    get_sensor_order,
    is_box_in_range,
    project_box_to_bev,
)
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points


def render_lidar_bev_png(nusc: NuScenes, sample_token: str, max_points: int = 40_000) -> bytes:
    """Scatter BEV (x, y) from LIDAR_TOP .bin file."""
    sample = nusc.get("sample", sample_token)
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    rel = lidar_sd["filename"]
    path = (Path(nusc.dataroot) / rel).resolve()
    pts = np.fromfile(str(path), dtype=np.float32).reshape(-1, 5)[:, :3]
    if pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx]
    xy = pts[:, :2]
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    ax.scatter(xy[:, 0], xy[:, 1], s=0.2, c=pts[:, 2], cmap="viridis", alpha=0.8)
    ax.set_aspect("equal")
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_title("LiDAR BEV (ego frame)")
    ax.grid(True, alpha=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def render_camera_with_boxes_png(nusc: NuScenes, sample_token: str, channel: str) -> bytes:
    """Single camera JPEG with projected 3D boxes (vehicles emphasized)."""
    img_path = get_sample_data_path(nusc, sample_token, channel)
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    sample = nusc.get("sample", sample_token)
    ego_pose = get_ego_pose(nusc, sample_token)
    cam_sd = nusc.get("sample_data", sample["data"][channel])
    calibrated_sensor = nusc.get(
        "calibrated_sensor", cam_sd["calibrated_sensor_token"]
    )
    cam_intrinsic = np.array(calibrated_sensor["camera_intrinsic"]).reshape(3, 3)

    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        if "vehicle" not in ann["category_name"]:
            continue
        box = Box(
            center=ann["translation"],
            size=ann["size"],
            orientation=Quaternion(ann["rotation"]),
        )
        box.translate(-np.array(ego_pose["translation"]))
        box.rotate(Quaternion(ego_pose["rotation"]).inverse)
        box.translate(-np.array(calibrated_sensor["translation"]))
        box.rotate(Quaternion(calibrated_sensor["rotation"]).inverse)
        corners = view_points(box.corners(), cam_intrinsic, normalize=True)
        if np.all(corners[2, :] > 0):
            idx = [0, 1, 5, 4, 0, 2, 6, 4, 6, 7, 3, 1, 3, 7, 5, 2]
            line = [(float(corners[0, i]), float(corners[1, i])) for i in idx[:16]]
            color = (255, 80, 80) if "car" in ann["category_name"] else (80, 255, 80)
            if len(line) >= 2:
                draw.line(line, fill=color, width=2)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_annotation_bev_png(nusc: NuScenes, sample_token: str) -> bytes:
    """BEV matplotlib: all annotations in ego frame (same spirit as generate_video BEV)."""
    ego_pose = get_ego_pose(nusc, sample_token)
    sample = nusc.get("sample", sample_token)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.scatter([0], [0], c="red", s=80, marker="s", zorder=10, label="Ego")
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        box = Box(
            center=ann["translation"],
            size=ann["size"],
            orientation=Quaternion(ann["rotation"]),
        )
        bev_corners = project_box_to_bev(box, ego_pose)
        if is_box_in_range(bev_corners):
            poly = Polygon(bev_corners.T, facecolor="cyan", alpha=0.25, edgecolor="cyan", lw=1.5)
            ax.add_patch(poly)
    ax.set_title("3D annotations BEV (ego)")
    ax.legend(loc="upper right")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def list_camera_channels() -> list[str]:
    flat: list[str] = []
    for row in get_sensor_order():
        flat.extend(row)
    return flat
