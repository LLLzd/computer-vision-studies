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
from nuscenes.utils.data_classes import Box, LidarPointCloud
from nuscenes.utils.geometry_utils import view_points


def _draw_bev_boxes_on_axes(
    ax: plt.Axes,
    nusc: NuScenes,
    sample_token: str,
    ego_pose: dict,
    *,
    face_alpha: float = 0.25,
) -> None:
    """Draw all sample annotations as BEV polygons in ego frame."""
    sample = nusc.get("sample", sample_token)
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        box = Box(
            center=ann["translation"],
            size=ann["size"],
            orientation=Quaternion(ann["rotation"]),
        )
        bev_corners = project_box_to_bev(box, ego_pose)
        if is_box_in_range(bev_corners):
            poly = Polygon(
                bev_corners.T,
                facecolor="cyan",
                alpha=face_alpha,
                edgecolor="cyan",
                lw=1.2,
            )
            ax.add_patch(poly)


def _lidar_points_ego_xy_z(nusc: NuScenes, sample_token: str, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Load LIDAR_TOP sweep, transform sensor frame -> ego (same as devkit map_pointcloud_to_image step 1)."""
    sample = nusc.get("sample", sample_token)
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    path = str((Path(nusc.dataroot) / lidar_sd["filename"]).resolve())
    pc = LidarPointCloud.from_file(path)
    cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    pc.rotate(Quaternion(cs["rotation"]).rotation_matrix)
    pc.translate(np.array(cs["translation"]))
    pts = pc.points[:3, :].T  # (N, 3) ego frame
    if pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx]
    return pts[:, 0], pts[:, 1], pts[:, 2]


def render_lidar_bev_png(
    nusc: NuScenes,
    sample_token: str,
    *,
    draw_boxes: bool = False,
    max_points: int = 40_000,
) -> bytes:
    """Scatter BEV (x, y) from LIDAR_TOP in ego frame; optional 3D box overlay (ego), matching nuScenes."""
    x, y, z = _lidar_points_ego_xy_z(nusc, sample_token, max_points)
    ego_pose = get_ego_pose(nusc, sample_token)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    ax.scatter(x, y, s=0.2, c=z, cmap="viridis", alpha=0.8)
    ax.set_aspect("equal")
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    title = "LiDAR BEV + 3D boxes (ego)" if draw_boxes else "LiDAR BEV (ego frame)"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if draw_boxes:
        _draw_bev_boxes_on_axes(ax, nusc, sample_token, ego_pose, face_alpha=0.35)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def render_camera_with_boxes_png(
    nusc: NuScenes,
    sample_token: str,
    channel: str,
    *,
    draw_boxes: bool = True,
) -> bytes:
    """Camera image; optional projected 3D boxes (vehicles only when draw_boxes)."""
    img_path = get_sample_data_path(nusc, sample_token, channel)
    img = Image.open(img_path).convert("RGB")
    if not draw_boxes:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

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
            ci = corners[:2, :]  # 2 x 8
            color = (255, 80, 80) if "car" in ann["category_name"] else (80, 255, 80)
            w = 2
            for i in range(4):
                draw.line(
                    [(float(ci[0, i]), float(ci[1, i])), (float(ci[0, i + 4]), float(ci[1, i + 4]))],
                    fill=color,
                    width=w,
                )
            for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
                draw.line(
                    [(float(ci[0, a]), float(ci[1, a])), (float(ci[0, b]), float(ci[1, b]))],
                    fill=color,
                    width=w,
                )
            for a, b in ((4, 5), (5, 6), (6, 7), (7, 4)):
                draw.line(
                    [(float(ci[0, a]), float(ci[1, a])), (float(ci[0, b]), float(ci[1, b]))],
                    fill=color,
                    width=w,
                )

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_annotation_bev_png(nusc: NuScenes, sample_token: str) -> bytes:
    """BEV matplotlib: all annotations in ego frame (same spirit as generate_video BEV)."""
    ego_pose = get_ego_pose(nusc, sample_token)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.scatter([0], [0], c="red", s=80, marker="s", zorder=10, label="Ego")
    _draw_bev_boxes_on_axes(ax, nusc, sample_token, ego_pose, face_alpha=0.25)
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
