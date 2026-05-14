"""Server-side renders: LiDAR BEV, camera with 3D box projection (MVP)."""

from __future__ import annotations

import io
import zlib
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Matplotlib defaults lack CJK glyphs; titles would show as tofu / mojibake.
_CJK_SANS = [
    "PingFang SC",
    "Hiragino Sans GB",
    "Heiti SC",
    "Heiti TC",
    "Songti SC",
    "STHeiti",
    "Microsoft YaHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]


def _configure_matplotlib_cjk_fonts() -> None:
    prev = plt.rcParams.get("font.sans-serif")
    if isinstance(prev, str):
        merged = list(_CJK_SANS) + [prev]
    elif isinstance(prev, list):
        merged = list(_CJK_SANS) + [f for f in prev if f not in _CJK_SANS]
    else:
        merged = list(_CJK_SANS)
    plt.rcParams["font.sans-serif"] = merged
    plt.rcParams["axes.unicode_minus"] = False


_configure_matplotlib_cjk_fonts()
from matplotlib.patches import Circle, Polygon
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
from nuscenes.utils.geometry_utils import BoxVisibility, view_points


def _ann_has_some_returns(ann: dict) -> bool:
    """Skip annotations with explicit zero lidar+radar returns (often spurious)."""
    nl = ann.get("num_lidar_pts", -1)
    nr = ann.get("num_radar_pts", -1)
    if nl is None:
        nl = -1
    if nr is None:
        nr = -1
    if nl >= 0 and nr >= 0 and nl + nr == 0:
        return False
    return True


def _ann_size_reasonable(ann: dict, max_dim_m: float = 22.0, min_dim_m: float = 0.08) -> bool:
    w, l, h = ann["size"]
    mx = max(float(w), float(l), float(h))
    mn = min(float(w), float(l), float(h))
    return mn >= min_dim_m and mx <= max_dim_m


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
        if not _ann_has_some_returns(ann) or not _ann_size_reasonable(ann):
            continue
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


def _lidar_points_ego_xy_z(nusc: NuScenes, sample_token: str, max_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load LIDAR_TOP sweep, transform sensor frame -> ego; subsample with stable RNG per sample."""
    sample = nusc.get("sample", sample_token)
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    path = str((Path(nusc.dataroot) / lidar_sd["filename"]).resolve())
    pc = LidarPointCloud.from_file(path)
    cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    pc.rotate(Quaternion(cs["rotation"]).rotation_matrix)
    pc.translate(np.array(cs["translation"]))
    pts = pc.points[:3, :].T  # (N, 3) ego frame
    n = pts.shape[0]
    if n > max_points:
        seed = zlib.crc32(sample_token.encode("utf-8")) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=min(max_points, n), replace=False)
        pts = pts[idx]
    return pts[:, 0], pts[:, 1], pts[:, 2]


def render_lidar_bev_png(
    nusc: NuScenes,
    sample_token: str,
    *,
    draw_boxes: bool = False,
    max_points: int = 40_000,
) -> bytes:
    """Scatter BEV in ego frame; color = horizontal range from ego (0–100 m, stable); 100 m ref circle."""
    x, y, _ = _lidar_points_ego_xy_z(nusc, sample_token, max_points)
    ego_pose = get_ego_pose(nusc, sample_token)
    rng_xy = np.hypot(x, y)
    lim = 105.0
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    ax.scatter(
        x,
        y,
        s=0.25,
        c=rng_xy,
        cmap="bone",
        alpha=0.85,
        vmin=0.0,
        vmax=100.0,
        rasterized=True,
    )
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ref = Circle((0.0, 0.0), 100.0, fill=False, edgecolor="0.85", linewidth=1.0, linestyle=(0, (6, 4)), alpha=0.85)
    ax.add_patch(ref)
    inner = Circle((0.0, 0.0), 50.0, fill=False, edgecolor="0.55", linewidth=0.6, linestyle=(0, (4, 6)), alpha=0.5)
    ax.add_patch(inner)
    title_zh = "LiDAR BEV（ego）+ 3D 框" if draw_boxes else "LiDAR BEV（ego）"
    # Use ASCII hyphen in numeric range; CJK fonts configured at module load.
    ax.set_title(f"{title_zh} · 颜色=水平距离 0-100 m · 虚线圆 100 m / 50 m")
    ax.grid(True, alpha=0.25)
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
    vehicle_only: bool = False,
) -> bytes:
    """Camera image; optional projected 3D boxes (all categories by default)."""
    img_path = get_sample_data_path(nusc, sample_token, channel)
    img = Image.open(img_path).convert("RGB")
    if not draw_boxes:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    draw = ImageDraw.Draw(img)
    sample = nusc.get("sample", sample_token)
    cam_sd_token = sample["data"][channel]
    # Official path: ego pose of *this* camera keyframe + get_boxes + box_in_image
    # (LIDAR_TOP ego_pose can differ slightly from CAM_* and misaligns projections).
    _, boxes_cam, cam_intrinsic_mx = nusc.get_sample_data(
        cam_sd_token,
        box_vis_level=BoxVisibility.ANY,
    )
    cam_intrinsic = np.asarray(cam_intrinsic_mx, dtype=np.float64).reshape(3, 3)

    for box in boxes_cam:
        cat = getattr(box, "name", "") or ""
        if vehicle_only and "vehicle" not in cat:
            continue
        corners = view_points(box.corners(), cam_intrinsic, normalize=True)
        if not np.isfinite(corners).all():
            continue
        ci = corners[:2, :]  # 2 x 8
        color = (255, 80, 80) if "car" in cat else (80, 255, 80)
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
    ax.set_title("3D 标注 BEV（ego）")
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
