"""训练/评测用的可视化：原图 vs 渲染 vs 误差图、新视角渲染（无 GT 对照）。"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from core.gaussian_model import GaussianModel
from core.renderer import render_view, resize_intrinsics
from train import TrainFrame, load_target_image
from utils.camera_pose import compose_w2c_world_yaw, numpy_w2c_to_torch


def _to_uint8_rgb(x: torch.Tensor) -> np.ndarray:
    return (x.detach().clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)


def _ssim_map(p: torch.Tensor, t: torch.Tensor, win: int = 11) -> torch.Tensor:
    """
    局部 SSIM 图（RGB 三通道各自滑窗后取平均），值域约 [0,1]，形状 (H, W)。
    p, t: (H, W, 3), 0~1
    """
    c1 = 0.01**2
    c2 = 0.03**2
    device, dtype = p.device, p.dtype
    ch = 3
    hw = torch.arange(win, device=device, dtype=dtype) - win // 2
    g = torch.exp(-(hw**2) / (2 * 1.5 * 1.5))
    g = g / (g.sum() + 1e-12)
    w2d = (g[:, None] @ g[None, :]).view(1, 1, win, win).expand(ch, 1, win, win).contiguous()
    p2 = p.permute(2, 0, 1).unsqueeze(0)
    t2 = t.permute(2, 0, 1).unsqueeze(0)
    pad = win // 2
    mu_p = F.conv2d(p2, w2d, padding=pad, groups=ch)
    mu_t = F.conv2d(t2, w2d, padding=pad, groups=ch)
    mu_pp = mu_p * mu_p
    mu_tt = mu_t * mu_t
    mu_pt = mu_p * mu_t
    sigma_p = F.conv2d(p2 * p2, w2d, padding=pad, groups=ch) - mu_pp
    sigma_t = F.conv2d(t2 * t2, w2d, padding=pad, groups=ch) - mu_tt
    sigma_pt = F.conv2d(p2 * t2, w2d, padding=pad, groups=ch) - mu_pt
    s = ((2 * mu_pt + c1) * (2 * sigma_pt + c2)) / ((mu_pp + mu_tt + c1) * (sigma_p + sigma_t + c2) + 1e-12)
    return s.clamp(0.0, 1.0).mean(dim=1).squeeze(0)


def save_triple_panel(gt: torch.Tensor, pred: torch.Tensor, out_path: Path, title: str = "") -> None:
    """
    横向 5 列 + 颜色条：
    GT | Pred | 平均绝对误差×8（灰度）| 误差纯热力图 + colorbar | SSIM 图 + colorbar
    """
    gt_np = gt.detach().clamp(0, 1).cpu().numpy()
    pr_np = pred.detach().clamp(0, 1).cpu().numpy()
    err = (pred - gt).abs().mean(dim=-1).detach().cpu().numpy()
    err_x8 = np.clip(err * 8.0, 0.0, 1.0)

    ssim_t = _ssim_map(pred, gt)
    ssim_np = ssim_t.detach().cpu().numpy()

    h, w = err.shape
    fig_w = max(14.0, w / 80.0 * 5.0)
    fig_h = max(3.2, h / 80.0 + 0.8)
    fig, axes = plt.subplots(1, 5, figsize=(fig_w, fig_h), constrained_layout=False)
    fig.subplots_adjust(left=0.02, right=0.94, top=0.88, bottom=0.06, wspace=0.35)

    axes[0].imshow(gt_np)
    axes[0].set_title("GT", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(pr_np)
    axes[1].set_title("Pred", fontsize=9)
    axes[1].axis("off")

    axes[2].imshow(err_x8, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("|err| mean RGB ×8", fontsize=9)
    axes[2].axis("off")

    vmax_err = float(np.percentile(err, 99.0)) if err.size else 1e-6
    vmax_err = max(vmax_err, 1e-6)
    im4 = axes[3].imshow(err, cmap="turbo", vmin=0.0, vmax=vmax_err)
    axes[3].set_title("|err| heatmap", fontsize=9)
    axes[3].axis("off")
    plt.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.02)

    im5 = axes[4].imshow(ssim_np, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[4].set_title("SSIM map", fontsize=9)
    axes[4].axis("off")
    plt.colorbar(im5, ax=axes[4], fraction=0.046, pad=0.02)

    fig.suptitle(title[:200] if title else "GT | Pred | |err|×8 | err heatmap | SSIM", fontsize=10, y=0.98)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=140, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def render_pair_for_frame(
    model: GaussianModel,
    frm: TrainFrame,
    render_h: int,
    device: torch.device,
    max_visible_splats: int,
    radius_px_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    aspect = frm.width / frm.height
    render_w = int(render_h * aspect)
    size_hw = (render_h, render_w)
    fx, fy, cx, cy = resize_intrinsics(*frm.intrinsics, src_w=frm.width, src_h=frm.height, dst_w=render_w, dst_h=render_h)
    target = load_target_image(frm.image_path, size_hw=size_hw, device=device)
    w2c = numpy_w2c_to_torch(frm.w2c, device)
    pred = render_view(
        model=model,
        w2c=w2c,
        intrinsics=(fx, fy, cx, cy),
        image_size=size_hw,
        max_visible_splats=max_visible_splats,
        radius_px_max=radius_px_max,
        bg_color=1.0,
    )
    return pred, target


@torch.no_grad()
def export_same_view_eval_grid(
    model: GaussianModel,
    frames: list[TrainFrame],
    *,
    out_dir: Path,
    device: torch.device,
    render_h: int,
    max_visible_splats: int,
    radius_px_max: float,
    max_panels: int = 6,
) -> None:
    """多张训练相机：五列对比图（GT|Pred|误差×8|热力图|SSIM），便于肉眼看差别。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(frames)
    if n == 0:
        return
    step = max(1, n // max_panels)
    picked = list(range(0, n, step))[:max_panels]
    for j, idx in enumerate(picked):
        frm = frames[idx]
        pred, target = render_pair_for_frame(model, frm, render_h, device, max_visible_splats, radius_px_max)
        name = frm.image_path.stem
        save_triple_panel(
            target,
            pred,
            out_dir / f"eval_sameview_{j:02d}_{name}.jpg",
            title=f"same view {name} | GT | Pred | |err|×8 | err heatmap | SSIM",
        )


@torch.no_grad()
def export_novel_yaw_views(
    model: GaussianModel,
    base_frm: TrainFrame,
    *,
    out_dir: Path,
    device: torch.device,
    render_h: int,
    max_visible_splats: int,
    radius_px_max: float,
    yaw_degrees: list[float],
) -> None:
    """
    以某一训练帧为基准，在世界系绕 Y 轴旋转相机（等价于绕物体环视），
    只有渲染图（无对应真值），用于看几何是否连贯。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    aspect = base_frm.width / base_frm.height
    render_w = int(render_h * aspect)
    size_hw = (render_h, render_w)
    fx, fy, cx, cy = resize_intrinsics(
        *base_frm.intrinsics, src_w=base_frm.width, src_h=base_frm.height, dst_w=render_w, dst_h=render_h
    )
    base_w2c = numpy_w2c_to_torch(base_frm.w2c, device)
    for yaw in yaw_degrees:
        w2c = compose_w2c_world_yaw(base_w2c, yaw)
        pred = render_view(
            model=model,
            w2c=w2c,
            intrinsics=(fx, fy, cx, cy),
            image_size=size_hw,
            max_visible_splats=max_visible_splats,
            radius_px_max=radius_px_max,
            bg_color=1.0,
        )
        img = _to_uint8_rgb(pred)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / f"eval_novel_yaw_{yaw:+04.0f}deg.jpg"), img_bgr)
