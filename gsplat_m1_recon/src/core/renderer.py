from __future__ import annotations

import torch

from .gaussian_model import GaussianModel


def resize_intrinsics(fx: float, fy: float, cx: float, cy: float, src_w: int, src_h: int, dst_w: int, dst_h: int):
    sx = dst_w / float(src_w)
    sy = dst_h / float(src_h)
    return fx * sx, fy * sy, cx * sx, cy * sy


def render_view(
    model: GaussianModel,
    w2c: torch.Tensor,
    intrinsics: tuple[float, float, float, float],
    image_size: tuple[int, int],
    max_visible_splats: int = 5000,
    radius_px_max: float = 10.0,
    bg_color: float = 1.0,
) -> torch.Tensor:
    """
    简化可微 splatting 渲染器（适合 M1 / MPS）。
    - 按深度排序并进行 alpha 合成
    - 使用小窗口 patch splat，避免全图逐高斯开销
    """
    device = model.xyz.device
    h, w = image_size
    fx, fy, cx, cy = intrinsics

    xyz_h = torch.cat([model.xyz, torch.ones((model.num_gaussians, 1), device=device)], dim=1)
    cam = (w2c @ xyz_h.t()).t()
    z = cam[:, 2]
    valid = z > 1e-3
    if valid.sum() == 0:
        return torch.ones((h, w, 3), device=device) * bg_color

    cam = cam[valid]
    z = cam[:, 2]
    idx_valid = torch.nonzero(valid, as_tuple=False).squeeze(1)
    u = fx * cam[:, 0] / z + cx
    v = fy * cam[:, 1] / z + cy

    in_view = (u > -16) & (u < w + 16) & (v > -16) & (v < h + 16)
    if in_view.sum() == 0:
        return torch.ones((h, w, 3), device=device) * bg_color

    u = u[in_view]
    v = v[in_view]
    z = z[in_view]
    keep_ids = idx_valid[in_view]

    if keep_ids.numel() > max_visible_splats:
        sel = torch.argsort(z)[:max_visible_splats]
        u, v, z, keep_ids = u[sel], v[sel], z[sel], keep_ids[sel]

    colors = model.colors()[keep_ids]
    opacity = model.opacities()[keep_ids]
    scales = model.scales()[keep_ids].mean(dim=1)

    radius = (fx * scales / (z + 1e-6)).clamp(1.0, radius_px_max)
    order = torch.argsort(z)  # 近到远，前向合成
    u, v, radius, colors, opacity = u[order], v[order], radius[order], colors[order], opacity[order]

    image = torch.ones((h, w, 3), device=device) * bg_color
    alpha_acc = torch.zeros((h, w), device=device)

    for i in range(u.shape[0]):
        cx_i = u[i]
        cy_i = v[i]
        r = radius[i]
        x0 = torch.clamp(torch.floor(cx_i - 3 * r).long(), 0, w - 1)
        x1 = torch.clamp(torch.ceil(cx_i + 3 * r).long(), 0, w - 1)
        y0 = torch.clamp(torch.floor(cy_i - 3 * r).long(), 0, h - 1)
        y1 = torch.clamp(torch.ceil(cy_i + 3 * r).long(), 0, h - 1)
        if (x1 <= x0) or (y1 <= y0):
            continue

        xs = torch.arange(x0, x1 + 1, device=device, dtype=torch.float32)
        ys = torch.arange(y0, y1 + 1, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        dx2 = (xx - cx_i) ** 2
        dy2 = (yy - cy_i) ** 2
        g = torch.exp(-(dx2 + dy2) / (2 * (r ** 2) + 1e-6))
        a = (opacity[i] * g).clamp(0.0, 0.95)

        patch_alpha = alpha_acc[y0 : y1 + 1, x0 : x1 + 1]
        wgt = (1.0 - patch_alpha) * a
        image[y0 : y1 + 1, x0 : x1 + 1, :] = image[y0 : y1 + 1, x0 : x1 + 1, :] * (1.0 - wgt.unsqueeze(-1)) + wgt.unsqueeze(-1) * colors[i]
        alpha_acc[y0 : y1 + 1, x0 : x1 + 1] = (patch_alpha + wgt).clamp(0.0, 1.0)

    return image.clamp(0.0, 1.0)
