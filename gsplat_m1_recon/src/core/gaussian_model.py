from __future__ import annotations

import torch


class GaussianModel(torch.nn.Module):
    """
    极简 3DGS 高斯参数模型（M1 友好版本）

    参数说明：
    - xyz: 每个高斯中心坐标，形状 (N, 3)
    - log_scales: 对数尺度，渲染时用 exp 变成正值
    - color_logits: 颜色 logits，经过 sigmoid 得到 0~1
    - opacity_logits: 不透明度 logits，经过 sigmoid 得到 0~1
    """

    def __init__(self, xyz: torch.Tensor, rgb: torch.Tensor):
        super().__init__()
        n = xyz.shape[0]
        self.xyz = torch.nn.Parameter(xyz.clone())
        self.log_scales = torch.nn.Parameter(torch.full((n, 3), -3.0, dtype=xyz.dtype, device=xyz.device))
        self.color_logits = torch.nn.Parameter(torch.logit(rgb.clamp(1e-4, 1 - 1e-4)))
        self.opacity_logits = torch.nn.Parameter(torch.full((n, 1), -1.2, dtype=xyz.dtype, device=xyz.device))

    @property
    def num_gaussians(self) -> int:
        return int(self.xyz.shape[0])

    def colors(self) -> torch.Tensor:
        return torch.sigmoid(self.color_logits)

    def opacities(self) -> torch.Tensor:
        return torch.sigmoid(self.opacity_logits).squeeze(-1)

    def scales(self) -> torch.Tensor:
        return torch.exp(self.log_scales)
