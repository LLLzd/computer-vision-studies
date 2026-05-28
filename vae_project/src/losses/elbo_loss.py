"""VAE 的 ELBO 损失实现。

ELBO（Evidence Lower BOund）由两部分组成：
1) 重构损失（Reconstruction Loss）：希望重构图像尽量接近输入图像；
2) KL 散度（KL Divergence）：希望编码得到的后验分布接近标准高斯先验。
"""

# 启用延迟类型注解支持。
from __future__ import annotations

# dataclass 用于声明轻量数据结构，提升可读性。
from dataclasses import dataclass

# PyTorch 主库。
import torch
# Tensor 类型别名，便于类型标注。
from torch import Tensor
# 函数式 API，包含 BCE 等损失函数。
from torch.nn import functional as F


@dataclass
class ELBOLossOutput:
    """结构化返回 ELBO 三项指标。"""

    # 总损失：recon + beta * kl
    total: Tensor
    # 重构损失（通常是 BCE）。
    recon: Tensor
    # KL 散度损失。
    kl: Tensor


def elbo_loss(
    reconstruction: Tensor,
    target: Tensor,
    mu: Tensor,
    logvar: Tensor,
    beta: float = 1.0,
) -> ELBOLossOutput:
    """计算 ELBO 损失。

    参数:
        reconstruction: 解码器输出，形状通常为 [B, C, H, W]。
        target: 输入真值图像，形状与 reconstruction 相同。
        mu: 编码器输出的均值向量。
        logvar: 编码器输出的对数方差向量。
        beta: KL 项权重（beta-VAE 中可调）。
    """
    # batch_size 用于把 sum-reduction 损失规范化为“每样本平均”。
    batch_size = target.size(0)
    # BCE 使用 sum 再除以 batch_size，是 VAE 经典写法。
    recon_loss = F.binary_cross_entropy(
        reconstruction,
        target,
        reduction="sum",
    ) / batch_size
    # KL(q(z|x) || p(z))，其中 p(z)=N(0,I) 时有闭式解。
    kl_loss = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp(),
    ) / batch_size
    # 总损失。
    total_loss = recon_loss + beta * kl_loss
    # 返回结构化结果，便于 trainer 分别记录 total/recon/kl。
    return ELBOLossOutput(total=total_loss, recon=recon_loss, kl=kl_loss)
