"""卷积版 VAE（Variational Autoencoder）实现。

包含三个核心组件：
1) Encoder: 输入图像 -> 潜变量分布参数 (mu, logvar)；
2) Decoder: 潜变量 z -> 重构图像；
3) VAE: 封装重参数化采样与完整前向流程。
"""

# 启用延迟类型注解支持。
from __future__ import annotations

# 类型工具：List/Sequence/Tuple。
from typing import List, Sequence, Tuple

# PyTorch 主库。
import torch
# Tensor 与神经网络模块别名。
from torch import Tensor, nn
# 函数式接口，这里用于尺寸对齐插值。
from torch.nn import functional as F


class Encoder(nn.Module):
    """卷积编码器：输出潜变量分布参数。"""

    def __init__(
        self,
        in_channels: int,
        hidden_dims: Sequence[int],
        latent_dim: int,
        image_size: int,
    ) -> None:
        # 初始化 nn.Module 父类。
        super().__init__()
        # 用于按层构建 encoder 网络结构。
        modules: List[nn.Module] = []
        # 当前特征图通道数，初始为输入通道数。
        current_channels = in_channels

        # 逐层堆叠 Conv2d + BN + ReLU。
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        current_channels,
                        h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU(inplace=True),
                )
            )
            # 更新下一层的输入通道数。
            current_channels = h_dim

        # 把子模块列表打包为顺序网络。
        self.conv = nn.Sequential(*modules)

        # 用一个 dummy 输入自动推断卷积输出形状，避免手写尺寸计算错误。
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size, image_size)
            conv_out = self.conv(dummy)
            # 记录卷积输出的 C/H/W，供 decoder 反卷积还原时使用。
            self.feature_shape = conv_out.shape[1:]
            # 展平后的总维度（1 个样本）。
            flattened_dim = int(conv_out.numel())

        # 保存展平维度。
        self.flattened_dim = flattened_dim
        # 线性层输出潜变量均值 mu。
        self.fc_mu = nn.Linear(self.flattened_dim, latent_dim)
        # 线性层输出潜变量 logvar。
        self.fc_logvar = nn.Linear(self.flattened_dim, latent_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """前向传播：输入图像 -> (mu, logvar)。"""
        # 卷积提取特征。
        features = self.conv(x)
        # 展平为二维张量 [B, D]。
        flattened = torch.flatten(features, start_dim=1)
        # 预测均值与对数方差。
        mu = self.fc_mu(flattened)
        logvar = self.fc_logvar(flattened)
        # 返回参数化后的高斯分布参数。
        return mu, logvar


class Decoder(nn.Module):
    """反卷积解码器：潜变量 -> 重构图像。"""

    def __init__(
        self,
        out_channels: int,
        hidden_dims: Sequence[int],
        latent_dim: int,
        feature_shape: Sequence[int],
    ) -> None:
        # 初始化父类。
        super().__init__()
        # feature_shape 来自编码器卷积输出形状，形如 (C, H, W)。
        self.feature_shape = tuple(int(v) for v in feature_shape)
        # 计算展平维度，供全连接层使用。
        flattened_dim = (
            self.feature_shape[0] * self.feature_shape[1] * self.feature_shape[2]
        )

        # 先把 z 投影回卷积特征平面。
        self.decoder_input = nn.Linear(latent_dim, flattened_dim)

        # 解码阶段通道顺序与编码器相反（镜像结构）。
        reversed_hidden_dims = list(hidden_dims)[::-1]
        modules: List[nn.Module] = []
        # 构建多层 ConvTranspose2d + BN + ReLU。
        for i in range(len(reversed_hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        reversed_hidden_dims[i],
                        reversed_hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(reversed_hidden_dims[i + 1]),
                    nn.ReLU(inplace=True),
                )
            )
        # 顺序打包。
        self.deconv = nn.Sequential(*modules)

        # 最后一层输出图像通道数，并用 Sigmoid 约束到 [0, 1]。
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                reversed_hidden_dims[-1],
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, z: Tensor) -> Tensor:
        """前向传播：输入潜变量 z，输出重构图。"""
        # 线性映射到卷积特征维度。
        result = self.decoder_input(z)
        # 从 [B, D] reshape 回 [B, C, H, W]。
        result = result.view(
            -1,
            self.feature_shape[0],
            self.feature_shape[1],
            self.feature_shape[2],
        )
        # 逐层反卷积上采样。
        result = self.deconv(result)
        # 最终映射到图像空间。
        result = self.final_layer(result)
        return result


class VAE(nn.Module):
    """VAE 总装类：封装编码、采样、解码全流程。"""

    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        hidden_dims: Sequence[int],
        image_size: int,
    ) -> None:
        # 初始化父类。
        super().__init__()
        # 记录潜变量维度，供采样函数使用。
        self.latent_dim = latent_dim
        # 记录目标图像尺寸。
        self.image_size = image_size
        # 构建编码器。
        self.encoder = Encoder(in_channels, hidden_dims, latent_dim, image_size)
        # 构建解码器（依赖 encoder 自动推断的 feature_shape）。
        self.decoder = Decoder(
            in_channels,
            hidden_dims,
            latent_dim,
            self.encoder.feature_shape,
        )

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """编码：x -> (mu, logvar)。"""
        return self.encoder(x)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """重参数化采样：z = mu + std * eps。"""
        # 由 logvar 得到标准差 std。
        std = torch.exp(0.5 * logvar)
        # 采样标准高斯噪声 eps。
        eps = torch.randn_like(std)
        # 构造可导采样结果 z。
        return eps * std + mu

    def decode(self, z: Tensor) -> Tensor:
        """解码：z -> reconstruction。"""
        # 先走解码器主干。
        reconstruction = self.decoder(z)
        # 目标图像尺寸（例如 MNIST 为 28x28）。
        target_size = (self.image_size, self.image_size)
        # 若反卷积输出尺寸与目标不一致，做一次安全对齐。
        if reconstruction.shape[-2:] != target_size:
            # 使用双线性插值对齐 H/W，避免 BCE 输入输出尺寸不一致报错。
            reconstruction = F.interpolate(
                reconstruction,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        # 返回对齐后的重构结果。
        return reconstruction

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """完整前向：x -> (reconstruction, mu, logvar, z)。"""
        # 编码阶段。
        mu, logvar = self.encode(x)
        # 重参数化采样。
        z = self.reparameterize(mu, logvar)
        # 解码阶段。
        reconstruction = self.decode(z)
        # 返回训练计算 ELBO 所需全部变量。
        return reconstruction, mu, logvar, z

    def sample(self, num_samples: int, device: torch.device) -> Tensor:
        """随机生成：采样 z~N(0,I) 后解码得到新图像。"""
        # 从标准高斯采样潜变量向量。
        z = torch.randn(num_samples, self.latent_dim, device=device)
        # 解码得到生成图像。
        return self.decode(z)
