"""可视化工具模块。

用于导出 VAE 常见实验图：
1) 重构图（原图 vs 重构）；
2) 随机生成图；
3) 潜空间 t-SNE 图；
4) 训练损失曲线图。
"""

# 启用延迟类型注解支持。
from __future__ import annotations

# 路径处理。
from pathlib import Path
# 类型标注工具。
from typing import Any, List, Sequence

# 绘图库。
import matplotlib.pyplot as plt
# 数值计算库。
import numpy as np
# PyTorch 主库。
import torch
# sklearn 的 t-SNE 实现。
from sklearn.manifold import TSNE
# Tensor 与 nn 类型。
from torch import Tensor, nn
# DataLoader 类型。
from torch.utils.data import DataLoader
# torchvision 网格拼图工具。
from torchvision.utils import make_grid

# 引入指标数据结构，用于 loss 曲线绘制。
from src.utils.metrics import EpochMetric


def _to_numpy_image(image_tensor: Tensor) -> np.ndarray:
    """将张量图像转为 NumPy 图像，供 matplotlib 显示。"""
    # detach: 脱离计算图；cpu: 放到 CPU；clamp: 限制像素在 [0,1]。
    image = image_tensor.detach().cpu().clamp(0, 1)
    # 单通道图像返回 HxW。
    if image.size(0) == 1:
        return image.squeeze(0).numpy()
    # 多通道图像从 CHW 转换为 HWC。
    return np.transpose(image.numpy(), (1, 2, 0))


def save_reconstruction_grid(
    model: nn.Module,
    batch_images: Tensor,
    device: torch.device,
    output_path: Path,
    num_images: int = 8,
) -> None:
    """保存原图与重构图对比网格。"""
    # 评估模式。
    model.eval()
    # 禁用梯度以提升推理效率。
    with torch.no_grad():
        # 从 batch 中取前 num_images 张做可视化。
        input_images = batch_images[:num_images].to(device)
        # 执行重构。
        recon_images, _, _, _ = model(input_images)

    # 按“原图在上、重构在下”拼接。
    pair_grid = torch.cat([input_images.cpu(), recon_images.cpu()], dim=0)
    # 生成网格图。
    grid_image = make_grid(pair_grid, nrow=num_images, padding=2)
    # 张量转 NumPy。
    np_image = _to_numpy_image(grid_image)

    # 绘图并保存。
    plt.figure(figsize=(14, 4))
    if np_image.ndim == 2:
        plt.imshow(np_image, cmap="gray")
    else:
        plt.imshow(np_image)
    plt.axis("off")
    plt.title("Top: Original | Bottom: Reconstruction")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_generation_grid(
    model: nn.Module,
    latent_dim: int,
    device: torch.device,
    output_path: Path,
    num_samples: int = 64,
) -> None:
    """保存随机生成网格图。"""
    # 评估模式。
    model.eval()
    with torch.no_grad():
        # 从标准高斯采样潜变量 z。
        z = torch.randn(num_samples, latent_dim, device=device)
        # 解码得到生成样本。
        generated = model.decode(z).cpu()

    # 生成网格图并转 NumPy。
    grid_image = make_grid(generated, nrow=8, padding=2)
    np_image = _to_numpy_image(grid_image)

    # 绘图保存。
    plt.figure(figsize=(8, 8))
    if np_image.ndim == 2:
        plt.imshow(np_image, cmap="gray")
    else:
        plt.imshow(np_image)
    plt.axis("off")
    plt.title("Random Generated Samples")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _extract_label(raw_label: Any) -> int:
    """统一标签类型为 int。"""
    # 已经是 Python int，直接返回。
    if isinstance(raw_label, int):
        return raw_label
    # 若是 Tensor，需要处理 0 维或高维情况。
    if torch.is_tensor(raw_label):
        if raw_label.ndim == 0:
            return int(raw_label.item())
        return int(raw_label.flatten()[0].item())
    # 兜底强转。
    return int(raw_label)


def save_latent_tsne(
    model: nn.Module,
    dataloader: DataLoader[Any],
    device: torch.device,
    output_path: Path,
    max_points: int = 2000,
) -> None:
    """将潜变量均值 mu 用 t-SNE 降到 2D 并保存散点图。"""
    # 评估模式。
    model.eval()
    # 存储潜变量和标签。
    latents: List[np.ndarray] = []
    labels: List[int] = []
    # 已收集样本数计数器（用于截断）。
    collected = 0

    # 关闭梯度，提取潜变量特征。
    with torch.no_grad():
        for images, targets in dataloader:
            # 图像送入设备。
            images = images.to(device)
            # 只取 mu 作为潜空间表示。
            mu, _ = model.encode(images)
            mu_np = mu.detach().cpu().numpy()

            # 逐样本收集，直到达到 max_points。
            for idx in range(mu_np.shape[0]):
                if collected >= max_points:
                    break
                latents.append(mu_np[idx])
                labels.append(_extract_label(targets[idx]))
                collected += 1

            if collected >= max_points:
                break

    # t-SNE 至少需要两个样本点。
    if len(latents) < 2:
        return

    # 执行 t-SNE 降维。
    tsne = TSNE(n_components=2, random_state=42, init="pca")
    embedding = tsne.fit_transform(np.asarray(latents))

    # 绘制 2D 散点图，颜色对应类别标签。
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=np.asarray(labels),
        cmap="tab10",
        s=8,
        alpha=0.8,
    )
    plt.title("2D Latent Space (t-SNE)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.colorbar(scatter)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_loss_curve(metrics: Sequence[EpochMetric], output_path: Path) -> None:
    """根据指标记录绘制 train/val loss 曲线。"""
    # 训练集 x/y 序列。
    train_epochs: List[int] = []
    train_losses: List[float] = []
    # 验证集 x/y 序列。
    val_epochs: List[int] = []
    val_losses: List[float] = []

    # 从指标记录中拆分 train 与 val 两条曲线。
    for metric in metrics:
        if metric.split == "train":
            train_epochs.append(metric.epoch)
            train_losses.append(metric.loss)
        elif metric.split == "val":
            val_epochs.append(metric.epoch)
            val_losses.append(metric.loss)

    # 没有训练数据时不绘图。
    if not train_epochs:
        return

    # 绘制曲线并保存。
    plt.figure(figsize=(8, 5))
    plt.plot(train_epochs, train_losses, label="Train Loss")
    if val_epochs:
        plt.plot(val_epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VAE Training Curves")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
