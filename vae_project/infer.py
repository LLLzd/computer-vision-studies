"""VAE 推理脚本：重构与随机生成。

该脚本提供两类“训练后查看效果”的能力：
1) 重构：输入一张真实图片，输出其重构结果；
2) 生成：随机采样潜变量 z，再通过解码器生成新图片。
"""

# 启用延迟类型注解，改善类型提示兼容性。
from __future__ import annotations

# 命令行参数解析。
import argparse
# 路径工具。
from pathlib import Path

# 用于绘图保存推理结果。
import matplotlib.pyplot as plt
# PyTorch 核心库。
import torch

# 数据加载函数（用于取一批验证样本做重构演示）。
from src.datasets.mnist_dataset import build_dataloaders
# VAE 模型定义。
from src.models.vae import VAE
# 配置与输出目录管理工具。
from src.utils.config_utils import ensure_output_dirs, load_config
# 设备选择工具。
from src.utils.device_utils import get_device
# 日志系统初始化。
from src.utils.logger import setup_logger
# 随机种子固定。
from src.utils.seed import set_seed
# 复用已有“随机生成网格图”工具函数。
from src.utils.visualizer import save_generation_grid


def parse_args() -> argparse.Namespace:
    """解析推理脚本命令行参数。"""
    # 创建参数解析器。
    parser = argparse.ArgumentParser(description="VAE inference script.")
    # 配置文件路径参数。
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file.",
    )
    # 模型权重路径参数。
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/best.pt",
        help="Checkpoint path. Fallback to outputs/checkpoints/vae_final.pt if missing.",
    )
    # 指定从验证 batch 中取第几张图做重构示例。
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Sample index inside one validation batch.",
    )
    # 指定随机生成图片数量。
    parser.add_argument(
        "--num-generate",
        type=int,
        default=64,
        help="Number of random samples to generate.",
    )
    # 返回解析结果。
    return parser.parse_args()


def _load_model(
    config: dict,
    checkpoint_path: Path,
    device: torch.device,
) -> VAE:
    """创建模型并加载 checkpoint 权重。"""
    # 先按配置构建与训练期一致的模型结构。
    model = VAE(
        in_channels=int(config["model"]["in_channels"]),
        latent_dim=int(config["model"]["latent_dim"]),
        hidden_dims=list(config["model"]["hidden_dims"]),
        image_size=int(config["dataset"]["image_size"]),
    )

    # 若默认 checkpoint 不存在，尝试回退到 quick 训练保存的 vae_final.pt。
    if not checkpoint_path.exists():
        fallback = checkpoint_path.parent / "vae_final.pt"
        if fallback.exists():
            checkpoint_path = fallback
        else:
            # 两个候选文件都不存在时，抛出清晰错误。
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path} or {fallback}",
            )

    # 加载 checkpoint（map_location 确保可在任意设备读取）。
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # 只恢复模型参数，不恢复优化器（推理阶段不需要）。
    model.load_state_dict(checkpoint["model_state_dict"])
    # 把模型移动到目标设备（cpu/cuda/mps）。
    model.to(device)
    # 设置评估模式，关闭 Dropout / 使用 BN 的推理统计行为。
    model.eval()
    return model


def main() -> None:
    """执行推理流程：输出重构图和随机生成图。"""
    # 1) 读取命令行参数。
    args = parse_args()
    # 2) 获取项目根目录。
    project_root = Path(__file__).resolve().parent
    # 3) 加载 YAML 配置。
    config = load_config(project_root / args.config)
    # 4) 把相对路径转绝对路径，避免 cwd 引起的问题。
    config["project"]["data_dir"] = str(project_root / config["project"]["data_dir"])
    config["project"]["output_dir"] = str(project_root / config["project"]["output_dir"])

    # 5) 固定随机种子，保证推理结果可重复（特别是随机生成功能）。
    set_seed(int(config["project"]["seed"]))
    # 6) 创建输出目录结构。
    output_dirs = ensure_output_dirs(config, project_root=project_root)
    # 7) 初始化日志器。
    logger = setup_logger("vae_infer", output_dirs["logs_dir"] / "infer.log")
    # 8) 自动选择可用设备。
    device = get_device(config)
    logger.info("Using device: %s", device)

    # 9) 构造 checkpoint 绝对路径并加载模型。
    checkpoint_path = project_root / args.checkpoint
    model = _load_model(config=config, checkpoint_path=checkpoint_path, device=device)

    # 10) 构建验证集加载器，取一个 batch 做重构可视化。
    infer_batch_size = max(32, int(config.get("quick", {}).get("batch_size", 128)))
    _, val_loader = build_dataloaders(config=config, batch_size=infer_batch_size)
    # 取第一批数据：images 为图像，labels 为标签。
    images, labels = next(iter(val_loader))
    # 使用取模避免 sample-index 越界。
    sample_index = int(args.sample_index) % images.size(0)
    # 取单张图（保留 batch 维度，形状为 [1, C, H, W]）。
    sample = images[sample_index : sample_index + 1].to(device)
    # 为标题展示标签（兼容张量或普通整数）。
    label_value = int(labels[sample_index]) if torch.is_tensor(labels[sample_index]) else int(labels[sample_index])

    # 11) 关闭梯度，执行前向推理（更省显存/更快）。
    with torch.no_grad():
        reconstruction, _, _, _ = model(sample)

    # 12) 张量 -> NumPy，供 matplotlib 绘图。
    original_np = sample.squeeze(0).detach().cpu().squeeze(0).numpy()
    recon_np = reconstruction.squeeze(0).detach().cpu().squeeze(0).numpy()

    # 13) 保存“原图 vs 重构图”对比图。
    infer_output = output_dirs["visuals_dir"] / "infer_reconstruction.png"
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(original_np, cmap="gray")
    plt.title(f"Original (label={label_value})")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(recon_np, cmap="gray")
    plt.title("Reconstruction")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(infer_output, dpi=200)
    plt.close()

    # 14) 保存“随机生成网格图”（z ~ N(0, I)）。
    generation_output = output_dirs["visuals_dir"] / "infer_generation.png"
    save_generation_grid(
        model=model,
        latent_dim=int(config["model"]["latent_dim"]),
        device=device,
        output_path=generation_output,
        num_samples=int(args.num_generate),
    )

    # 15) 记录输出文件路径，便于快速定位结果。
    logger.info("Saved reconstruction image to %s", infer_output)
    logger.info("Saved generation image to %s", generation_output)


if __name__ == "__main__":
    # 脚本入口。
    main()
