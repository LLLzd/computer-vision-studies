"""独立可视化脚本：从 checkpoint 直接导出核心图像结果。

该脚本适合“训练完成后补做可视化”的场景：
1) 重构图（Original vs Reconstruction）；
2) 随机生成图（z 采样）；
3) 潜空间 t-SNE 散点图。
"""

# 启用延迟类型注解支持。
from __future__ import annotations

# 命令行参数解析库。
import argparse
# 路径处理工具。
from pathlib import Path

# PyTorch 核心库。
import torch

# 数据加载器构建函数。
from src.datasets.mnist_dataset import build_dataloaders
# VAE 模型定义。
from src.models.vae import VAE
# 配置读取与输出目录管理函数。
from src.utils.config_utils import ensure_output_dirs, load_config
# 自动设备选择函数。
from src.utils.device_utils import get_device
# 日志器构建函数。
from src.utils.logger import setup_logger
# 固定随机种子函数。
from src.utils.seed import set_seed
# 可视化工具函数集合。
from src.utils.visualizer import (
    save_generation_grid,
    save_latent_tsne,
    save_reconstruction_grid,
)


def parse_args() -> argparse.Namespace:
    """解析脚本参数。"""
    # 创建参数解析器。
    parser = argparse.ArgumentParser(
        description="Generate reconstruction/generation/t-SNE visualizations.",
    )
    # 配置文件路径。
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file.",
    )
    # checkpoint 路径。
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/best.pt",
        help="Checkpoint path. Fallback to outputs/checkpoints/vae_final.pt if missing.",
    )
    # 可视化读取数据时的 batch size。
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size used for visualization dataloader.",
    )
    # 返回解析结果。
    return parser.parse_args()


def _load_model(
    config: dict,
    checkpoint_path: Path,
    device: torch.device,
) -> VAE:
    """创建模型并恢复训练好的参数。"""
    # 按配置构建同结构 VAE。
    model = VAE(
        in_channels=int(config["model"]["in_channels"]),
        latent_dim=int(config["model"]["latent_dim"]),
        hidden_dims=list(config["model"]["hidden_dims"]),
        image_size=int(config["dataset"]["image_size"]),
    )

    # 若 best.pt 缺失，尝试回退到 quick 模式的最终权重。
    if not checkpoint_path.exists():
        fallback = checkpoint_path.parent / "vae_final.pt"
        if fallback.exists():
            checkpoint_path = fallback
        else:
            # 两个候选都不存在时给出明确错误。
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path} or {fallback}",
            )

    # 读取 checkpoint 并恢复模型权重。
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    # 切换到目标设备并设置 eval 模式。
    model.to(device)
    model.eval()
    return model


def main() -> None:
    """从训练权重生成三类可视化图。"""
    # 1) 解析参数。
    args = parse_args()
    # 2) 定位项目根目录。
    project_root = Path(__file__).resolve().parent
    # 3) 读取配置文件。
    config = load_config(project_root / args.config)
    # 4) 统一路径为绝对路径，减少运行歧义。
    config["project"]["data_dir"] = str(project_root / config["project"]["data_dir"])
    config["project"]["output_dir"] = str(project_root / config["project"]["output_dir"])

    # 5) 固定随机种子（t-SNE 和采样过程更可复现）。
    set_seed(int(config["project"]["seed"]))
    # 6) 创建输出目录。
    output_dirs = ensure_output_dirs(config, project_root=project_root)
    # 7) 初始化日志。
    logger = setup_logger("vae_visualize", output_dirs["logs_dir"] / "visualize.log")
    # 8) 自动选择设备。
    device = get_device(config)
    logger.info("Using device: %s", device)

    # 9) 加载已训练模型权重。
    model = _load_model(
        config=config,
        checkpoint_path=project_root / args.checkpoint,
        device=device,
    )
    # 10) 构建验证集加载器，供重构与 t-SNE 使用。
    _, val_loader = build_dataloaders(
        config=config,
        batch_size=int(args.batch_size),
    )
    # 取一批图像做重构图。
    images, _ = next(iter(val_loader))

    # 11) 导出重构对比图。
    save_reconstruction_grid(
        model=model,
        batch_images=images,
        device=device,
        output_path=output_dirs["visuals_dir"] / "reconstruction.png",
    )
    # 12) 导出随机生成网格图。
    save_generation_grid(
        model=model,
        latent_dim=int(config["model"]["latent_dim"]),
        device=device,
        output_path=output_dirs["visuals_dir"] / "generation.png",
    )
    # 13) 导出潜空间 t-SNE 图。
    save_latent_tsne(
        model=model,
        dataloader=val_loader,
        device=device,
        output_path=output_dirs["visuals_dir"] / "latent_tsne.png",
    )
    # 14) 打印可视化输出目录。
    logger.info("All visualization outputs have been saved to %s", output_dirs["visuals_dir"])


if __name__ == "__main__":
    # 脚本入口。
    main()
