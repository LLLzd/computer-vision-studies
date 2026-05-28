"""Standard 模式训练入口。

该脚本用于正式实验（更长训练、更完整记录）：
1) 读取配置并切换到 standard 模式；
2) 构建数据集、模型、日志与输出目录；
3) 调用 standard trainer 执行训练、验证与 checkpoint 保存。
"""

# 支持更灵活的类型注解写法（延迟求值）。
from __future__ import annotations

# 命令行参数解析器。
import argparse
# 路径处理工具。
from pathlib import Path

# 数据加载器构建函数。
from src.datasets.mnist_dataset import build_dataloaders
# VAE 模型定义。
from src.models.vae import VAE
# 标准训练流程。
from src.trainer.standard_trainer import run_standard_training
# 配置相关工具函数。
from src.utils.config_utils import apply_mode, ensure_output_dirs, load_config
# 设备选择函数。
from src.utils.device_utils import get_device
# 日志初始化函数。
from src.utils.logger import setup_logger
# 随机种子固定函数。
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    # 创建参数解析器对象。
    parser = argparse.ArgumentParser(description="Standard training for VAE.")
    # 允许用户替换默认配置文件。
    parser.add_argument(
        "--config",
        type=str,
        # 默认配置路径。
        default="configs/default.yaml",
        help="Path to YAML config file.",
    )
    # 返回解析结果。
    return parser.parse_args()


def main() -> None:
    """执行 standard 模式完整训练流水线。"""
    # 1) 参数解析。
    args = parse_args()
    # 2) 获取项目根目录。
    project_root = Path(__file__).resolve().parent
    # 3) 读取配置文件。
    config = load_config(project_root / args.config)
    # 4) 合并 standard 模式参数到 runtime。
    config = apply_mode(config, mode="standard")

    # 5) 统一把 data/output 路径转成绝对路径。
    config["project"]["data_dir"] = str(project_root / config["project"]["data_dir"])
    config["project"]["output_dir"] = str(project_root / config["project"]["output_dir"])

    # 6) 固定随机种子，便于结果复现。
    set_seed(int(config["project"]["seed"]))
    # 7) 创建输出目录结构。
    output_dirs = ensure_output_dirs(config, project_root=project_root)
    # 8) 初始化日志系统（写入标准训练日志文件）。
    logger = setup_logger(
        "vae_standard",
        output_dirs["logs_dir"] / "standard_train.log",
    )
    # 9) 自动选择最优可用设备。
    device = get_device(config)
    logger.info("Using device: %s", device)

    # 10) 构建训练与验证数据集加载器。
    train_loader, val_loader = build_dataloaders(
        config=config,
        batch_size=int(config["runtime"]["batch_size"]),
    )
    # 11) 构建 VAE 模型实例。
    model = VAE(
        in_channels=int(config["model"]["in_channels"]),
        latent_dim=int(config["model"]["latent_dim"]),
        hidden_dims=list(config["model"]["hidden_dims"]),
        image_size=int(config["dataset"]["image_size"]),
    )

    # 12) 启动标准训练流程（含 val、checkpoint、可视化）。
    run_standard_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        output_dirs=output_dirs,
        logger=logger,
    )


if __name__ == "__main__":
    # 脚本入口。
    main()
