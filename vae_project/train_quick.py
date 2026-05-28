"""Quick 模式训练入口。

该脚本面向“快速验证模型是否跑通”的场景：
1) 读取 YAML 配置；
2) 选择 quick 模式超参数（如 10 epochs）；
3) 构建数据、模型和日志系统；
4) 调用 quick trainer 执行训练。
"""

# 允许在当前文件中使用延迟类型注解（提升类型标注兼容性）。
from __future__ import annotations

# argparse: 解析命令行参数（如 --config）。
import argparse
# Path: 跨平台路径处理，比字符串拼接更安全。
from pathlib import Path

# 构建训练/验证 DataLoader。
from src.datasets.mnist_dataset import build_dataloaders
# VAE 模型定义。
from src.models.vae import VAE
# quick 训练流程封装。
from src.trainer.quick_trainer import run_quick_training
# 配置读取、模式合并、输出目录创建。
from src.utils.config_utils import apply_mode, ensure_output_dirs, load_config
# 自动选择 CUDA / MPS / CPU。
from src.utils.device_utils import get_device
# 初始化控制台与文件日志器。
from src.utils.logger import setup_logger
# 固定随机种子，保障可复现性。
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """定义并解析命令行参数。"""
    # 创建命令行解析器，并写入脚本说明。
    parser = argparse.ArgumentParser(description="Quick training for VAE.")
    # --config 允许用户指定任意配置文件路径。
    parser.add_argument(
        "--config",
        type=str,
        # 默认读取项目内的默认配置文件。
        default="configs/default.yaml",
        help="Path to YAML config file.",
    )
    # 真正执行解析并返回 Namespace 对象。
    return parser.parse_args()


def main() -> None:
    """执行 quick 模式完整训练流水线。"""
    # 1) 读取命令行参数。
    args = parse_args()
    # 2) 获取当前脚本所在目录，作为项目根目录。
    project_root = Path(__file__).resolve().parent
    # 3) 加载 YAML 配置为 Python 字典。
    config = load_config(project_root / args.config)
    # 4) 合并 quick 模式参数到 config["runtime"]。
    config = apply_mode(config, mode="quick")

    # 5) 将相对路径转换为绝对路径，避免从不同 cwd 运行时出错。
    config["project"]["data_dir"] = str(project_root / config["project"]["data_dir"])
    config["project"]["output_dir"] = str(project_root / config["project"]["output_dir"])

    # 6) 固定随机种子，保证可重复实验结果。
    set_seed(int(config["project"]["seed"]))
    # 7) 自动创建 outputs/checkpoints/logs/visuals 目录。
    output_dirs = ensure_output_dirs(config, project_root=project_root)
    # 8) 初始化日志器，日志落盘到 quick_train.log。
    logger = setup_logger("vae_quick", output_dirs["logs_dir"] / "quick_train.log")
    # 9) 自动选择训练设备（cuda > mps > cpu）。
    device = get_device(config)
    logger.info("Using device: %s", device)

    # 10) 构建训练/验证数据加载器，batch_size 从 runtime 读取。
    train_loader, val_loader = build_dataloaders(
        config=config,
        batch_size=int(config["runtime"]["batch_size"]),
    )
    # 11) 按配置实例化 VAE 模型。
    model = VAE(
        in_channels=int(config["model"]["in_channels"]),
        latent_dim=int(config["model"]["latent_dim"]),
        hidden_dims=list(config["model"]["hidden_dims"]),
        image_size=int(config["dataset"]["image_size"]),
    )

    # 12) 调用 quick trainer，执行训练与可视化导出。
    run_quick_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        output_dirs=output_dirs,
        logger=logger,
    )


if __name__ == "__main__":
    # 仅当本文件被“直接运行”时执行 main()。
    main()
