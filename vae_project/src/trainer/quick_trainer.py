"""Quick 模式训练器。

定位：
- 快速检查模型是否正常收敛；
- 保留最关键的日志、模型与可视化产物。
"""

# 启用延迟类型注解支持。
from __future__ import annotations

# 日志接口类型。
import logging
# 路径类型。
from pathlib import Path
# 类型标注工具。
from typing import Any, Dict

# PyTorch 主库。
import torch
# 神经网络模块基类。
from torch import nn
# Adam 优化器。
from torch.optim import Adam
# 数据加载器类型。
from torch.utils.data import DataLoader

# ELBO 损失函数。
from src.losses.elbo_loss import elbo_loss
# 指标记录器。
from src.utils.metrics import MetricsTracker
# 可视化导出函数。
from src.utils.visualizer import (
    save_generation_grid,
    save_latent_tsne,
    save_reconstruction_grid,
)


def run_quick_training(
    model: nn.Module,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    config: Dict[str, Any],
    device: torch.device,
    output_dirs: Dict[str, Path],
    logger: logging.Logger,
) -> None:
    """执行 quick 训练并导出关键结果。"""
    # 读取运行时参数与训练超参。
    runtime = config["runtime"]
    train_cfg = config["train"]
    epochs = int(runtime["epochs"])
    beta = float(train_cfg["beta"])
    learning_rate = float(train_cfg["lr"])

    # 把模型移动到目标设备。
    model.to(device)
    # 创建 Adam 优化器。
    optimizer = Adam(model.parameters(), lr=learning_rate)
    # 初始化指标追踪器。
    tracker = MetricsTracker()

    # 打印 quick 模式启动信息。
    logger.info("Quick training started. epochs=%d, batch_size=%d", epochs, runtime["batch_size"])

    # 逐 epoch 训练。
    for epoch in range(1, epochs + 1):
        # 开启训练模式（启用 BN 训练行为等）。
        model.train()
        # 累计器：记录 epoch 内平均损失。
        epoch_total = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0

        # 逐 batch 迭代。
        for iteration, (images, _) in enumerate(train_loader, start=1):
            # 数据放到目标设备。
            images = images.to(device)
            # 清空上一轮梯度。
            optimizer.zero_grad()

            # 前向得到重构与潜变量统计量。
            reconstruction, mu, logvar, _ = model(images)
            # 计算 ELBO。
            loss_out = elbo_loss(reconstruction, images, mu, logvar, beta=beta)
            # 反向传播。
            loss_out.total.backward()
            # 参数更新。
            optimizer.step()

            # 累加本 batch 各项损失。
            epoch_total += float(loss_out.total.item())
            epoch_recon += float(loss_out.recon.item())
            epoch_kl += float(loss_out.kl.item())

            # 打印迭代级别日志（便于观察训练稳定性）。
            logger.info(
                "Epoch [%d/%d] Iter [%d/%d] loss=%.4f recon_loss=%.4f kl_loss=%.4f",
                epoch,
                epochs,
                iteration,
                len(train_loader),
                loss_out.total.item(),
                loss_out.recon.item(),
                loss_out.kl.item(),
            )

        # 计算 epoch 平均损失。
        avg_total = epoch_total / len(train_loader)
        avg_recon = epoch_recon / len(train_loader)
        avg_kl = epoch_kl / len(train_loader)
        # 保存到指标追踪器。
        tracker.add("train", epoch, avg_total, avg_recon, avg_kl)
        # 打印 epoch 汇总日志。
        logger.info(
            "Epoch [%d/%d] summary: loss=%.4f recon_loss=%.4f kl_loss=%.4f",
            epoch,
            epochs,
            avg_total,
            avg_recon,
            avg_kl,
        )

    # 训练结束后保存最终 checkpoint。
    final_ckpt = output_dirs["checkpoints_dir"] / "vae_final.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epochs,
            "config": config,
        },
        final_ckpt,
    )
    logger.info("Saved final model to %s", final_ckpt)

    # 从验证集取一个 batch 生成重构图。
    sample_images, _ = next(iter(val_loader))
    save_reconstruction_grid(
        model=model,
        batch_images=sample_images,
        device=device,
        output_path=output_dirs["visuals_dir"] / "reconstruction.png",
    )
    # 导出随机生成图。
    save_generation_grid(
        model=model,
        latent_dim=int(config["model"]["latent_dim"]),
        device=device,
        output_path=output_dirs["visuals_dir"] / "generation.png",
    )
    # 导出潜空间 t-SNE 图。
    save_latent_tsne(
        model=model,
        dataloader=val_loader,
        device=device,
        output_path=output_dirs["visuals_dir"] / "latent_tsne.png",
    )

    # 持久化指标到 JSON 和 CSV。
    tracker.save_json(output_dirs["logs_dir"] / "quick_metrics.json")
    tracker.save_csv(output_dirs["logs_dir"] / "quick_metrics.csv")
    # quick 训练结束日志。
    logger.info("Quick training completed.")
