"""Standard 模式训练器。

相比 quick 模式，本训练器增加了：
1) 每个 epoch 的验证集评估；
2) 每个 epoch 的 checkpoint 保存；
3) best 模型保存；
4) loss 曲线持续更新。
"""

# 启用延迟类型注解支持。
from __future__ import annotations

# 日志模块。
import logging
# 路径类型。
from pathlib import Path
# 类型标注工具。
from typing import Any, Dict, Tuple

# PyTorch 主库。
import torch
# 神经网络模块基类。
from torch import nn
# Adam 优化器。
from torch.optim import Adam
# DataLoader 类型。
from torch.utils.data import DataLoader

# ELBO 损失函数。
from src.losses.elbo_loss import elbo_loss
# 指标追踪器。
from src.utils.metrics import MetricsTracker
# 可视化函数（loss 曲线、重构、生成、t-SNE）。
from src.utils.visualizer import (
    plot_loss_curve,
    save_generation_grid,
    save_latent_tsne,
    save_reconstruction_grid,
)


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    optimizer: Adam,
    beta: float,
    device: torch.device,
    epoch: int,
    epochs: int,
    logger: logging.Logger,
) -> Tuple[float, float, float]:
    """训练单个 epoch，返回平均 total/recon/kl。"""
    # 训练模式。
    model.train()
    # epoch 累计器。
    total_loss = 0.0
    recon_loss = 0.0
    kl_loss = 0.0

    # 逐 batch 训练。
    for iteration, (images, _) in enumerate(loader, start=1):
        # 数据转移到设备。
        images = images.to(device)
        # 清空梯度。
        optimizer.zero_grad()

        # 前向 + 损失计算。
        reconstruction, mu, logvar, _ = model(images)
        loss_out = elbo_loss(reconstruction, images, mu, logvar, beta=beta)
        # 反向传播。
        loss_out.total.backward()
        # 参数更新。
        optimizer.step()

        # 累加当前 batch 损失。
        total_loss += float(loss_out.total.item())
        recon_loss += float(loss_out.recon.item())
        kl_loss += float(loss_out.kl.item())

        # 打印迭代日志。
        logger.info(
            "Epoch [%d/%d] Iter [%d/%d] loss=%.4f recon_loss=%.4f kl_loss=%.4f",
            epoch,
            epochs,
            iteration,
            len(loader),
            loss_out.total.item(),
            loss_out.recon.item(),
            loss_out.kl.item(),
        )

    # 返回 epoch 平均损失。
    return total_loss / len(loader), recon_loss / len(loader), kl_loss / len(loader)


def _validate_one_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    beta: float,
    device: torch.device,
) -> Tuple[float, float, float]:
    """验证单个 epoch，返回平均 total/recon/kl。"""
    # 评估模式（关闭训练态行为）。
    model.eval()
    # 验证累加器。
    total_loss = 0.0
    recon_loss = 0.0
    kl_loss = 0.0

    # 关闭梯度，减少显存和计算。
    with torch.no_grad():
        for images, _ in loader:
            # 数据转移到设备。
            images = images.to(device)
            # 前向和损失计算。
            reconstruction, mu, logvar, _ = model(images)
            loss_out = elbo_loss(reconstruction, images, mu, logvar, beta=beta)
            total_loss += float(loss_out.total.item())
            recon_loss += float(loss_out.recon.item())
            kl_loss += float(loss_out.kl.item())

    # 返回验证集平均损失。
    return total_loss / len(loader), recon_loss / len(loader), kl_loss / len(loader)


def run_standard_training(
    model: nn.Module,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    config: Dict[str, Any],
    device: torch.device,
    output_dirs: Dict[str, Path],
    logger: logging.Logger,
) -> None:
    """执行完整标准训练流程。"""
    # 读取运行时参数和训练超参数。
    runtime = config["runtime"]
    train_cfg = config["train"]
    epochs = int(runtime["epochs"])
    beta = float(train_cfg["beta"])
    learning_rate = float(train_cfg["lr"])

    # 模型转到目标设备。
    model.to(device)
    # 初始化优化器。
    optimizer = Adam(model.parameters(), lr=learning_rate)
    # 初始化指标追踪器。
    tracker = MetricsTracker()
    # 记录最佳验证损失。
    best_val_loss = float("inf")

    # 启动日志。
    logger.info("Standard training started. epochs=%d, batch_size=%d", epochs, runtime["batch_size"])

    # 主训练循环。
    for epoch in range(1, epochs + 1):
        # 训练一个 epoch。
        train_total, train_recon, train_kl = _train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            beta=beta,
            device=device,
            epoch=epoch,
            epochs=epochs,
            logger=logger,
        )
        # 验证一个 epoch。
        val_total, val_recon, val_kl = _validate_one_epoch(
            model=model,
            loader=val_loader,
            beta=beta,
            device=device,
        )

        # 记录 train/val 指标。
        tracker.add("train", epoch, train_total, train_recon, train_kl)
        tracker.add("val", epoch, val_total, val_recon, val_kl)

        # 打印 epoch 汇总。
        logger.info(
            "Epoch [%d/%d] train_loss=%.4f val_loss=%.4f val_recon=%.4f val_kl=%.4f",
            epoch,
            epochs,
            train_total,
            val_total,
            val_recon,
            val_kl,
        )

        # 组织 checkpoint 字典。
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
        }

        # 若配置开启，则每个 epoch 都保存一个 checkpoint。
        if bool(runtime["save_every_epoch"]):
            epoch_path = output_dirs["checkpoints_dir"] / f"epoch_{epoch:03d}.pt"
            torch.save(checkpoint, epoch_path)

        # 若当前验证损失更优，则覆盖 best.pt。
        if val_total < best_val_loss:
            best_val_loss = val_total
            best_path = output_dirs["checkpoints_dir"] / "best.pt"
            torch.save(checkpoint, best_path)

        # 每个 epoch 更新一次 loss 曲线图。
        plot_loss_curve(
            metrics=tracker.records,
            output_path=output_dirs["visuals_dir"] / "loss_curve.png",
        )

    # 训练结束后，生成最终重构图。
    sample_images, _ = next(iter(val_loader))
    save_reconstruction_grid(
        model=model,
        batch_images=sample_images,
        device=device,
        output_path=output_dirs["visuals_dir"] / "reconstruction.png",
    )
    # 生成随机采样图。
    save_generation_grid(
        model=model,
        latent_dim=int(config["model"]["latent_dim"]),
        device=device,
        output_path=output_dirs["visuals_dir"] / "generation.png",
    )
    # 生成潜空间 t-SNE 图。
    save_latent_tsne(
        model=model,
        dataloader=val_loader,
        device=device,
        output_path=output_dirs["visuals_dir"] / "latent_tsne.png",
    )

    # 保存指标文件。
    tracker.save_json(output_dirs["logs_dir"] / "standard_metrics.json")
    tracker.save_csv(output_dirs["logs_dir"] / "standard_metrics.csv")
    # 训练结束日志。
    logger.info("Standard training completed.")
