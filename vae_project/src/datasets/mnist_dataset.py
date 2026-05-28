"""数据集与 DataLoader 构建模块。

目标：
1) 支持 MNIST / FashionMNIST / CelebA；
2) 提供统一的 build_dataloaders 接口；
3) 为训练脚本屏蔽数据准备细节。
"""

# 启用延迟类型注解支持。
from __future__ import annotations

# 路径处理工具。
from pathlib import Path
# 类型标注工具。
from typing import Any, Dict, Tuple

# PyTorch 主库（用于随机数生成器等）。
import torch
# DataLoader、Dataset、random_split 构建训练输入管线。
from torch.utils.data import DataLoader, Dataset, random_split
# torchvision 数据集与预处理变换。
from torchvision import datasets, transforms


def _build_transform(dataset_name: str, image_size: int) -> transforms.Compose:
    """根据数据集类型构建预处理变换。"""
    # MNIST/FashionMNIST 是灰度小图，通常只需 ToTensor。
    if dataset_name in {"mnist", "fashion_mnist"}:
        tfms = [transforms.ToTensor()]
        # 若配置了非 28 尺寸，先 resize 再转 tensor。
        if image_size != 28:
            tfms.insert(0, transforms.Resize((image_size, image_size)))
        return transforms.Compose(tfms)

    # CelebA 原图较大，通常先裁剪人脸区域再缩放。
    if dataset_name == "celeba":
        return transforms.Compose(
            [
                transforms.CenterCrop(140),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    # 不支持的数据集名，明确报错。
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _build_dataset(
    dataset_name: str,
    root_dir: Path,
    transform: transforms.Compose,
    train: bool,
) -> Dataset[Any]:
    """根据数据集名实例化 torchvision 数据集对象。"""
    # MNIST 数据集。
    if dataset_name == "mnist":
        return datasets.MNIST(
            root=str(root_dir),
            train=train,
            transform=transform,
            download=True,
        )
    # FashionMNIST 数据集。
    if dataset_name == "fashion_mnist":
        return datasets.FashionMNIST(
            root=str(root_dir),
            train=train,
            transform=transform,
            download=True,
        )
    # CelebA 数据集：train/valid split 命名与 MNIST 不同。
    if dataset_name == "celeba":
        split = "train" if train else "valid"
        return datasets.CelebA(
            root=str(root_dir),
            split=split,
            target_type="attr",
            transform=transform,
            download=True,
        )
    # 不支持的数据集名。
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_dataloaders(
    config: Dict[str, Any],
    batch_size: int,
) -> Tuple[DataLoader[Any], DataLoader[Any]]:
    """从配置构建 train/val DataLoader。"""
    # 读取配置节点。
    dataset_cfg = config["dataset"]
    project_cfg = config["project"]
    # 标准化数据集名，避免大小写差异。
    dataset_name = str(dataset_cfg["name"]).lower()
    # 图像尺寸。
    image_size = int(dataset_cfg["image_size"])
    # 验证集比例。
    val_split = float(dataset_cfg.get("val_split", 0.1))
    # DataLoader 线程数。
    num_workers = int(dataset_cfg.get("num_workers", 2))
    # 随机种子（用于可复现数据划分）。
    seed = int(project_cfg["seed"])
    # 数据目录。
    data_dir = Path(project_cfg["data_dir"])
    # 若目录不存在则创建。
    data_dir.mkdir(parents=True, exist_ok=True)

    # 构建预处理流程与数据集实例。
    transform = _build_transform(dataset_name, image_size)
    full_train = _build_dataset(dataset_name, data_dir, transform, train=True)
    val_dataset = _build_dataset(dataset_name, data_dir, transform, train=False)

    # 对 MNIST/FashionMNIST：从训练集再切一部分作为验证集（更统一）。
    if dataset_name in {"mnist", "fashion_mnist"}:
        # 验证集最少保留 1 条样本，避免空集。
        val_size = max(1, int(len(full_train) * val_split))
        train_size = len(full_train) - val_size
        # 固定划分随机性，保证每次切分一致。
        generator = torch.Generator().manual_seed(seed)
        train_dataset, split_val_dataset = random_split(
            full_train,
            [train_size, val_size],
            generator=generator,
        )
        val_dataset = split_val_dataset
    else:
        # CelebA 已有官方 train/valid 划分，直接使用。
        train_dataset = full_train

    # pin_memory 仅在 CUDA 有意义；MPS/CPU 下关闭以避免无效告警。
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    # 验证集不打乱，便于稳定评估。
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    # 返回训练与验证加载器。
    return train_loader, val_loader
