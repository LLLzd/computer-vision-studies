from __future__ import annotations

import psutil
import torch


def pick_torch_device() -> torch.device:
    """优先使用 Apple Silicon 的 MPS，其次 CPU。"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def memory_report_gb() -> str:
    """返回当前进程常驻内存，单位 GB。"""
    mem = psutil.Process().memory_info().rss / (1024 ** 3)
    return f"{mem:.2f} GB"
