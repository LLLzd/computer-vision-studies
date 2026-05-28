"""设备选择工具。

目标：在不改代码的前提下，自动或手动选择训练设备：
- auto: 优先 cuda，其次 mps，最后 cpu；
- cuda/mps/cpu: 按用户指定尝试，若不可用则回退 cpu。
"""

# 启用延迟类型注解支持。
from __future__ import annotations

# 类型标注工具。
from typing import Any, Dict

# PyTorch 设备能力查询。
import torch


def get_device(config: Dict[str, Any]) -> torch.device:
    """根据配置和可用硬件返回训练设备。"""
    # 读取配置中的 device.type，并统一转小写处理。
    preferred = str(config.get("device", {}).get("type", "auto")).lower()

    # 自动模式：优先 NVIDIA CUDA，再尝试 Apple MPS，最后 CPU。
    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # 手动指定 CUDA，且 CUDA 可用。
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    # 手动指定 MPS，且 MPS 可用。
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    # 其余情况安全回退到 CPU。
    return torch.device("cpu")
