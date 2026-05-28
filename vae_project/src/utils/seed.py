"""随机种子工具。

在深度学习实验中，固定随机种子可以尽量保证可复现：
- 数据划分一致；
- 参数初始化一致；
- 采样过程一致。
"""

# 启用延迟类型注解支持。
from __future__ import annotations

# Python 内置随机库。
import random

# NumPy 随机库。
import numpy as np
# PyTorch 随机库。
import torch


def set_seed(seed: int) -> None:
    """设置所有常见随机源的种子。"""
    # 固定 Python 原生随机。
    random.seed(seed)
    # 固定 NumPy 随机。
    np.random.seed(seed)
    # 固定 PyTorch CPU 随机。
    torch.manual_seed(seed)

    # 如果使用 CUDA，额外固定 GPU 随机。
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 让 CuDNN 走确定性算法（可复现更强，但速度可能略慢）。
    torch.backends.cudnn.deterministic = True
    # 关闭 benchmark，防止动态选择最快算法带来非确定性。
    torch.backends.cudnn.benchmark = False
