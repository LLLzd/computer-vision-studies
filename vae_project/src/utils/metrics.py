"""训练指标记录模块。

提供 epoch 级别指标追踪，并支持导出到 JSON / CSV。
"""

# 启用延迟类型注解支持。
from __future__ import annotations

# CSV 文件写入模块。
import csv
# JSON 文件写入模块。
import json
# dataclass 用于定义结构化数据对象。
from dataclasses import asdict, dataclass
# 路径工具。
from pathlib import Path
# 类型标注工具。
from typing import Dict, List


@dataclass
class EpochMetric:
    """单个 epoch 的指标记录结构。"""

    # 数据切分名称：train 或 val。
    split: str
    # epoch 编号（从 1 开始）。
    epoch: int
    # 总损失。
    loss: float
    # 重构损失。
    recon_loss: float
    # KL 散度损失。
    kl_loss: float


class MetricsTracker:
    """负责收集并持久化多个 epoch 指标。"""

    def __init__(self) -> None:
        # records 是指标主存储列表。
        self.records: List[EpochMetric] = []

    def add(
        self,
        split: str,
        epoch: int,
        loss: float,
        recon_loss: float,
        kl_loss: float,
    ) -> None:
        """添加一条 epoch 指标记录。"""
        # 把传入原始值封装为 EpochMetric 对象后加入列表。
        self.records.append(
            EpochMetric(
                split=split,
                epoch=epoch,
                loss=loss,
                recon_loss=recon_loss,
                kl_loss=kl_loss,
            )
        )

    def to_dict_list(self) -> List[Dict[str, float | int | str]]:
        """把 dataclass 列表转换为可序列化字典列表。"""
        # asdict 会把 dataclass 自动展开成标准字典。
        return [asdict(record) for record in self.records]

    def save_json(self, output_path: Path) -> None:
        """将指标保存为 JSON 文件。"""
        # 确保输出目录存在。
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # UTF-8 写入，ensure_ascii=False 保留中文可读性。
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict_list(), handle, ensure_ascii=False, indent=2)

    def save_csv(self, output_path: Path) -> None:
        """将指标保存为 CSV 文件。"""
        # 确保输出目录存在。
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # newline="" 避免在某些系统上出现空行问题。
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            # 设定 CSV 列顺序，便于后续分析工具读取。
            writer = csv.DictWriter(
                handle,
                fieldnames=["split", "epoch", "loss", "recon_loss", "kl_loss"],
            )
            # 写表头。
            writer.writeheader()
            # 逐行写入每条记录。
            for item in self.to_dict_list():
                writer.writerow(item)
