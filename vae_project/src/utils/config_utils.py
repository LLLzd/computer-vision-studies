"""配置处理工具模块。

职责：
1) 从 YAML 文件加载配置；
2) 把 quick / standard 模式参数合并到 runtime；
3) 创建输出目录并返回路径字典。
"""

# 启用延迟类型注解支持。
from __future__ import annotations

# Path 用于跨平台路径操作。
from pathlib import Path
# Any/Dict 用于配置字典的类型标注。
from typing import Any, Dict

# PyYAML 用于读取 YAML 文件。
import yaml


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """从 YAML 文件加载配置字典。"""
    # 统一将输入转为 Path，便于后续文件操作。
    path = Path(config_path)
    # 配置文件不存在时，立刻报错，避免后续出现隐式错误。
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    # 以 UTF-8 打开文件并安全解析 YAML。
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    # 期望配置顶层是 mapping/dict。
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    # 返回配置字典。
    return data


def apply_mode(config: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """将 quick/standard 模式参数合并到 runtime 字段。"""
    # 仅支持两种训练模式，防止拼写错误导致静默失败。
    if mode not in {"quick", "standard"}:
        raise ValueError(f"Unsupported mode: {mode}")

    # 浅拷贝顶层配置，避免直接修改原始对象引用。
    merged = dict(config)
    # 提取指定模式配置（例如 quick 节）。
    mode_cfg = dict(config.get(mode, {}))
    # 统一 runtime 字段，后续代码只读取 runtime，避免分支重复。
    runtime = {
        "mode": mode,
        "epochs": int(mode_cfg["epochs"]),
        "batch_size": int(mode_cfg["batch_size"]),
        "log_interval": int(mode_cfg.get("log_interval", 1)),
        "save_every_epoch": bool(mode_cfg.get("save_every_epoch", False)),
    }
    # 写回 runtime。
    merged["runtime"] = runtime
    # 返回合并后的配置。
    return merged


def ensure_output_dirs(config: Dict[str, Any], project_root: Path) -> Dict[str, Path]:
    """创建输出目录并返回目录路径字典。"""
    # 输出根目录（通常是 <project_root>/outputs）。
    output_root = project_root / str(config["project"]["output_dir"])
    # checkpoint 保存目录。
    checkpoints_dir = output_root / "checkpoints"
    # 日志目录。
    logs_dir = output_root / "logs"
    # 可视化结果目录。
    visuals_dir = output_root / "visuals"

    # 逐个创建目录；parents=True 表示父目录不存在时一并创建。
    for directory in (output_root, checkpoints_dir, logs_dir, visuals_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # 返回统一路径字典，便于各模块按 key 获取。
    return {
        "output_root": output_root,
        "checkpoints_dir": checkpoints_dir,
        "logs_dir": logs_dir,
        "visuals_dir": visuals_dir,
    }
