"""日志工具模块。

提供统一日志输出：
- 控制台：用于实时查看训练进度；
- 文件：用于实验留档和问题排查。
"""

# 启用延迟类型注解支持。
from __future__ import annotations

# Python 标准日志库。
import logging
# 路径工具。
from pathlib import Path


def setup_logger(name: str, log_file: Path) -> logging.Logger:
    """创建同时输出到控制台和文件的 logger。"""
    # 通过名字获取（或创建）logger 实例。
    logger = logging.getLogger(name)
    # logger 总级别设置为 DEBUG，具体输出由 handler 级别控制。
    logger.setLevel(logging.DEBUG)
    # 关闭向父 logger 传播，避免重复打印。
    logger.propagate = False

    # 防止重复调用时反复添加 handler（会导致日志重复）。
    if logger.handlers:
        return logger

    # 统一日志格式：时间 + 级别 + 消息内容。
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台 handler：通常展示 INFO 及以上。
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # 确保日志文件父目录存在。
    log_file.parent.mkdir(parents=True, exist_ok=True)
    # 文件 handler：保存 DEBUG 及以上，信息更完整。
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # 把两个 handler 都挂到 logger 上。
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    # 返回配置好的 logger。
    return logger
