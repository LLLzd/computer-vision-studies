"""Qwen3-VL Embedding 模型封装。"""

from .qwen3_vl_embedding import Qwen3VLEmbedder
from .wrapper import EmbeddingEngine

__all__ = ["Qwen3VLEmbedder", "EmbeddingEngine"]
