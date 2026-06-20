"""
Embedding 引擎：封装 Qwen3-VL，支持单帧/批量/文本编码。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from config import (
    DEFAULT_INSTRUCTION,
    EMBED_BATCH_SIZE,
    MODEL_HF_ID,
    MODEL_LOCAL_DIR,
    QUERY_INSTRUCTION,
)
from src.embedder.qwen3_vl_embedding import Qwen3VLEmbedder


class EmbeddingEngine:
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        instruction: str = DEFAULT_INSTRUCTION,
        batch_size: int = EMBED_BATCH_SIZE,
    ):
        self.instruction = instruction
        self.batch_size = batch_size
        self.model_path = self._resolve_model_path(model_path)
        self._embedder: Optional[Qwen3VLEmbedder] = None

    @staticmethod
    def _resolve_model_path(model_path: Optional[Union[str, Path]]) -> str:
        if model_path is not None:
            return str(model_path)
        if MODEL_LOCAL_DIR.exists():
            return str(MODEL_LOCAL_DIR)
        return MODEL_HF_ID

    @property
    def embedder(self) -> Qwen3VLEmbedder:
        if self._embedder is None:
            print(f"Loading Qwen3-VL Embedding: {self.model_path}")
            self._embedder = Qwen3VLEmbedder(
                model_name_or_path=self.model_path,
                default_instruction=self.instruction,
            )
        return self._embedder

    def embed_images(
        self,
        images: List[Image.Image],
        texts: Optional[List[Optional[str]]] = None,
        instruction: Optional[str] = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """批量编码图像（可选附带文本描述）。"""
        if not images:
            return np.zeros((0, 0), dtype=np.float32)

        if texts is None:
            texts = [None] * len(images)
        if len(texts) != len(images):
            raise ValueError("texts 长度必须与 images 一致")

        all_embeddings: List[np.ndarray] = []
        inst = instruction or self.instruction
        batches = range(0, len(images), self.batch_size)
        iterator = tqdm(batches, desc="Embedding", disable=not show_progress)

        for start in iterator:
            end = min(start + self.batch_size, len(images))
            batch_inputs = []
            for i in range(start, end):
                item: dict = {"image": images[i], "instruction": inst}
                if texts[i]:
                    item["text"] = texts[i]
                batch_inputs.append(item)

            emb = self.embedder.process(batch_inputs)
            all_embeddings.append(emb.cpu().float().numpy())

        return np.vstack(all_embeddings).astype(np.float32)

    def embed_text(self, query: str, instruction: Optional[str] = None) -> np.ndarray:
        """编码文本查询。"""
        inst = instruction or QUERY_INSTRUCTION
        emb = self.embedder.process([{"text": query, "instruction": inst}])
        return emb.cpu().float().numpy()[0]

    def embed_video_file(
        self,
        video_path: Union[str, Path],
        instruction: Optional[str] = None,
        fps: float = 1.0,
        max_frames: int = 64,
    ) -> np.ndarray:
        """直接对整段视频文件编码（模型内部抽帧）。"""
        inst = instruction or self.instruction
        emb = self.embedder.process(
            [{"video": str(video_path), "instruction": inst, "fps": fps, "max_frames": max_frames}]
        )
        return emb.cpu().float().numpy()[0]

    @staticmethod
    def cosine_similarity(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        q = query / (np.linalg.norm(query) + 1e-8)
        v = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
        return v @ q
