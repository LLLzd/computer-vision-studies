"""
基于 FAISS 的向量库，存储视频帧 embedding 与元数据。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np


@dataclass
class VectorRecord:
    id: int
    frame_index: int
    timestamp_sec: float
    source_video: str
    saved_path: Optional[str] = None
    caption: Optional[str] = None


class VectorStore:
    INDEX_FILENAME = "embeddings.faiss"
    META_FILENAME = "metadata.json"

    def __init__(self, dim: int, index_dir: Path):
        self.dim = dim
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index = faiss.IndexFlatIP(dim)  # 内积（向量已 L2 归一化时等价于余弦相似度）
        self.records: List[VectorRecord] = []

    def add(self, vectors: np.ndarray, metas: List[Dict[str, Any]]) -> None:
        if vectors.ndim != 2:
            raise ValueError("vectors 必须是二维数组 [N, dim]")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"向量维度 {vectors.shape[1]} != {self.dim}")
        if len(metas) != vectors.shape[0]:
            raise ValueError("metas 数量必须与 vectors 行数一致")

        faiss.normalize_L2(vectors)
        start_id = len(self.records)
        self.index.add(vectors.astype(np.float32))

        for i, meta in enumerate(metas):
            self.records.append(
                VectorRecord(
                    id=start_id + i,
                    frame_index=int(meta["frame_index"]),
                    timestamp_sec=float(meta["timestamp_sec"]),
                    source_video=str(meta["source_video"]),
                    saved_path=meta.get("saved_path"),
                    caption=meta.get("caption"),
                )
            )

    def search(self, query: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []

        q = query.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            rec = self.records[idx]
            results.append(
                {
                    "score": float(score),
                    "id": rec.id,
                    "frame_index": rec.frame_index,
                    "timestamp_sec": rec.timestamp_sec,
                    "source_video": rec.source_video,
                    "saved_path": rec.saved_path,
                    "caption": rec.caption,
                }
            )
        return results

    def save(self) -> None:
        faiss.write_index(self.index, str(self.index_dir / self.INDEX_FILENAME))
        payload = {
            "dim": self.dim,
            "records": [asdict(r) for r in self.records],
        }
        with open(self.index_dir / self.META_FILENAME, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, index_dir: Path) -> "VectorStore":
        index_dir = Path(index_dir)
        meta_path = index_dir / cls.META_FILENAME
        index_path = index_dir / cls.INDEX_FILENAME

        if not meta_path.exists() or not index_path.exists():
            raise FileNotFoundError(f"向量库不存在: {index_dir}")

        with open(meta_path, encoding="utf-8") as f:
            payload = json.load(f)

        store = cls(dim=payload["dim"], index_dir=index_dir)
        store.index = faiss.read_index(str(index_path))
        store.records = [VectorRecord(**r) for r in payload["records"]]
        return store

    @property
    def size(self) -> int:
        return self.index.ntotal
