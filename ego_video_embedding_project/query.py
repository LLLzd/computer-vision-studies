#!/usr/bin/env python3
"""
在向量库中检索与文本查询最相似的视频帧。

用法:
    python query.py "我在厨房拿杯子"
    python query.py "打开冰箱" --top-k 10 --index outputs/index
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_INDEX_DIR, TOP_K_DEFAULT  # noqa: E402
from src.embedder.wrapper import EmbeddingEngine  # noqa: E402
from src.vector_store.faiss_store import VectorStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="文本检索第一视角视频帧")
    parser.add_argument("query", help="检索文本")
    parser.add_argument(
        "--index",
        default=str(DEFAULT_INDEX_DIR),
        help=f"向量库目录（默认: {DEFAULT_INDEX_DIR}）",
    )
    parser.add_argument("--top-k", type=int, default=TOP_K_DEFAULT, help="返回 Top-K 结果")
    parser.add_argument("--model", default=None, help="模型路径或 HuggingFace ID")
    return parser.parse_args()


def format_timestamp(sec: float) -> str:
    if sec < 0:
        return "full_video"
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def main() -> None:
    args = parse_args()
    index_dir = Path(args.index)

    store = VectorStore.load(index_dir)
    engine = EmbeddingEngine(model_path=args.model)

    print(f"查询: {args.query}")
    print(f"向量库: {index_dir}（{store.size} 条）\n")

    query_vec = engine.embed_text(args.query)
    results = store.search(query_vec, top_k=args.top_k)

    if not results:
        print("未找到结果（向量库为空？）")
        return

    for rank, hit in enumerate(results, start=1):
        ts = format_timestamp(hit["timestamp_sec"])
        path_info = hit["saved_path"] or hit["source_video"]
        print(
            f"#{rank}  score={hit['score']:.4f}  "
            f"time={ts}  frame={hit['frame_index']}  "
            f"path={path_info}"
        )


if __name__ == "__main__":
    main()
