#!/usr/bin/env python3
"""
将第一视角视频抽帧 → Qwen3-VL Embedding → 写入 FAISS 向量库。

用法:
    python build_index.py -i inputs/my_video.mp4
    python build_index.py -i inputs/my_video.mp4 --interval 0.5 --max-edge 512
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    DEFAULT_INDEX_DIR,
    DEFAULT_FRAMES_DIR,
    EMBEDDING_DIM,
    FRAME_INTERVAL_SEC,
    MAX_FRAME_EDGE,
    MAX_FRAMES,
    RESIZE_MODE,
)
from src.embedder.wrapper import EmbeddingEngine  # noqa: E402
from src.preprocessing.video import extract_frames  # noqa: E402
from src.vector_store.faiss_store import VectorStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建第一视角视频 Embedding 向量库")
    parser.add_argument("-i", "--input", required=True, help="输入视频路径")
    parser.add_argument(
        "-o",
        "--output",
        default=str(DEFAULT_INDEX_DIR),
        help=f"向量库输出目录（默认: {DEFAULT_INDEX_DIR}）",
    )
    parser.add_argument(
        "--frames-dir",
        default=str(DEFAULT_FRAMES_DIR),
        help=f"预处理帧保存目录（默认: {DEFAULT_FRAMES_DIR}）",
    )
    parser.add_argument("--interval", type=float, default=FRAME_INTERVAL_SEC, help="抽帧间隔（秒）")
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES, help="最大帧数")
    parser.add_argument("--max-edge", type=int, default=MAX_FRAME_EDGE, help="帧最长边像素")
    parser.add_argument(
        "--resize-mode",
        choices=["letterbox", "center_crop", "stretch"],
        default=RESIZE_MODE,
        help="缩放模式",
    )
    parser.add_argument("--model", default=None, help="模型路径或 HuggingFace ID")
    parser.add_argument("--batch-size", type=int, default=1, help="Embedding 批大小")
    parser.add_argument(
        "--mode",
        choices=["frames", "video"],
        default="frames",
        help="frames=逐帧编码（推荐）; video=整段视频一次编码",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.input)
    index_dir = Path(args.output)
    frames_dir = Path(args.frames_dir)

    if not video_path.exists():
        print(f"错误: 视频不存在 {video_path}")
        sys.exit(1)

    engine = EmbeddingEngine(model_path=args.model, batch_size=args.batch_size)
    store = VectorStore(dim=EMBEDDING_DIM, index_dir=index_dir)

    if args.mode == "video":
        print("模式: 整段视频编码")
        vec = engine.embed_video_file(video_path, fps=1.0 / max(args.interval, 0.01), max_frames=args.max_frames)
        store.add(
            vec.reshape(1, -1),
            [
                {
                    "frame_index": -1,
                    "timestamp_sec": 0.0,
                    "source_video": str(video_path.resolve()),
                    "saved_path": None,
                    "caption": "full_video",
                }
            ],
        )
    else:
        print("模式: 逐帧编码")
        records = extract_frames(
            video_path=video_path,
            output_frames_dir=frames_dir,
            interval_sec=args.interval,
            max_frames=args.max_frames,
            max_edge=args.max_edge,
            resize_mode=args.resize_mode,
        )
        if not records:
            print("错误: 未能抽取任何帧")
            sys.exit(1)

        print(f"共抽取 {len(records)} 帧，开始 Embedding...")
        images = [r.image for r in records]
        vectors = engine.embed_images(images)

        metas = [
            {
                "frame_index": r.frame_index,
                "timestamp_sec": r.timestamp_sec,
                "source_video": r.source_video,
                "saved_path": r.saved_path,
            }
            for r in records
        ]
        store.add(vectors, metas)

    store.save()
    print(f"向量库已保存: {index_dir}（共 {store.size} 条）")


if __name__ == "__main__":
    main()
