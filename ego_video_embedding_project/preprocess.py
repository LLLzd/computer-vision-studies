#!/usr/bin/env python3
"""
仅做视频预处理（抽帧 + 缩放），不加载 Embedding 模型。

用法:
    python preprocess.py -i inputs/my_video.mp4
    python preprocess.py -i inputs/my_video.mp4 -o outputs/frames --interval 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    DEFAULT_FRAMES_DIR,
    FRAME_INTERVAL_SEC,
    MAX_FRAME_EDGE,
    MAX_FRAMES,
    RESIZE_MODE,
)
from src.preprocessing.video import extract_frames  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="第一视角视频预处理（抽帧）")
    parser.add_argument("-i", "--input", required=True, help="输入视频路径")
    parser.add_argument(
        "-o",
        "--output",
        default=str(DEFAULT_FRAMES_DIR),
        help=f"输出帧目录（默认: {DEFAULT_FRAMES_DIR}）",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.input)
    output_dir = Path(args.output)

    if not video_path.exists():
        print(f"错误: 视频不存在 {video_path}")
        sys.exit(1)

    records = extract_frames(
        video_path=video_path,
        output_frames_dir=output_dir,
        interval_sec=args.interval,
        max_frames=args.max_frames,
        max_edge=args.max_edge,
        resize_mode=args.resize_mode,
    )

    manifest = [
        {
            "frame_index": r.frame_index,
            "timestamp_sec": r.timestamp_sec,
            "saved_path": r.saved_path,
            "size": list(r.image.size),
        }
        for r in records
    ]
    manifest_path = output_dir / "manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"完成: 抽取 {len(records)} 帧 -> {output_dir}")
    print(f"清单: {manifest_path}")


if __name__ == "__main__":
    main()
