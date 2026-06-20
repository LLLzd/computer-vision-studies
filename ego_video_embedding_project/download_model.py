#!/usr/bin/env python3
"""下载 Qwen3-VL-Embedding-2B 到本地 models/ 目录。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import MODEL_HF_ID, MODEL_LOCAL_DIR, MODELS_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载 Qwen3-VL Embedding 模型")
    parser.add_argument(
        "--repo",
        default=MODEL_HF_ID,
        help=f"HuggingFace 模型 ID（默认: {MODEL_HF_ID}）",
    )
    parser.add_argument(
        "--output",
        default=str(MODEL_LOCAL_DIR),
        help=f"本地保存路径（默认: {MODEL_LOCAL_DIR}）",
    )
    parser.add_argument(
        "--source",
        choices=["hf", "modelscope"],
        default="hf",
        help="下载源: hf=HuggingFace, modelscope=ModelScope（国内更快）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"模型目录已存在: {output_dir}")
        return

    print(f"下载 {args.repo} -> {output_dir}")

    if args.source == "modelscope":
        from modelscope import snapshot_download

        snapshot_download(args.repo, local_dir=str(output_dir))
    else:
        from huggingface_hub import snapshot_download

        snapshot_download(args.repo, local_dir=str(output_dir))

    print("下载完成")


if __name__ == "__main__":
    main()
