from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from utils.config_utils import load_config


def run(cmd: list[str]) -> None:
    print(f"\n[RUN] {' '.join(cmd)}")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"命令失败: {' '.join(cmd)}")


def latest_checkpoint(ckpt_dir: str) -> str:
    files = sorted(Path(ckpt_dir).glob("step_*.pt"))
    if not files:
        raise RuntimeError(f"未找到 checkpoint: {ckpt_dir}")
    return str(files[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description="一键执行: 抽帧 -> COLMAP -> 训练 -> 预览 -> 导出")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--skip_extract", action="store_true")
    parser.add_argument("--skip_colmap", action="store_true")
    parser.add_argument("--skip_preview", action="store_true")
    parser.add_argument("--skip_export", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    py = sys.executable

    if not args.skip_extract:
        run(
            [
                py,
                "src/preprocess/extract_frames.py",
                "--video",
                cfg["paths"]["input_video"],
                "--output",
                cfg["paths"]["frames_dir"],
                "--stride",
                str(cfg["data"]["frame_stride"]),
                "--max_frames",
                str(cfg["data"]["max_frames"]),
                "--resize_width",
                str(cfg["data"]["resize_width"]),
                "--sharpness_threshold",
                str(cfg["data"]["sharpness_threshold"]),
            ]
        )

    if not args.skip_colmap:
        run(
            [
                py,
                "src/preprocess/run_colmap.py",
                "--images",
                cfg["paths"]["frames_dir"],
                "--workspace",
                cfg["paths"]["colmap_workspace"],
            ]
        )

    run([py, "src/train.py", "--config", args.config])
    ckpt = latest_checkpoint(cfg["paths"]["checkpoints_dir"])
    print(f"[INFO] latest checkpoint: {ckpt}")

    if not args.skip_preview:
        run(
            [
                py,
                "src/preview.py",
                "--config",
                args.config,
                "--checkpoint",
                ckpt,
                "--output",
                str(Path(cfg["paths"]["preview_dir"]) / "compare.mp4"),
                "--frames",
                str(cfg["preview"]["frames"]),
                "--fps",
                str(cfg["preview"]["fps"]),
            ]
        )

    if not args.skip_export:
        run([py, "src/export_model.py", "--checkpoint", ckpt, "--export_dir", cfg["paths"]["export_dir"]])

    print("\n[DONE] 全流程完成。")


if __name__ == "__main__":
    main()
