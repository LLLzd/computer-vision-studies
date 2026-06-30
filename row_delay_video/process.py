#!/usr/bin/env python3
"""Generate a row-delayed video from a normal video.

Rows are grouped into bands; each band lags one frame behind the band above.
This produces a horizontal ripple instead of per-line chaos on tall videos.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def read_video(path: Path) -> tuple[list[np.ndarray], float, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)

    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from: {path}")

    return frames, fps, width, height


def max_row_delay(height: int, row_step: int) -> int:
    num_bands = (height + row_step - 1) // row_step
    return max(0, num_bands - 1)


def clamp_source_frame(src_t: int, last_idx: int, fill_mode: str) -> int | None:
    if src_t < 0:
        return None if fill_mode == "black" else 0
    if src_t > last_idx:
        return None if fill_mode == "black" else last_idx
    return src_t


def compose_row_delayed_frame(
    frames: list[np.ndarray],
    t: int,
    height: int,
    width: int,
    row_step: int,
    fill_mode: str,
) -> np.ndarray:
    """Build output frame t by stitching row bands from staggered source frames."""
    channels = frames[0].shape[2] if frames[0].ndim == 3 else 1
    if fill_mode == "black":
        out = np.zeros((height, width, channels), dtype=np.uint8)
    else:
        out = np.empty((height, width, channels), dtype=np.uint8)

    last_idx = len(frames) - 1
    num_bands = (height + row_step - 1) // row_step

    for band in range(num_bands):
        r_start = band * row_step
        r_end = min((band + 1) * row_step, height)
        src_t = clamp_source_frame(t - band, last_idx, fill_mode)
        if src_t is None:
            continue
        out[r_start:r_end] = frames[src_t][r_start:r_end]

    return out


def process_video(
    input_path: Path,
    output_path: Path,
    row_step: int = 10,
    fill_mode: str = "clamp",
) -> dict:
    if row_step < 1:
        raise ValueError("row_step must be >= 1")

    frames, fps, width, height = read_video(input_path)
    num_frames = len(frames)
    delay_span = max_row_delay(height, row_step)
    num_output_frames = num_frames + delay_span

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {output_path}")

    for t in tqdm(range(num_output_frames), desc="Rendering", unit="frame"):
        out_frame = compose_row_delayed_frame(
            frames, t, height, width, row_step, fill_mode
        )
        writer.write(out_frame)

    writer.release()

    num_bands = (height + row_step - 1) // row_step
    return {
        "input_frames": num_frames,
        "output_frames": num_output_frames,
        "height": height,
        "width": width,
        "fps": fps,
        "row_step": row_step,
        "num_bands": num_bands,
        "delay_span_frames": delay_span,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a row-delayed video with horizontal ripple bands. "
            "Every N rows share the same time offset; each band lags 1 frame."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the source video",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output video path (default: outputs/<input_stem>_row_delay.mp4)",
    )
    parser.add_argument(
        "--row-step",
        type=int,
        default=10,
        help=(
            "Number of consecutive rows that share the same frame delay "
            "(default: 10). Larger values = smoother, slower ripple."
        ),
    )
    parser.add_argument(
        "--fill-mode",
        choices=("clamp", "black"),
        default="clamp",
        help=(
            "How to fill rows when the required source frame is unavailable: "
            "'clamp' repeats first/last frame, 'black' uses black pixels."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input video not found: {input_path}")

    if args.output is None:
        suffix = f"_row_delay_s{args.row_step}"
        output_path = (
            Path(__file__).resolve().parent / "outputs" / f"{input_path.stem}{suffix}.mp4"
        )
    else:
        output_path = args.output.expanduser().resolve()

    stats = process_video(
        input_path,
        output_path,
        row_step=args.row_step,
        fill_mode=args.fill_mode,
    )

    print("\nDone.")
    print(f"  Input:      {input_path}")
    print(f"  Output:     {output_path}")
    print(f"  Size:       {stats['width']}x{stats['height']}")
    print(f"  FPS:        {stats['fps']:.3f}")
    print(f"  Row step:   {stats['row_step']} rows / 1 frame delay")
    print(f"  Bands:      {stats['num_bands']}")
    print(
        f"  Frames:     {stats['input_frames']} -> {stats['output_frames']} "
        f"(+{stats['delay_span_frames']})"
    )


if __name__ == "__main__":
    main()
