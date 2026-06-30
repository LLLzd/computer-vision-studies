#!/usr/bin/env python3
"""Create a short synthetic test video with a moving marker."""

from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "inputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test_moving_ball.mp4"

    width, height = 640, 360
    fps = 30.0
    num_frames = 90

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    for t in range(num_frames):
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        x = int((t / (num_frames - 1)) * (width - 80) + 40)
        y = height // 2
        cv2.circle(frame, (x, y), 28, (0, 220, 255), -1)
        cv2.putText(
            frame,
            f"frame {t + 1}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)

    writer.release()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
