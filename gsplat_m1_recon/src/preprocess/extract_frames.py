from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def laplacian_sharpness(img_bgr: np.ndarray) -> float:
    """用拉普拉斯方差估计清晰度，值越大越清晰。"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_frames(
    video_path: str,
    output_dir: str,
    stride: int = 2,
    max_frames: int = 220,
    resize_width: int = 960,
    sharpness_threshold: float = 35.0,
) -> int:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"视频打不开: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved = 0
    idx = 0
    write_idx = 0

    with tqdm(total=total_frames, desc="抽帧中") as pbar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if idx % stride == 0:
                h, w = frame.shape[:2]
                if resize_width > 0 and w != resize_width:
                    new_h = int(h * (resize_width / w))
                    frame = cv2.resize(frame, (resize_width, new_h), interpolation=cv2.INTER_AREA)

                if laplacian_sharpness(frame) >= sharpness_threshold:
                    out = Path(output_dir) / f"frame_{write_idx:05d}.jpg"
                    cv2.imwrite(str(out), frame)
                    write_idx += 1
                    saved += 1

                if saved >= max_frames:
                    break

            idx += 1
            pbar.update(1)

    cap.release()
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="从手机视频提取训练图像帧（带清晰度过滤）")
    parser.add_argument("--video", required=True, help="输入视频（MOV/MP4）")
    parser.add_argument("--output", required=True, help="输出帧目录")
    parser.add_argument("--stride", type=int, default=2, help="每隔多少帧取1帧")
    parser.add_argument("--max_frames", type=int, default=220, help="最多保留帧数")
    parser.add_argument("--resize_width", type=int, default=960, help="统一缩放宽度")
    parser.add_argument("--sharpness_threshold", type=float, default=35.0, help="清晰度阈值")
    args = parser.parse_args()

    saved = extract_frames(
        video_path=args.video,
        output_dir=args.output,
        stride=args.stride,
        max_frames=args.max_frames,
        resize_width=args.resize_width,
        sharpness_threshold=args.sharpness_threshold,
    )
    print(f"抽帧完成，保留 {saved} 张图像，输出目录: {args.output}")


if __name__ == "__main__":
    main()
