#!/usr/bin/env python3
"""
从视频中提取帧用于3DGS重建
"""

import os
import cv2
import argparse
import numpy as np
from pathlib import Path


def extract_frames(video_path, output_dir, sample_rate=5, max_frames=100, resize_factor=1.0):
    """
    从视频中提取帧

    参数：
        video_path: str - 视频文件路径
        output_dir: str - 输出目录
        sample_rate: int - 每隔多少帧提取一次
        max_frames: int - 最大帧数
        resize_factor: float - 图像缩放因子
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video info: {width}x{height}, {fps:.1f}fps, {total_frames} frames")

    frame_paths = []
    frame_idx = 0
    extracted = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_rate == 0 and extracted < max_frames:
            # 缩放图像
            if resize_factor != 1.0:
                frame = cv2.resize(
                    frame, (int(width * resize_factor), int(height * resize_factor))
                )

            # 保存帧
            frame_path = os.path.join(output_dir, f"frame_{extracted:04d}.png")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            extracted += 1

            if extracted % 10 == 0:
                print(f"Extracted {extracted} frames...")

        frame_idx += 1

    cap.release()
    print(f"\nExtracted {len(frame_paths)} frames to {output_dir}")

    return frame_paths


def main():
    parser = argparse.ArgumentParser(description="Extract frames from video for 3DGS")
    parser.add_argument("--video", "-i", type=str, default="IMG_7834.MOV",
                        help="Input video path")
    parser.add_argument("--output", "-o", type=str, default="frames",
                        help="Output directory for frames")
    parser.add_argument("--sample_rate", "-r", type=int, default=5,
                        help="Sample every N frames")
    parser.add_argument("--max_frames", "-m", type=int, default=100,
                        help="Maximum number of frames to extract")
    parser.add_argument("--resize", "-s", type=float, default=0.5,
                        help="Resize factor for frames (0.5 = half size)")

    args = parser.parse_args()

    video_path = args.video
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return

    frame_paths = extract_frames(
        video_path,
        args.output,
        sample_rate=args.sample_rate,
        max_frames=args.max_frames,
        resize_factor=args.resize
    )

    print(f"\nExtracted frames saved to: {os.path.abspath(args.output)}")
    print(f"Total frames: {len(frame_paths)}")


if __name__ == "__main__":
    main()