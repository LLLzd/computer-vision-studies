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


def _resolve_video_path(video_path: str) -> str:
    """解析视频路径：支持大小写、常见扩展名。"""
    p = Path(video_path)
    if p.is_file():
        return str(p.resolve())
    parent = p.parent
    stem = p.stem
    if parent.is_dir():
        for ext in (".MOV", ".mov", ".mp4", ".MP4", ".m4v"):
            cand = parent / f"{stem}{ext}"
            if cand.is_file():
                return str(cand.resolve())
    raise RuntimeError(f"视频文件不存在: {video_path}")


def extract_frames(
    video_path: str,
    output_dir: str,
    stride: int = 2,
    max_frames: int = 220,
    resize_width: int = 960,
    sharpness_threshold: float = 15.0,
) -> int:
    """
    sharpness_threshold:
        > 0  时仅保留拉普拉斯方差 >= 该值的帧（过滤糊帧）。
        <= 0 时关闭清晰度过滤，按 stride 全部保留（适合调试或极暗/极糊素材）。
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    resolved = _resolve_video_path(video_path)
    cap = cv2.VideoCapture(resolved)
    if not cap.isOpened():
        raise RuntimeError(f"视频打不开: {resolved}")

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

                sharp = laplacian_sharpness(frame)
                keep = sharpness_threshold <= 0 or sharp >= sharpness_threshold
                if keep:
                    out = Path(output_dir) / f"frame_{write_idx:05d}.jpg"
                    cv2.imwrite(str(out), frame)
                    write_idx += 1
                    saved += 1

                if saved >= max_frames:
                    break

            idx += 1
            pbar.update(1)

    cap.release()

    if saved == 0 and sharpness_threshold > 0:
        print(
            "[WARN] 清晰度过滤后未保留任何帧。常见原因：运动模糊、暗光、或阈值过高。\n"
            "      请将 data.sharpness_threshold 调低（如 8~15），或在命令行传 "
            "--sharpness_threshold 0 关闭过滤后重试。"
        )
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="从手机视频提取训练图像帧（带清晰度过滤）")
    parser.add_argument("--video", required=True, help="输入视频（MOV/MP4）")
    parser.add_argument("--output", required=True, help="输出帧目录")
    parser.add_argument("--stride", type=int, default=2, help="每隔多少帧取1帧")
    parser.add_argument("--max_frames", type=int, default=220, help="最多保留帧数")
    parser.add_argument("--resize_width", type=int, default=960, help="统一缩放宽度")
    parser.add_argument(
        "--sharpness_threshold",
        type=float,
        default=15.0,
        help="清晰度阈值；<=0 表示不过滤，全部按 stride 保存",
    )
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
