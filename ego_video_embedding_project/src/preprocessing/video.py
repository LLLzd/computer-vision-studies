"""
视频预处理：从第一视角视频中均匀抽帧。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
from PIL import Image
from tqdm import tqdm

from config import (
    FRAME_INTERVAL_SEC,
    MAX_FRAMES,
    MAX_FRAME_EDGE,
    RESIZE_MODE,
    SAVE_PREPROCESSED_FRAMES,
)
from src.preprocessing.image import preprocess_frame, save_frame


@dataclass
class FrameRecord:
    """单帧元数据。"""

    frame_index: int
    timestamp_sec: float
    image: Image.Image
    source_video: str
    saved_path: Optional[str] = None


def extract_frames_from_video(
    video_path: Path,
    interval_sec: float = FRAME_INTERVAL_SEC,
    max_frames: int = MAX_FRAMES,
    max_edge: int = MAX_FRAME_EDGE,
    resize_mode: str = RESIZE_MODE,
    save_dir: Optional[Path] = None,
    show_progress: bool = True,
) -> List[FrameRecord]:
    """
    从视频中按时间间隔抽帧并预处理。

    Returns:
        FrameRecord 列表，每帧含 PIL 图像与时间戳。
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"视频不存在: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0.0

    # 计算采样时间点
    if duration_sec <= 0:
        sample_times = [0.0]
    else:
        sample_times = []
        t = 0.0
        while t <= duration_sec and len(sample_times) < max_frames:
            sample_times.append(t)
            t += interval_sec

    records: List[FrameRecord] = []
    iterator = tqdm(sample_times, desc="抽帧", disable=not show_progress)

    for ts in iterator:
        frame_idx = int(ts * fps)
        frame_idx = min(frame_idx, max(0, total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            continue

        image = preprocess_frame(
            frame_bgr,
            max_edge=max_edge,
            resize_mode=resize_mode,  # type: ignore[arg-type]
        )

        saved_path = None
        if save_dir is not None and SAVE_PREPROCESSED_FRAMES:
            out_path = save_dir / f"frame_{frame_idx:06d}_t{ts:.2f}s.jpg"
            save_frame(image, out_path)
            saved_path = str(out_path)

        records.append(
            FrameRecord(
                frame_index=frame_idx,
                timestamp_sec=round(ts, 3),
                image=image,
                source_video=str(video_path.resolve()),
                saved_path=saved_path,
            )
        )

    cap.release()
    return records


def extract_frames(
    video_path: Path,
    output_frames_dir: Optional[Path] = None,
    **kwargs,
) -> List[FrameRecord]:
    """便捷入口：抽帧并可选保存到输出目录。"""
    return extract_frames_from_video(
        video_path=video_path,
        save_dir=output_frames_dir,
        **kwargs,
    )
