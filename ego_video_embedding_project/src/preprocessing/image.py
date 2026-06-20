"""
图像预处理：缩放、裁剪、归一化等，适配 Qwen3-VL Embedding 输入。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Tuple

import cv2
import numpy as np
from PIL import Image

ResizeMode = Literal["letterbox", "center_crop", "stretch"]


def letterbox_resize(
    image: Image.Image,
    target_size: Tuple[int, int],
    fill_color: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """等比缩放后填充黑边，保持第一视角画面不变形。"""
    target_w, target_h = target_size
    src_w, src_h = image.size
    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), fill_color)
    offset = ((target_w - new_w) // 2, (target_h - new_h) // 2)
    canvas.paste(resized, offset)
    return canvas


def center_crop_resize(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """中心裁剪后缩放，适合固定宽高比的下游模型。"""
    target_w, target_h = target_size
    src_w, src_h = image.size
    target_ratio = target_w / target_h
    src_ratio = src_w / src_h

    if src_ratio > target_ratio:
        crop_h = src_h
        crop_w = int(crop_h * target_ratio)
    else:
        crop_w = src_w
        crop_h = int(crop_w / target_ratio)

    left = (src_w - crop_w) // 2
    top = (src_h - crop_h) // 2
    cropped = image.crop((left, top, left + crop_w, top + crop_h))
    return cropped.resize((target_w, target_h), Image.Resampling.LANCZOS)


def compute_target_size(width: int, height: int, max_edge: int) -> Tuple[int, int]:
    """按最长边限制计算目标尺寸（保持宽高比）。"""
    if max(width, height) <= max_edge:
        return width, height
    scale = max_edge / max(width, height)
    return max(1, int(width * scale)), max(1, int(height * scale))


def preprocess_frame(
    frame_bgr: np.ndarray,
    max_edge: int = 768,
    resize_mode: ResizeMode = "letterbox",
) -> Image.Image:
    """
    将 OpenCV BGR 帧转为 PIL RGB，并按配置缩放。

    Args:
        frame_bgr: OpenCV 读取的 BGR 数组
        max_edge: 最长边上限
        resize_mode: letterbox / center_crop / stretch
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    w, h = image.size
    target_w, target_h = compute_target_size(w, h, max_edge)

    if w == target_w and h == target_h:
        return image

    if resize_mode == "stretch":
        return image.resize((target_w, target_h), Image.Resampling.LANCZOS)
    if resize_mode == "center_crop":
        return center_crop_resize(image, (target_w, target_h))
    return letterbox_resize(image, (target_w, target_h))


def save_frame(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=95)
