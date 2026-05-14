#!/usr/bin/env python3
"""
按「输入视频」的帧率与总帧数，生成左右对比 MP4：
  左：原视频当前帧
  右：同一相机下 splat **神经渲染**（与训练时相近的像素分辨率），再双线性放大到与左图相同尺寸；
      不是点云、不是高斯中心可视化。若在整幅 1080p 上直接 splat，可见 splat 数上限会导致画面像稀疏「点云」。

说明：
  - COLMAP 只对抽帧后的 JPG 有位姿；原 MOV 每一帧用时间比例在排序后的相机之间做 SE(3) 插值（平移线性 + 旋转 slerp），轨迹与拍摄顺序对齐。
  - 内参从 COLMAP 图像尺寸缩放到当前视频帧尺寸。
  - 默认使用 checkpoints 里 step 号最大的模型（可通过 --checkpoint 指定）。
  - --frame_stride N：每 N 帧取 1 帧渲染（加速）；输出 fps 自动设为 fps*M/n_total，
    使播放时长与原始视频一致（M 为实际写入帧数）。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp
from tqdm import tqdm

from core.gaussian_model import GaussianModel
from core.renderer import render_view, resize_intrinsics
from train import TrainFrame, build_train_frames, load_config
from utils.device_utils import pick_torch_device
from utils.natural_sort import natural_sort_key as _natural_sort_key


def load_model_from_ckpt(ckpt_path: str, device: torch.device) -> GaussianModel:
    ckpt = torch.load(ckpt_path, map_location=device)
    xyz = ckpt["xyz"].to(device)
    rgb = torch.sigmoid(ckpt["color_logits"]).to(device)
    model = GaussianModel(xyz=xyz, rgb=rgb)
    model.log_scales.data = ckpt["log_scales"].to(device)
    model.color_logits.data = ckpt["color_logits"].to(device)
    model.opacity_logits.data = ckpt["opacity_logits"].to(device)
    return model


def w2c_to_c2w(w: np.ndarray) -> np.ndarray:
    """world2cam 4x4 -> cam2world 4x4（列向量约定）。"""
    w = w.astype(np.float64)
    r, t = w[:3, :3], w[:3, 3]
    c = np.eye(4, dtype=np.float64)
    c[:3, :3] = r.T
    c[:3, 3] = (-r.T @ t).reshape(3)
    return c


def c2w_to_w2c(c: np.ndarray) -> np.ndarray:
    """cam2world -> world2cam。"""
    c = c.astype(np.float64)
    r, t = c[:3, :3], c[:3, 3]
    w = np.eye(4, dtype=np.float64)
    w[:3, :3] = r.T
    w[:3, 3] = (-r.T @ t).reshape(3)
    return w


def interpolate_w2c(frames: list[TrainFrame], alpha: float) -> np.ndarray:
    """alpha∈[0,1]，在按文件名排序的相机序列上插值 world2cam。"""
    n = len(frames)
    if n == 0:
        raise RuntimeError("无可用相机帧")
    if n == 1:
        return frames[0].w2c.copy()
    a = float(np.clip(alpha, 0.0, 1.0)) * (n - 1)
    i0 = int(np.floor(a))
    i1 = int(np.ceil(a))
    t = a - i0
    i0 = min(max(i0, 0), n - 1)
    i1 = min(max(i1, 0), n - 1)
    if i0 == i1:
        return frames[i0].w2c.copy()

    c0 = w2c_to_c2w(frames[i0].w2c)
    c1 = w2c_to_c2w(frames[i1].w2c)
    r0, t0 = c0[:3, :3], c0[:3, 3]
    r1, t1 = c1[:3, :3], c1[:3, 3]
    t_mid = (1.0 - t) * t0 + t * t1
    key_times = [0.0, 1.0]
    key_rots = Rotation.from_matrix(np.stack([r0, r1], axis=0))
    slerp = Slerp(key_times, key_rots)
    r_mid = slerp(t).as_matrix()
    c_mid = np.eye(4, dtype=np.float64)
    c_mid[:3, :3] = r_mid
    c_mid[:3, 3] = t_mid
    return c2w_to_w2c(c_mid).astype(np.float32)


def latest_checkpoint(ckpt_dir: Path) -> Path:
    files = sorted(ckpt_dir.glob("step_*.pt"))
    if not files:
        raise RuntimeError(f"未找到 checkpoint: {ckpt_dir}")

    def step_key(p: Path) -> int:
        m = re.search(r"step_(\d+)\.pt$", p.name)
        return int(m.group(1)) if m else -1

    return max(files, key=step_key)


@torch.no_grad()
def run(
    config_path: str,
    checkpoint: str | None,
    video_path: str | None,
    output_path: str,
    render_height: int | None,
    frame_stride: int = 1,
) -> None:
    cfg = load_config(config_path)
    device = pick_torch_device()
    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
    ckpt_path = Path(checkpoint) if checkpoint else latest_checkpoint(ckpt_dir)
    print(f"[INFO] checkpoint: {ckpt_path}")

    vid = Path(video_path or cfg["paths"]["input_video"])
    if not vid.is_file():
        raise FileNotFoundError(f"视频不存在: {vid}")

    frames = sorted(
        build_train_frames(cfg["paths"]["colmap_text_model"], cfg["paths"]["frames_dir"]),
        key=lambda f: _natural_sort_key(f.image_path),
    )
    print(f"[INFO] COLMAP 相机数: {len(frames)}")

    ref = frames[0]
    mode = cfg["train"]["mode"]
    if render_height is None:
        rh = cfg["train"]["render_height_standard"] if mode == "standard" else cfg["train"]["render_height_quick"]
    else:
        rh = render_height
    max_vis = cfg["train"]["max_visible_splats_standard"] if mode == "standard" else cfg["train"]["max_visible_splats_quick"]
    if cfg["train"].get("low_memory"):
        max_vis = int(max_vis * 0.8)
        rh = int(rh * 0.9)
    rmax = 10.0 if mode == "standard" else 8.0

    cap = cv2.VideoCapture(str(vid))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {vid}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-3:
        fps = 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] 视频: {n_total} 帧, {fps:.3f} fps, {vw}x{vh}")

    # 与训练 / preview 一致：在较小分辨率上 splat，再放大到视频尺寸；否则全幅面仅 K 个 splat 会像稀疏点云。
    render_w = max(1, int(round(rh * float(vw) / float(vh))))
    render_h = rh
    print(f"[INFO] 右侧为 splat 神经渲染 {render_w}x{render_h}，双线性放大至 {vw}x{vh} 与左侧对齐")

    model = load_model_from_ckpt(str(ckpt_path), device)
    model.eval()

    stride = max(1, int(frame_stride))
    indices = list(range(0, max(n_total, 1), stride))
    if not indices:
        indices = [0]
    m_out = len(indices)
    # 使 M / out_fps == n_total / fps，播放时长与源一致
    out_fps = float(fps) * float(m_out) / max(float(n_total), 1.0)
    print(f"[INFO] frame_stride={stride}，写入 {m_out} 帧，输出 fps={out_fps:.4f}（时长≈{n_total/fps:.2f}s）")

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_p), fourcc, out_fps, (vw * 2, vh))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建 VideoWriter: {out_p}")

    for i in tqdm(range(n_total), desc="trajectory_compare"):
        ok, bgr = cap.read()
        if not ok:
            break
        if i % stride != 0:
            continue
        alpha = i / max(n_total - 1, 1)
        w2c_np = interpolate_w2c(frames, alpha)
        w2c = torch.from_numpy(w2c_np).to(device)

        fx, fy, cx, cy = resize_intrinsics(
            *ref.intrinsics, src_w=ref.width, src_h=ref.height, dst_w=render_w, dst_h=render_h
        )
        pred = render_view(
            model=model,
            w2c=w2c,
            intrinsics=(fx, fy, cx, cy),
            image_size=(render_h, render_w),
            max_visible_splats=max_vis,
            radius_px_max=rmax,
            bg_color=1.0,
        )
        pr = (pred.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        pr_bgr = cv2.cvtColor(pr, cv2.COLOR_RGB2BGR)
        pr_bgr = cv2.resize(pr_bgr, (vw, vh), interpolation=cv2.INTER_LINEAR)
        panel = np.concatenate([bgr, pr_bgr], axis=1)
        writer.write(panel)

    cap.release()
    writer.release()
    print(f"[DONE] 已写入: {out_p.resolve()}")


def main() -> None:
    p = argparse.ArgumentParser(description="原视频 | 渲染 等时长轨迹对比 MP4")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--checkpoint", default=None, help="默认: checkpoints 中 step 最大的 .pt")
    p.add_argument("--video", default=None, help="默认: config.paths.input_video")
    p.add_argument(
        "--output",
        default="output/previews/trajectory_compare_full.mp4",
        help="输出并排视频路径",
    )
    p.add_argument("--render_height", type=int, default=None, help="内部 splat 渲染高度（默认按 config train.mode），再放大到视频尺寸")
    p.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="每 N 帧取 1 帧（如 30 约 30 倍加速）；输出 fps 自动调整，总时长与源视频一致",
    )
    args = p.parse_args()
    run(
        args.config,
        args.checkpoint,
        args.video,
        args.output,
        args.render_height,
        frame_stride=args.frame_stride,
    )


if __name__ == "__main__":
    main()
