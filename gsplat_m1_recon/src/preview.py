from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm

from core.gaussian_model import GaussianModel
from core.renderer import render_view, resize_intrinsics
from train import TrainFrame, build_train_frames, load_config
from utils.device_utils import pick_torch_device


def load_model_from_ckpt(ckpt_path: str, device: torch.device) -> GaussianModel:
    ckpt = torch.load(ckpt_path, map_location=device)
    xyz = ckpt["xyz"].to(device)
    rgb = torch.sigmoid(ckpt["color_logits"]).to(device)
    model = GaussianModel(xyz=xyz, rgb=rgb)
    model.log_scales.data = ckpt["log_scales"].to(device)
    model.color_logits.data = ckpt["color_logits"].to(device)
    model.opacity_logits.data = ckpt["opacity_logits"].to(device)
    return model


def pick_nearest_frame(frames: list[TrainFrame], idx: int) -> TrainFrame:
    return frames[idx % len(frames)]


def render_preview_video(config_path: str, ckpt_path: str, out_path: str, n_frames: int = 120, fps: int = 24) -> None:
    cfg = load_config(config_path)
    device = pick_torch_device()
    model = load_model_from_ckpt(ckpt_path, device)
    model.eval()

    frames = build_train_frames(cfg["paths"]["colmap_text_model"], cfg["paths"]["frames_dir"])
    writer = imageio.get_writer(out_path, fps=fps)

    for i in tqdm(range(n_frames), desc="preview"):
        frm = pick_nearest_frame(frames, i)
        render_h = cfg["train"]["render_height_standard"]
        render_w = int(render_h * (frm.width / frm.height))
        fx, fy, cx, cy = resize_intrinsics(*frm.intrinsics, frm.width, frm.height, render_w, render_h)

        pred = render_view(
            model=model,
            w2c=torch.from_numpy(frm.w2c).to(device),
            intrinsics=(fx, fy, cx, cy),
            image_size=(render_h, render_w),
            max_visible_splats=cfg["train"]["max_visible_splats_standard"],
        )
        pred_np = (pred.detach().cpu().numpy() * 255).astype(np.uint8)

        gt = cv2.imread(str(frm.image_path))
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        gt = cv2.resize(gt, (render_w, render_h), interpolation=cv2.INTER_AREA)
        panel = np.concatenate([gt, pred_np], axis=1)
        writer.append_data(panel)

    writer.close()
    print(f"[DONE] 预览视频已生成: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 GT/Render 对比预览视频")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", required=True, help="训练 checkpoint 路径")
    parser.add_argument("--output", default="output/previews/compare.mp4")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    render_preview_video(args.config, args.checkpoint, args.output, args.frames, args.fps)


if __name__ == "__main__":
    main()
