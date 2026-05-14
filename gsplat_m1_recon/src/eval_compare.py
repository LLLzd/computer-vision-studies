#!/usr/bin/env python3
"""从已有 checkpoint 导出「同视角 GT|Pred|误差」与「新视角仅渲染」，无需重新训练。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from compare_vis import export_novel_yaw_views, export_same_view_eval_grid
from core.gaussian_model import GaussianModel
from train import build_train_frames, load_config
from utils.device_utils import pick_torch_device
from utils.natural_sort import natural_sort_key


def load_model_from_ckpt(ckpt_path: str, device: torch.device) -> GaussianModel:
    ckpt = torch.load(ckpt_path, map_location=device)
    xyz = ckpt["xyz"].to(device)
    rgb = torch.sigmoid(ckpt["color_logits"]).to(device)
    model = GaussianModel(xyz=xyz, rgb=rgb)
    model.log_scales.data = ckpt["log_scales"].to(device)
    model.color_logits.data = ckpt["color_logits"].to(device)
    model.opacity_logits.data = ckpt["opacity_logits"].to(device)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", default="", help="默认: output/previews/eval_from_ckpt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = pick_torch_device()
    model = load_model_from_ckpt(args.checkpoint, device)

    frames = sorted(
        build_train_frames(cfg["paths"]["colmap_text_model"], cfg["paths"]["frames_dir"]),
        key=lambda f: natural_sort_key(f.image_path),
    )
    mode = cfg["train"]["mode"]
    render_h = cfg["train"]["render_height_quick"] if mode == "quick" else cfg["train"]["render_height_standard"]
    max_vis = cfg["train"]["max_visible_splats_quick"] if mode == "quick" else cfg["train"]["max_visible_splats_standard"]
    if cfg["train"].get("low_memory"):
        max_vis = int(max_vis * 0.8)
        render_h = int(render_h * 0.9)
    rmax = 10.0 if mode == "standard" else 8.0

    out = Path(args.out_dir or (Path(cfg["paths"]["preview_dir"]) / "eval_from_ckpt"))
    eval_cfg = cfg.get("eval") or {}
    num_panels = int(eval_cfg.get("num_same_view_panels", 6))
    novel_yaws = list(eval_cfg.get("novel_yaws", [-15, -8, 0, 8, 15]))

    export_same_view_eval_grid(
        model,
        frames,
        out_dir=out / "same_view",
        device=device,
        render_h=render_h,
        max_visible_splats=max_vis,
        radius_px_max=rmax,
        max_panels=num_panels,
    )
    mid = frames[len(frames) // 2]
    export_novel_yaw_views(
        model,
        mid,
        out_dir=out / "novel_view",
        device=device,
        render_h=render_h,
        max_visible_splats=max_vis,
        radius_px_max=rmax,
        yaw_degrees=novel_yaws,
    )
    print(f"[DONE] 已写入: {out}")


if __name__ == "__main__":
    main()
