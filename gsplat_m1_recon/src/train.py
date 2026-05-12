from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from core.gaussian_model import GaussianModel
from core.renderer import render_view, resize_intrinsics
from utils.colmap_io import qvec_to_rotmat, read_cameras_txt, read_images_txt, read_points3d_txt
from utils.config_utils import ensure_dirs, load_config
from utils.device_utils import memory_report_gb, pick_torch_device


@dataclass
class TrainFrame:
    image_path: Path
    w2c: np.ndarray
    intrinsics: tuple[float, float, float, float]
    width: int
    height: int


def build_train_frames(colmap_txt_dir: str, frames_dir: str) -> list[TrainFrame]:
    colmap_dir = Path(colmap_txt_dir)
    cameras = read_cameras_txt(colmap_dir / "cameras.txt")
    images = read_images_txt(colmap_dir / "images.txt")
    frames_root = Path(frames_dir)

    items: list[TrainFrame] = []
    for image in images.values():
        cam = cameras[image.camera_id]
        r = qvec_to_rotmat(image.qvec)
        t = image.tvec.reshape(3, 1)
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = r
        w2c[:3, 3:] = t

        if cam.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
            f, cx, cy = float(cam.params[0]), float(cam.params[1]), float(cam.params[2])
            fx = fy = f
        else:
            fx, fy, cx, cy = map(float, cam.params[:4])

        img_path = frames_root / image.name
        if not img_path.exists():
            continue
        items.append(
            TrainFrame(
                image_path=img_path,
                w2c=w2c,
                intrinsics=(fx, fy, cx, cy),
                width=cam.width,
                height=cam.height,
            )
        )
    if not items:
        raise RuntimeError("没有读取到可用训练帧。请确认 frames 目录和 COLMAP 模型对应。")
    return items


def load_target_image(path: Path, size_hw: tuple[int, int], device: torch.device) -> torch.Tensor:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"读取图像失败: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = size_hw
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).to(device)


def init_gaussians(points_xyz: np.ndarray, points_rgb: np.ndarray, max_gaussians: int, device: torch.device) -> GaussianModel:
    n = points_xyz.shape[0]
    if n > max_gaussians:
        sel = np.random.choice(n, max_gaussians, replace=False)
        points_xyz = points_xyz[sel]
        points_rgb = points_rgb[sel]
    xyz = torch.from_numpy(points_xyz).to(device)
    rgb = torch.from_numpy(points_rgb).to(device)
    return GaussianModel(xyz=xyz, rgb=rgb)


def save_checkpoint(model: GaussianModel, step: int, ckpt_dir: Path) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out = ckpt_dir / f"step_{step:06d}.pt"
    torch.save(
        {
            "step": step,
            "xyz": model.xyz.detach().cpu(),
            "log_scales": model.log_scales.detach().cpu(),
            "color_logits": model.color_logits.detach().cpu(),
            "opacity_logits": model.opacity_logits.detach().cpu(),
        },
        out,
    )
    return out


def train(config_path: str) -> None:
    cfg = load_config(config_path)
    random.seed(cfg["train"]["seed"])
    np.random.seed(cfg["train"]["seed"])
    torch.manual_seed(cfg["train"]["seed"])

    device = pick_torch_device()
    print(f"[INFO] device={device}, memory={memory_report_gb()}")

    mode = cfg["train"]["mode"]
    if mode not in {"quick", "standard"}:
        raise ValueError("train.mode 必须是 quick 或 standard")

    max_gaussians = cfg["train"]["max_gaussians_quick"] if mode == "quick" else cfg["train"]["max_gaussians_standard"]
    max_visible_splats = cfg["train"]["max_visible_splats_quick"] if mode == "quick" else cfg["train"]["max_visible_splats_standard"]
    render_h = cfg["train"]["render_height_quick"] if mode == "quick" else cfg["train"]["render_height_standard"]
    iterations = cfg["train"]["iterations_quick"] if mode == "quick" else cfg["train"]["iterations_standard"]
    if cfg["train"]["low_memory"]:
        max_visible_splats = int(max_visible_splats * 0.8)
        render_h = int(render_h * 0.9)

    ensure_dirs([cfg["paths"]["checkpoints_dir"], cfg["paths"]["preview_dir"], cfg["paths"]["export_dir"]])

    frames = build_train_frames(cfg["paths"]["colmap_text_model"], cfg["paths"]["frames_dir"])
    points_xyz, points_rgb = read_points3d_txt(Path(cfg["paths"]["colmap_text_model"]) / "points3D.txt")
    model = init_gaussians(points_xyz, points_rgb, max_gaussians=max_gaussians, device=device)
    print(f"[INFO] total gaussians: {model.num_gaussians}")

    optimizer = torch.optim.Adam(
        [
            {"params": [model.xyz], "lr": cfg["train"]["lr"]},
            {"params": [model.log_scales], "lr": cfg["train"]["lr"] * 0.5},
            {"params": [model.color_logits], "lr": cfg["train"]["lr"] * 0.2},
            {"params": [model.opacity_logits], "lr": cfg["train"]["lr"] * 0.5},
        ]
    )

    pbar = tqdm(range(1, iterations + 1), desc=f"train-{mode}")
    for step in pbar:
        frm = random.choice(frames)
        aspect = frm.width / frm.height
        render_w = int(render_h * aspect)
        size_hw = (render_h, render_w)

        fx, fy, cx, cy = resize_intrinsics(*frm.intrinsics, src_w=frm.width, src_h=frm.height, dst_w=render_w, dst_h=render_h)
        target = load_target_image(frm.image_path, size_hw=size_hw, device=device)
        w2c = torch.from_numpy(frm.w2c).to(device)

        pred = render_view(
            model=model,
            w2c=w2c,
            intrinsics=(fx, fy, cx, cy),
            image_size=size_hw,
            max_visible_splats=max_visible_splats,
            radius_px_max=10.0 if mode == "standard" else 8.0,
            bg_color=1.0,
        )

        l1 = (pred - target).abs().mean()
        mse = ((pred - target) ** 2).mean()
        reg = model.scales().mean() * 0.001 + model.opacities().mean() * 0.0005
        loss = 0.75 * l1 + 0.25 * mse + reg

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        pbar.set_postfix(loss=f"{loss.item():.5f}", mem=memory_report_gb())

        if step % cfg["train"]["save_every"] == 0 or step == iterations:
            ckpt = save_checkpoint(model, step=step, ckpt_dir=Path(cfg["paths"]["checkpoints_dir"]))
            print(f"[INFO] checkpoint: {ckpt}")

        if step % cfg["train"]["preview_every"] == 0 or step == iterations:
            preview = (pred.detach().cpu().numpy() * 255).astype(np.uint8)
            out = Path(cfg["paths"]["preview_dir"]) / f"preview_{step:06d}.jpg"
            cv2.imwrite(str(out), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))

    print(f"[DONE] training complete, checkpoints at {cfg['paths']['checkpoints_dir']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="3DGS 训练（M1/MPS 优化版）")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
