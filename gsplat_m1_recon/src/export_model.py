from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def load_ckpt(path: str):
    ckpt = torch.load(path, map_location="cpu")
    xyz = ckpt["xyz"].numpy()
    scales = torch.exp(ckpt["log_scales"]).numpy()
    colors = torch.sigmoid(ckpt["color_logits"]).numpy()
    opacity = torch.sigmoid(ckpt["opacity_logits"]).numpy()
    return xyz, scales, colors, opacity


def export_npz(out_path: str, xyz: np.ndarray, scales: np.ndarray, colors: np.ndarray, opacity: np.ndarray) -> None:
    np.savez_compressed(out_path, xyz=xyz, scales=scales, colors=colors, opacity=opacity)


def export_ply(out_path: str, xyz: np.ndarray, colors: np.ndarray, opacity: np.ndarray) -> None:
    rgb255 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    op255 = np.clip(opacity.squeeze(-1) * 255.0, 0, 255).astype(np.uint8)
    n = xyz.shape[0]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property uchar alpha\n")
        f.write("end_header\n")
        for i in range(n):
            f.write(
                f"{xyz[i,0]:.6f} {xyz[i,1]:.6f} {xyz[i,2]:.6f} "
                f"{int(rgb255[i,0])} {int(rgb255[i,1])} {int(rgb255[i,2])} {int(op255[i])}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="导出训练后的 3DGS 模型为 NPZ/PLY")
    parser.add_argument("--checkpoint", required=True, help="checkpoint 文件")
    parser.add_argument("--export_dir", default="output/export", help="导出目录")
    args = parser.parse_args()

    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    xyz, scales, colors, opacity = load_ckpt(args.checkpoint)
    npz_path = export_dir / "gaussians_model.npz"
    ply_path = export_dir / "gaussians_points.ply"
    export_npz(str(npz_path), xyz, scales, colors, opacity)
    export_ply(str(ply_path), xyz, colors, opacity)
    print(f"[DONE] 导出完成:\n- {npz_path}\n- {ply_path}")


if __name__ == "__main__":
    main()
