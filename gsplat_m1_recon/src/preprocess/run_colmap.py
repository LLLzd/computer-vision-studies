from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"命令失败: {' '.join(cmd)}")


def run_colmap_pipeline(images_dir: str, workspace: str, camera_model: str = "SIMPLE_RADIAL") -> None:
    if shutil.which("colmap") is None:
        raise RuntimeError(
            "未找到 colmap 命令。请先安装:\n"
            "  brew install colmap\n"
            "并确认 `colmap -h` 可运行。"
        )

    ws = Path(workspace)
    db_path = ws / "database.db"
    sparse_dir = ws / "sparse"
    txt_dir = ws / "sparse_txt"
    ws.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    # 注意：COLMAP 3.9+ 已移除部分旧版 gflags（如 --SiftExtraction.use_gpu）。
    # Apple Silicon 上通常无 CUDA，默认即为 CPU SIFT，无需再传 use_gpu。
    images_path = Path(images_dir)
    if not images_path.is_dir():
        raise RuntimeError(f"图像目录不存在: {images_path}")
    has_image = any(
        p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        for p in images_path.iterdir()
        if p.is_file()
    )
    if not has_image:
        raise RuntimeError(
            f"图像目录中没有可用图像: {images_path}。\n"
            "请先成功抽帧；若抽帧为 0 张，请降低 config 里 data.sharpness_threshold，"
            "或设为 0 关闭清晰度过滤。"
        )

    run(
        [
            "colmap",
            "feature_extractor",
            "--database_path",
            str(db_path),
            "--image_path",
            str(images_path),
            "--ImageReader.single_camera",
            "1",
            "--ImageReader.camera_model",
            camera_model,
        ]
    )
    run(["colmap", "exhaustive_matcher", "--database_path", str(db_path)])
    run(
        [
            "colmap",
            "mapper",
            "--database_path",
            str(db_path),
            "--image_path",
            str(Path(images_dir)),
            "--output_path",
            str(sparse_dir),
        ]
    )
    run(
        [
            "colmap",
            "model_converter",
            "--input_path",
            str(sparse_dir / "0"),
            "--output_path",
            str(txt_dir),
            "--output_type",
            "TXT",
        ]
    )
    print(f"COLMAP 处理完成，文本模型输出: {txt_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 COLMAP，生成相机位姿和稀疏点云")
    parser.add_argument("--images", required=True, help="抽帧目录")
    parser.add_argument("--workspace", required=True, help="COLMAP 工作目录")
    parser.add_argument("--camera_model", default="SIMPLE_RADIAL", help="COLMAP 相机模型")
    args = parser.parse_args()

    run_colmap_pipeline(args.images, args.workspace, args.camera_model)


if __name__ == "__main__":
    main()
