#!/usr/bin/env python3
"""
简化版3DGS推理和可视化脚本
用于从不同视角渲染高斯
"""

import os
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_gaussians(gaussians_file):
    """加载高斯"""
    with open(gaussians_file, 'rb') as f:
        data = pickle.load(f)
    return data[:5000]


def create_camera(angle, elevation, distance, look_at=np.array([0, 0, 0])):
    """创建相机参数"""
    pos = np.array([
        distance * np.sin(angle) * np.cos(elevation),
        distance * np.cos(angle) * np.cos(elevation),
        distance * np.sin(elevation)
    ], dtype=np.float32)

    return {
        'position': pos,
        'look_at': look_at.astype(np.float32),
        'up_vector': np.array([0, 0, 1], dtype=np.float32),
        'fov': 50.0
    }


def project_gaussians_to_image(gaussians, camera, image_size):
    """将高斯投影到2D图像平面"""
    width, height = image_size

    fov_rad = np.radians(camera['fov'])
    focal_length = (width / 2) / np.tan(fov_rad / 2)

    cam_pos = camera['position']
    look_at = camera['look_at']
    up = camera['up_vector']

    z_axis = look_at - cam_pos
    z_axis_norm = np.linalg.norm(z_axis)
    if z_axis_norm < 0.001:
        z_axis = np.array([0, 0, 1])
    else:
        z_axis = z_axis / z_axis_norm

    x_axis = np.cross(z_axis, up)
    x_axis_norm = np.linalg.norm(x_axis)
    if x_axis_norm < 0.001:
        x_axis = np.array([1, 0, 0])
    else:
        x_axis = x_axis / x_axis_norm

    y_axis = np.cross(x_axis, z_axis)

    view_matrix = np.array([
        [x_axis[0], x_axis[1], x_axis[2], -np.dot(x_axis, cam_pos)],
        [y_axis[0], y_axis[1], y_axis[2], -np.dot(y_axis, cam_pos)],
        [z_axis[0], z_axis[1], z_axis[2], -np.dot(z_axis, cam_pos)],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    projected = []

    for g in gaussians:
        pos = g['position']
        scale = g['scale']
        color = g['color']
        opacity = g['opacity'] if isinstance(g['opacity'], float) else float(g['opacity'])

        pos_homo = np.append(pos, 1.0)
        cam_pos_homo = view_matrix @ pos_homo

        if cam_pos_homo[2] <= 0.05:
            continue

        u = focal_length * cam_pos_homo[0] / cam_pos_homo[2] + width / 2
        v = focal_length * cam_pos_homo[1] / cam_pos_homo[2] + height / 2

        if u < -100 or u >= width + 100 or v < -100 or v >= height + 100:
            continue

        projected.append({
            'center': (float(u), float(v)),
            'scale': scale,
            'color': color,
            'opacity': opacity,
            'depth': float(cam_pos_homo[2])
        })

    projected.sort(key=lambda x: x['depth'])

    return projected


def render_image(projected, image_size, background_color=(0.9, 0.9, 0.9)):
    """使用2D高斯渲染图像"""
    width, height = image_size

    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    image = np.ones((height, width, 3)) * np.array(background_color)
    accumulated_alpha = np.zeros((height, width))

    for p in projected:
        cx, cy = p['center']
        scale = p['scale']
        color = p['color']
        alpha = p['opacity']

        a = max(float(scale[0]) * 200, 1.0)
        b = max(float(scale[1]) * 200, 1.0)

        dx = xx - cx
        dy = yy - cy

        gaussian_value = np.exp(-0.5 * ((dx / (a + 0.1)) ** 2 + (dy / (b + 0.1)) ** 2))

        weight = alpha * gaussian_value * (1 - accumulated_alpha)
        image += weight[:, :, np.newaxis] * color
        accumulated_alpha += weight

    image = np.clip(image, 0, 1)
    return image


def visualize_gaussians_3d(gaussians, output_file=None):
    """可视化3D高斯分布"""
    positions = []
    colors = []

    for g in gaussians[:1000]:
        pos = g['position']
        color = g['color']
        positions.append(pos)
        colors.append(color)

    positions = np.array(positions)
    colors = np.array(colors)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=colors, s=10, alpha=0.6)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Gaussians Distribution')

    if output_file:
        plt.savefig(output_file)
        print(f"Saved 3D visualization to {output_file}")

    plt.close()
    return fig


def render_360_video(gaussians, output_dir, num_frames=12, image_size=(640, 360)):
    """渲染360度旋转视频"""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Rendering {num_frames} frames...")

    for i in range(num_frames):
        angle = 2 * np.pi * i / num_frames
        camera = create_camera(angle, np.pi / 6, 0.5)

        projected = project_gaussians_to_image(gaussians, camera, image_size)
        rendered = render_image(projected, image_size)
        rendered_img = (rendered * 255).astype(np.uint8)

        output_path = os.path.join(output_dir, f"frame_{i:04d}.png")
        cv2.imwrite(output_path, cv2.cvtColor(rendered_img, cv2.COLOR_RGB2BGR))

        print(f"Rendered frame {i + 1}/{num_frames}")

    print(f"Video frames saved to {output_dir}")


def render_comparison(gaussians, frames_dir, output_dir, num_views=4):
    """渲染并与原图对比"""
    os.makedirs(output_dir, exist_ok=True)

    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.png', '.jpg'))])
    indices = np.linspace(0, len(frame_files) - 1, num_views, dtype=int)

    for i, idx in enumerate(indices):
        frame = cv2.imread(os.path.join(frames_dir, frame_files[idx]))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame, (640, 360))

        angle = 2 * np.pi * i / num_views
        camera = create_camera(angle, np.pi / 6, 0.5)

        projected = project_gaussians_to_image(gaussians, camera, (640, 360))
        rendered = render_image(projected, (640, 360))
        rendered_img = (rendered * 255).astype(np.uint8)

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        axes[0].imshow(rendered_img)
        axes[0].set_title('Rendered')
        axes[0].axis('off')

        axes[1].imshow(frame_resized)
        axes[1].set_title('Original')
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'comparison_{i:02d}.png'), dpi=100)
        plt.close()

    print(f"Comparison images saved to {output_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="3DGS Visualization and Inference")
    parser.add_argument("--gaussians", "-g", type=str, default="output/gaussians_init.pkl",
                        help="Path to Gaussians file")
    parser.add_argument("--frames", "-f", type=str, default="frames",
                        help="Path to original frames directory")
    parser.add_argument("--output", "-o", type=str, default="output/renders",
                        help="Output directory for renders")
    parser.add_argument("--mode", "-m", type=str, default="360",
                        choices=['360', 'compare', '3d'],
                        help="Rendering mode")

    args = parser.parse_args()

    if not os.path.exists(args.gaussians):
        print(f"Gaussians file not found: {args.gaussians}")
        print("Please run initialize.py first")
        return

    print(f"Loading Gaussians from {args.gaussians}...")
    gaussians = load_gaussians(args.gaussians)
    print(f"Loaded {len(gaussians)} Gaussians")

    if args.mode == '360':
        render_360_video(gaussians, args.output)
    elif args.mode == 'compare':
        render_comparison(gaussians, args.frames, args.output)
    elif args.mode == '3d':
        visualize_gaussians_3d(gaussians, os.path.join(args.output, "gaussians_3d.png"))


if __name__ == "__main__":
    main()