#!/usr/bin/env python3
"""
简化版3DGS训练脚本 - 使用NumPy进行快速原型验证
优化版本：减少高斯数量、增加保存频率、向量化加速
"""

import os
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_gaussians(init_file, max_gaussians=5000):
    """加载高斯，限制数量"""
    with open(init_file, 'rb') as f:
        data = pickle.load(f)
    return data[:max_gaussians]


def create_cameras_from_frames(frames_dir, num_cameras=5):
    """从帧创建虚拟相机"""
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.png', '.jpg'))])

    cameras = []
    n_frames = len(frame_files)

    for i in range(min(num_cameras, n_frames)):
        angle = 2 * np.pi * i / n_frames

        camera = {
            'position': np.array([
                np.sin(angle) * 0.3,
                np.cos(angle) * 0.3,
                0.1
            ], dtype=np.float32),
            'look_at': np.array([0, 0, 0], dtype=np.float32),
            'up_vector': np.array([0, 0, 1], dtype=np.float32),
            'fov': 50.0
        }
        cameras.append(camera)

    return cameras


def project_gaussians_to_image(gaussians, camera, image_size):
    """将高斯投影到2D图像平面 - 向量化版本"""
    width, height = image_size

    fov_rad = np.radians(camera['fov'])
    focal_length = (width / 2) / np.tan(fov_rad / 2)

    cam_pos = camera['position']
    look_at = camera['look_at']
    up = camera['up_vector']

    z_axis = look_at - cam_pos
    z_axis = z_axis / np.linalg.norm(z_axis)

    x_axis = np.cross(z_axis, up)
    x_axis = x_axis / np.linalg.norm(x_axis)

    y_axis = np.cross(x_axis, z_axis)

    view_matrix = np.array([
        [x_axis[0], x_axis[1], x_axis[2], -np.dot(x_axis, cam_pos)],
        [y_axis[0], y_axis[1], y_axis[2], -np.dot(y_axis, cam_pos)],
        [z_axis[0], z_axis[1], z_axis[2], -np.dot(z_axis, cam_pos)],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    positions = np.array([g['position'] for g in gaussians])
    scales = np.array([g['scale'] for g in gaussians])
    colors = np.array([g['color'] for g in gaussians])
    opacities = np.array([float(g['opacity']) for g in gaussians])

    n_gaussians = len(gaussians)
    pos_homo = np.ones((n_gaussians, 4))
    pos_homo[:, :3] = positions

    cam_pos_homo = (view_matrix @ pos_homo.T).T

    valid_mask = cam_pos_homo[:, 2] > 0.1

    u = focal_length * cam_pos_homo[valid_mask, 0] / cam_pos_homo[valid_mask, 2] + width / 2
    v = focal_length * cam_pos_homo[valid_mask, 1] / cam_pos_homo[valid_mask, 2] + height / 2

    in_bounds = (u >= -50) & (u < width + 50) & (v >= -50) & (v < height + 50)

    projected = []
    valid_idx = np.where(valid_mask)[0][in_bounds]

    for i, idx in enumerate(valid_idx):
        projected.append({
            'center': (float(u[i]), float(v[i])),
            'scale': scales[idx],
            'color': colors[idx],
            'opacity': float(opacities[idx]),
            'depth': float(cam_pos_homo[idx, 2]),
            'gaussian_idx': idx
        })

    projected.sort(key=lambda x: x['depth'])

    return projected


def render_image(projected, image_size, background_color=(0.9, 0.9, 0.9)):
    """使用2D高斯渲染图像"""
    width, height = image_size

    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    image = np.ones((height, width, 3)) * background_color
    accumulated_alpha = np.zeros((height, width))

    for p in projected:
        cx, cy = p['center']
        scale = p['scale']
        color = p['color']
        alpha = p['opacity']

        a = max(scale[0] * 0.003125 * 100, 1.0)
        b = max(scale[1] * 0.003125 * 100, 1.0)

        dx = xx - cx
        dy = yy - cy

        gaussian_value = np.exp(-0.5 * ((dx / (a + 0.1)) ** 2 + (dy / (b + 0.1)) ** 2))

        weight = alpha * gaussian_value * (1 - accumulated_alpha)
        image += weight[:, :, np.newaxis] * color
        accumulated_alpha += weight

    image = np.clip(image, 0, 1)
    return image


def train(frames_dir, init_file, output_dir, num_iterations=100, max_gaussians=2000, save_interval=10):
    """训练高斯 - 优化版本"""
    os.makedirs(output_dir, exist_ok=True)

    print("Loading Gaussians...")
    gaussians = load_gaussians(init_file, max_gaussians)
    print(f"Loaded {len(gaussians)} Gaussians (max: {max_gaussians})")

    cameras = create_cameras_from_frames(frames_dir, num_cameras=5)
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.png', '.jpg'))])

    target_images = []
    for i, cam_idx in enumerate(np.linspace(0, len(frame_files) - 1, len(cameras), dtype=int)):
        img = cv2.imread(os.path.join(frames_dir, frame_files[cam_idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (320, 180))
        img = img.astype(np.float32) / 255.0
        target_images.append(img)

    image_size = (320, 180)

    positions = np.array([g['position'] for g in gaussians], dtype=np.float32)
    scales = np.array([g['scale'] for g in gaussians], dtype=np.float32)
    colors = np.array([g['color'] for g in gaussians], dtype=np.float32)
    opacities = np.array([float(g['opacity']) for g in gaussians], dtype=np.float32)

    lr_pos = 0.005
    lr_color = 0.005
    lr_opacity = 0.001

    losses_history = []

    print(f"\nStarting training for {num_iterations} iterations...")
    print(f"Gaussians: {len(gaussians)}, Save interval: {save_interval}")

    for iteration in tqdm(range(num_iterations)):
        cam_idx = iteration % len(cameras)
        camera = cameras[cam_idx]
        target = target_images[cam_idx]

        projected = project_gaussians_to_image(gaussians, camera, image_size)
        rendered = render_image(projected, image_size)

        loss = np.mean((rendered - target) ** 2)
        losses_history.append(loss)

        diff = target - rendered

        for p in projected:
            cx, cy = p['center']
            g_idx = p['gaussian_idx']

            if 0 <= int(cy) < image_size[1] and 0 <= int(cx) < image_size[0]:
                pixel_loss = diff[int(cy), int(cx)]

                dx = positions[g_idx, 0] - (cx - image_size[0] / 2) * 0.001
                dy = positions[g_idx, 1] - (cy - image_size[1] / 2) * 0.001
                dist = np.sqrt(dx ** 2 + dy ** 2)

                if dist < 0.15:
                    influence = np.exp(-dist * 15)
                    color_loss = float(pixel_loss.mean())

                    colors[g_idx] = np.clip(colors[g_idx] + color_loss * influence * lr_color, 0, 1)
                    positions[g_idx, 0] += color_loss * dx * influence * lr_pos
                    positions[g_idx, 1] += color_loss * dy * influence * lr_pos

        if (iteration + 1) % save_interval == 0:
            print(f"\nIteration {iteration + 1}, Loss: {loss:.6f}")

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].imshow(rendered)
            axes[0].set_title('Rendered')
            axes[0].axis('off')
            axes[1].imshow(target)
            axes[1].set_title('Target')
            axes[1].axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'comparison_{iteration + 1:04d}.png'))
            plt.close()

            gaussians_data = []
            for i in range(len(gaussians)):
                gaussians_data.append({
                    'position': positions[i],
                    'scale': scales[i],
                    'rotation': gaussians[i]['rotation'],
                    'color': colors[i],
                    'opacity': float(opacities[i])
                })

            checkpoint_file = os.path.join(output_dir, f'gaussians_checkpoint_{iteration + 1:04d}.pkl')
            save_trained_gaussians(gaussians_data, checkpoint_file)

    print("\nTraining complete!")

    plt.figure(figsize=(10, 5))
    plt.plot(losses_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()

    gaussians_data = []
    for i in range(len(gaussians)):
        gaussians_data.append({
            'position': positions[i],
            'scale': scales[i],
            'rotation': gaussians[i]['rotation'],
            'color': colors[i],
            'opacity': float(opacities[i])
        })

    save_trained_gaussians(gaussians_data, os.path.join(output_dir, 'gaussians_trained.pkl'))

    return gaussians_data, losses_history


def save_trained_gaussians(gaussians, output_file):
    """保存训练后的高斯"""
    with open(output_file, 'wb') as f:
        pickle.dump(gaussians, f)

    print(f"Saved {len(gaussians)} Gaussians to {output_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train 3DGS (simplified - optimized for M1)")
    parser.add_argument("--frames", "-f", type=str, default="frames",
                        help="Directory containing frames")
    parser.add_argument("--init", "-i", type=str, default="output/gaussians_init.pkl",
                        help="Initial Gaussians file")
    parser.add_argument("--output", "-o", type=str, default="output",
                        help="Output directory")
    parser.add_argument("--iterations", "-n", type=int, default=100,
                        help="Number of training iterations")
    parser.add_argument("--max-gaussians", "-g", type=int, default=2000,
                        help="Maximum number of Gaussians to use")
    parser.add_argument("--save-interval", "-s", type=int, default=10,
                        help="Save checkpoint every N iterations")

    args = parser.parse_args()

    if not os.path.exists(args.init):
        print(f"Init file not found: {args.init}")
        print("Please run initialize.py first")
        return

    train(args.frames, args.init, args.output, args.iterations, args.max_gaussians, args.save_interval)


if __name__ == "__main__":
    main()