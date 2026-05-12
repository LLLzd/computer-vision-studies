#!/usr/bin/env python3
"""
3DGS训练脚本 - 使用PyTorch进行可微渲染和梯度优化
"""

import os
import sys
import pickle
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


class Gaussian3D_torch(torch.nn.Module):
    """PyTorch版本的3D高斯"""

    def __init__(self, position, scale, rotation, color, opacity):
        super().__init__()
        self.position = torch.nn.Parameter(torch.tensor(position, dtype=torch.float32))
        self.scale = torch.nn.Parameter(torch.tensor(scale, dtype=torch.float32))
        self.rotation = torch.nn.Parameter(torch.tensor(rotation, dtype=torch.float32))
        self.color = torch.nn.Parameter(torch.tensor(color, dtype=torch.float32))
        self.opacity = torch.nn.Parameter(torch.tensor(opacity, dtype=torch.float32))

    def get_covariance(self):
        scale_matrix = torch.diag(self.scale ** 2 + 0.001)

        rx, ry, rz = self.rotation

        cos_x, sin_x = torch.cos(rx), torch.sin(rx)
        cos_y, sin_y = torch.cos(ry), torch.sin(ry)
        cos_z, sin_z = torch.cos(rz), torch.sin(rz)

        Rx = torch.tensor([
            [1, 0, 0],
            [0, cos_x, -sin_x],
            [0, sin_x, cos_x]
        ], dtype=torch.float32, device=self.scale.device)

        Ry = torch.tensor([
            [cos_y, 0, sin_y],
            [0, 1, 0],
            [-sin_y, 0, cos_y]
        ], dtype=torch.float32, device=self.scale.device)

        Rz = torch.tensor([
            [cos_z, -sin_z, 0],
            [sin_z, cos_z, 0],
            [0, 0, 1]
        ], dtype=torch.float32, device=self.scale.device)

        R = Rz @ Ry @ Rx
        return R @ scale_matrix @ R.T

    def project_to_2d(self, camera_pos, look_at, up_vector, fov, image_size):
        width, height = image_size

        z_axis = torch.nn.functional.normalize(look_at - camera_pos, dim=0)
        x_axis = torch.nn.functional.normalize(torch.cross(z_axis, up_vector, dim=0), dim=0)
        y_axis = torch.cross(x_axis, z_axis, dim=0)

        view_matrix = torch.tensor([
            [x_axis[0], x_axis[1], x_axis[2], -torch.dot(x_axis, camera_pos)],
            [y_axis[0], y_axis[1], y_axis[2], -torch.dot(y_axis, camera_pos)],
            [z_axis[0], z_axis[1], z_axis[2], -torch.dot(z_axis, camera_pos)],
            [0, 0, 0, 1]
        ], dtype=torch.float32, device=camera_pos.device)

        pos_homo = torch.cat([self.position, torch.ones(1, device=camera_pos.device)])
        cam_pos = view_matrix @ pos_homo

        if cam_pos[2] <= 0.1:
            return None

        fov_rad = np.radians(fov)
        focal_length = (width / 2) / np.tan(fov_rad / 2)

        u = focal_length * cam_pos[0] / cam_pos[2] + width / 2
        v = focal_length * cam_pos[1] / cam_pos[2] + height / 2

        if u < -100 or u >= width + 100 or v < -100 or v >= height + 100:
            return None

        return {
            'center': torch.stack([u, v]),
            'depth': cam_pos[2],
            'covariance': self.get_covariance(),
            'view_matrix': view_matrix[:3, :3]
        }


def compute_2d_gaussian(covariance_3d, view_matrix, focal_length, width, height):
    R = view_matrix

    J_proj = torch.tensor([
        [focal_length, 0, -focal_length * 0],
        [0, focal_length, -focal_length * 0]
    ], dtype=torch.float32, device=covariance_3d.device)

    cov_2d = J_proj @ R @ covariance_3d @ R.T @ J_proj.T

    eigenvalues, eigenvectors = torch.linalg.eig(cov_2d)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real

    idx = torch.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    semi_axes = 3 * torch.sqrt(torch.clamp(eigenvalues, min=0.1))
    angle = torch.atan2(eigenvectors[1, 1], eigenvectors[0, 1])

    return semi_axes, angle


def render_gaussians_pytorch(gaussians, camera, image_size, background_color=(0, 0, 0)):
    """使用PyTorch渲染高斯"""
    width, height = image_size

    camera_pos = torch.tensor(camera['position'], dtype=torch.float32)
    look_at = torch.tensor(camera['look_at'], dtype=torch.float32)
    up = torch.tensor(camera['up_vector'], dtype=torch.float32)
    fov = camera['fov']

    fov_rad = np.radians(fov)
    focal_length = (width / 2) / np.tan(fov_rad / 2)

    projected = []
    for g in gaussians:
        proj = g.project_to_2d(camera_pos, look_at, up, fov, image_size)
        if proj is not None:
            semi_axes, angle = compute_2d_gaussian(
                proj['covariance'], proj['view_matrix'],
                focal_length, width, height
            )
            projected.append({
                'center': proj['center'],
                'semi_axes': semi_axes,
                'angle': angle,
                'color': torch.sigmoid(g.color),
                'opacity': torch.sigmoid(g.opacity),
                'depth': proj['depth']
            })

    if len(projected) == 0:
        return torch.ones(height, width, 3)

    projected.sort(key=lambda x: x['depth'].item())

    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing='ij'
    )

    image = torch.zeros(height, width, 3)
    accumulated_alpha = torch.zeros(height, width)

    for p in projected:
        cx, cy = p['center']
        a, b = p['semi_axes']
        angle = p['angle']
        color = p['color']
        alpha = p['opacity']

        cos_a = torch.cos(-angle)
        sin_a = torch.sin(-angle)

        dx = xx - cx
        dy = yy - cy

        dx_rot = cos_a * dx - sin_a * dy
        dy_rot = sin_a * dx + cos_a * dy

        gaussian_value = torch.exp(-0.5 * ((dx_rot / (a + 0.1)) ** 2 + (dy_rot / (b + 0.1)) ** 2))

        weight = alpha * gaussian_value * (1 - accumulated_alpha)
        image += weight.unsqueeze(-1) * color
        accumulated_alpha += weight

    image = image + (1 - accumulated_alpha.unsqueeze(-1)) * torch.tensor(background_color, dtype=torch.float32)

    return torch.clamp(image, 0, 1)


def load_gaussians(init_file):
    """加载初始化的高斯"""
    with open(init_file, 'rb') as f:
        data = pickle.load(f)

    gaussians = []
    for g in data:
        gaussians.append(Gaussian3D_torch(
            g['position'].astype(np.float32),
            g['scale'].astype(np.float32),
            g['rotation'].astype(np.float32),
            g['color'].astype(np.float32),
            np.array([g['opacity']], dtype=np.float32)
        ))

    return gaussians


def create_cameras_from_frames(frames_dir, num_cameras=10):
    """从帧创建虚拟相机"""
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.png', '.jpg'))])

    cameras = []
    n_frames = len(frame_files)

    for i in range(min(num_cameras, n_frames)):
        angle = 2 * np.pi * i / n_frames

        camera = {
            'position': np.array([
                np.sin(angle) * 0.5,
                np.cos(angle) * 0.5,
                0.3 + 0.1 * np.sin(i * 0.5)
            ], dtype=np.float32),
            'look_at': np.array([0, 0, 0], dtype=np.float32),
            'up_vector': np.array([0, 0, 1], dtype=np.float32),
            'fov': 45.0
        }
        cameras.append(camera)

    return cameras


def train(frames_dir, init_file, output_dir, num_iterations=500, lr=0.01):
    """训练高斯"""
    os.makedirs(output_dir, exist_ok=True)

    print("Loading Gaussians...")
    gaussians = load_gaussians(init_file)
    print(f"Loaded {len(gaussians)} Gaussians")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    gaussians = [g.to(device) for g in gaussians]

    cameras = create_cameras_from_frames(frames_dir, num_cameras=10)
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.png', '.jpg'))])

    target_images = []
    for i, cam_idx in enumerate(np.linspace(0, len(frame_files) - 1, len(cameras), dtype=int)):
        img = cv2.imread(os.path.join(frames_dir, frame_files[cam_idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 360))
        img = img.astype(np.float32) / 255.0
        target_images.append(torch.tensor(img, dtype=torch.float32).to(device))

    optimizer = torch.optim.Adam([
        {'params': [g.position for g in gaussians], 'lr': lr},
        {'params': [g.scale for g in gaussians], 'lr': lr * 0.1},
        {'params': [g.color for g in gaussians], 'lr': lr * 0.01},
        {'params': [g.opacity for g in gaussians], 'lr': lr * 0.1},
    ])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    image_size = (640, 360)
    losses_history = []

    print(f"\nStarting training for {num_iterations} iterations...")

    for iteration in tqdm(range(num_iterations)):
        optimizer.zero_grad()

        cam_idx = iteration % len(cameras)
        camera = cameras[cam_idx]
        camera = {
            'position': torch.tensor(camera['position'], dtype=torch.float32).to(device),
            'look_at': torch.tensor(camera['look_at'], dtype=torch.float32).to(device),
            'up_vector': torch.tensor(camera['up_vector'], dtype=torch.float32).to(device),
            'fov': camera['fov']
        }

        rendered = render_gaussians_pytorch(gaussians, camera, image_size)

        target = target_images[cam_idx]

        loss = torch.mean((rendered - target) ** 2)

        loss.backward()
        optimizer.step()
        scheduler.step()

        losses_history.append(loss.item())

        if (iteration + 1) % 100 == 0:
            print(f"\nIteration {iteration + 1}, Loss: {loss.item():.6f}")

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            rendered_img = rendered.detach().cpu().numpy()
            axes[0].imshow(rendered_img)
            axes[0].set_title('Rendered')
            axes[0].axis('off')

            target_img = target.cpu().numpy()
            axes[1].imshow(target_img)
            axes[1].set_title('Target')
            axes[1].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'comparison_{iteration + 1:04d}.png'))
            plt.close()

    print("\nTraining complete!")

    plt.figure(figsize=(10, 5))
    plt.plot(losses_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()

    save_trained_gaussians(gaussians, os.path.join(output_dir, 'gaussians_trained.pkl'))

    return gaussians, losses_history


def save_trained_gaussians(gaussians, output_file):
    """保存训练后的高斯"""
    data = []
    for g in gaussians:
        data.append({
            'position': g.position.detach().cpu().numpy(),
            'scale': g.scale.detach().cpu().numpy(),
            'rotation': g.rotation.detach().cpu().numpy(),
            'color': torch.sigmoid(g.color).detach().cpu().numpy(),
            'opacity': torch.sigmoid(g.opacity).detach().cpu().numpy()
        })

    with open(output_file, 'wb') as f:
        pickle.dump(data, f)

    print(f"Saved {len(data)} Gaussians to {output_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train 3DGS")
    parser.add_argument("--frames", "-f", type=str, default="frames",
                        help="Directory containing frames")
    parser.add_argument("--init", "-i", type=str, default="output/gaussians_init.pkl",
                        help="Initial Gaussians file")
    parser.add_argument("--output", "-o", type=str, default="output",
                        help="Output directory")
    parser.add_argument("--iterations", "-n", type=int, default=500,
                        help="Number of training iterations")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate")

    args = parser.parse_args()

    if not os.path.exists(args.init):
        print(f"Init file not found: {args.init}")
        print("Please run initialize.py first")
        return

    train(args.frames, args.init, args.output, args.iterations, args.lr)


if __name__ == "__main__":
    main()