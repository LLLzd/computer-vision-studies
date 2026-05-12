#!/usr/bin/env python3
"""
3DGS推理和可视化脚本
用于从不同视角渲染重建的3D场景
"""

import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2


class Gaussian3D_torch(torch.nn.Module):
    """PyTorch版本的3D高斯（用于渲染）"""

    def __init__(self, position, scale, rotation, color, opacity):
        super().__init__()
        self.position = torch.nn.Parameter(torch.tensor(position, dtype=torch.float32), requires_grad=False)
        self.scale = torch.nn.Parameter(torch.tensor(scale, dtype=torch.float32), requires_grad=False)
        self.rotation = torch.nn.Parameter(torch.tensor(rotation, dtype=torch.float32), requires_grad=False)
        self.color = torch.nn.Parameter(torch.tensor(color, dtype=torch.float32), requires_grad=False)
        self.opacity = torch.nn.Parameter(torch.tensor(opacity, dtype=torch.float32), requires_grad=False)

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

        if u < -200 or u >= width + 200 or v < -200 or v >= height + 200:
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
                'color': g.color,
                'opacity': g.opacity,
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


def load_gaussians(gaussians_file):
    """加载高斯"""
    with open(gaussians_file, 'rb') as f:
        data = pickle.load(f)

    gaussians = []
    for g in data:
        gaussians.append(Gaussian3D_torch(
            g['position'].astype(np.float32),
            g['scale'].astype(np.float32),
            g['rotation'].astype(np.float32),
            g['color'].astype(np.float32),
            np.array([g['opacity'].item() if hasattr(g['opacity'], 'item') else g['opacity']], dtype=np.float32)
        ))

    return gaussians


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
        'fov': 45.0
    }


def visualize_gaussians_3d(gaussians, output_file=None):
    """可视化3D高斯分布"""
    positions = []
    colors = []
    scales = []

    for g in gaussians:
        pos = g.position.detach().cpu().numpy()
        color = g.color.detach().cpu().numpy()
        scale = g.scale.detach().cpu().numpy()

        positions.append(pos)
        colors.append(color)
        scales.append(np.mean(scale) * 100)

    positions = np.array(positions)
    colors = np.array(colors)
    scales = np.array(scales)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        positions[:, 0], positions[:, 1], positions[:, 2],
        c=colors, s=scales, alpha=0.6
    )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Gaussians Distribution')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file)
        print(f"Saved 3D visualization to {output_file}")

    plt.close()

    return fig


def render_360_video(gaussians, output_dir, num_frames=36, image_size=(640, 360)):
    """渲染360度旋转视频"""
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gaussians = [g.to(device) for g in gaussians]

    print(f"Rendering {num_frames} frames...")

    for i in range(num_frames):
        angle = 2 * np.pi * i / num_frames
        camera = create_camera(angle, np.pi / 6, 1.0)
        camera = {
            'position': torch.tensor(camera['position'], dtype=torch.float32).to(device),
            'look_at': torch.tensor(camera['look_at'], dtype=torch.float32).to(device),
            'up_vector': torch.tensor(camera['up_vector'], dtype=torch.float32).to(device),
            'fov': camera['fov']
        }

        rendered = render_gaussians_pytorch(gaussians, camera, image_size)
        rendered_img = (rendered.detach().cpu().numpy() * 255).astype(np.uint8)

        output_path = os.path.join(output_dir, f"frame_{i:04d}.png")
        cv2.imwrite(output_path, cv2.cvtColor(rendered_img, cv2.COLOR_RGB2BGR))

        if (i + 1) % 10 == 0:
            print(f"Rendered {i + 1}/{num_frames} frames")

    print(f"Video frames saved to {output_dir}")


def render_comparison(gaussians, frames_dir, output_dir, num_views=5):
    """渲染并与原图对比"""
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gaussians = [g.to(device) for g in gaussians]

    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.png', '.jpg'))])

    indices = np.linspace(0, len(frame_files) - 1, num_views, dtype=int)

    for i, idx in enumerate(indices):
        frame = cv2.imread(os.path.join(frames_dir, frame_files[idx]))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        angle = 2 * np.pi * i / num_views
        camera = create_camera(angle, np.pi / 6, 1.0)
        camera = {
            'position': torch.tensor(camera['position'], dtype=torch.float32).to(device),
            'look_at': torch.tensor(camera['look_at'], dtype=torch.float32).to(device),
            'up_vector': torch.tensor(camera['up_vector'], dtype=torch.float32).to(device),
            'fov': camera['fov']
        }

        rendered = render_gaussians_pytorch(gaussians, camera, (frame.shape[1], frame.shape[0]))
        rendered_img = (rendered.detach().cpu().numpy() * 255).astype(np.uint8)

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        axes[0].imshow(rendered_img)
        axes[0].set_title('Rendered')
        axes[0].axis('off')

        axes[1].imshow(frame)
        axes[1].set_title('Original')
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'comparison_{i:02d}.png'), dpi=100)
        plt.close()

    print(f"Comparison images saved to {output_dir}")


def interactive_viewer(gaussians, image_size=(640, 360)):
    """简单的交互式查看器（命令行控制视角）"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gaussians = [g.to(device) for g in gaussians]

    print("\nInteractive 3DGS Viewer")
    print("Commands: 'q' to quit, 'r' to reset, 's' to save, arrow keys to rotate")
    print("Default: angle=0, elevation=30deg, distance=1.0")

    angle = 0.0
    elevation = np.pi / 6
    distance = 1.0

    while True:
        camera = create_camera(angle, elevation, distance)
        camera = {
            'position': torch.tensor(camera['position'], dtype=torch.float32).to(device),
            'look_at': torch.tensor(camera['look_at'], dtype=torch.float32).to(device),
            'up_vector': torch.tensor(camera['up_vector'], dtype=torch.float32).to(device),
            'fov': camera['fov']
        }

        rendered = render_gaussians_pytorch(gaussians, camera, image_size)
        rendered_img = (rendered.detach().cpu().numpy() * 255).astype(np.uint8)

        plt.figure(figsize=(10, 6))
        plt.imshow(rendered_img)
        plt.title(f"Angle: {np.degrees(angle):.1f}°, Elevation: {np.degrees(elevation):.1f}°, Distance: {distance:.2f}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        cmd = input("Command (q/r/s/←/→/↑/↓/+/-): ").strip().lower()

        if cmd == 'q':
            break
        elif cmd == 'r':
            angle = 0.0
            elevation = np.pi / 6
            distance = 1.0
        elif cmd == 's':
            filename = f"rendered_{np.degrees(angle):.0f}_{np.degrees(elevation):.0f}.png"
            cv2.imwrite(filename, cv2.cvtColor(rendered_img, cv2.COLOR_RGB2BGR))
            print(f"Saved to {filename}")
        elif cmd == '←':
            angle -= 0.2
        elif cmd == '→':
            angle += 0.2
        elif cmd == '↑':
            elevation = min(elevation + 0.1, np.pi / 2 - 0.1)
        elif cmd == '↓':
            elevation = max(elevation - 0.1, -np.pi / 2 + 0.1)
        elif cmd == '+':
            distance = max(distance - 0.1, 0.3)
        elif cmd == '-':
            distance = min(distance + 0.1, 3.0)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="3DGS Visualization and Inference")
    parser.add_argument("--gaussians", "-g", type=str, default="output/gaussians_trained.pkl",
                        help="Path to trained Gaussians file")
    parser.add_argument("--frames", "-f", type=str, default="frames",
                        help="Path to original frames directory")
    parser.add_argument("--output", "-o", type=str, default="output/renders",
                        help="Output directory for renders")
    parser.add_argument("--mode", "-m", type=str, default="360",
                        choices=['360', 'compare', '3d', 'interactive'],
                        help="Rendering mode")

    args = parser.parse_args()

    if not os.path.exists(args.gaussians):
        print(f"Gaussians file not found: {args.gaussians}")
        print("Please run training first: python train_3dgs.py")
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
    elif args.mode == 'interactive':
        interactive_viewer(gaussians)


if __name__ == "__main__":
    main()