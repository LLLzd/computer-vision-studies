import torch
import pickle
import numpy as np
from train.train_torch import GaussianModel, render_gaussian_splatting, create_view_matrix, create_proj_matrix

# 加载模型
with open('output/gaussians_init.pkl', 'rb') as f:
    data = pickle.load(f)[:1000]

positions = torch.tensor([g['position'] for g in data], dtype=torch.float32)
scales = torch.tensor([g['scale'] for g in data], dtype=torch.float32)
rotations = torch.tensor([g['rotation'] for g in data], dtype=torch.float32)
colors = torch.tensor([g['color'] for g in data], dtype=torch.float32)
opacities = torch.tensor([[g['opacity']] for g in data], dtype=torch.float32)

print("高斯参数统计:")
print(f"  位置范围: x={positions[:,0].min():.3f}~{positions[:,0].max():.3f}, y={positions[:,1].min():.3f}~{positions[:,1].max():.3f}, z={positions[:,2].min():.3f}~{positions[:,2].max():.3f}")
print(f"  颜色范围: {colors.min():.3f}~{colors.max():.3f}")
print(f"  不透明度范围: {opacities.min():.3f}~{opacities.max():.3f}")

model = GaussianModel(positions, scales, rotations, colors, opacities)

# 创建相机（确保浮点类型）- 使用更合适的参数
# 高斯位置范围: x=-0.27~0.27, y=-0.48~0.48, z=0~0.5
camera_pos = torch.tensor([2.0, 0.0, 0.3], dtype=torch.float32)  # 更低的相机位置
look_at = torch.tensor([0.0, 0.0, 0.2], dtype=torch.float32)     # 看向场景中心
up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
viewmat = create_view_matrix(camera_pos, look_at, up)
projmat = create_proj_matrix(45, 320/180)

print("\n视图矩阵:")
print(viewmat)
print("\n投影矩阵:")
print(projmat)

# 手动调用forward检查投影
print("\n=== 手动检查投影 ===")
proj_points, cov2d, colors, opacities, valid_mask = model(viewmat, projmat, (320, 180))
print(f"总高斯数: {len(proj_points)}")
print(f"有效高斯数: {valid_mask.sum().item()}")
print(f"位置深度范围: {proj_points[:,2].min():.3f} ~ {proj_points[:,2].max():.3f}")
print(f"投影点x范围: {proj_points[:,0].min():.1f} ~ {proj_points[:,0].max():.1f}")
print(f"投影点y范围: {proj_points[:,1].min():.1f} ~ {proj_points[:,1].max():.1f}")

# 检查前几个点
print("\n前5个投影点:")
for i in range(min(5, len(proj_points))):
    print(f"  [{proj_points[i,0]:.1f}, {proj_points[i,1]:.1f}, {proj_points[i,2]:.3f}]")

# 渲染（带调试）
image = render_gaussian_splatting(model, viewmat, projmat, (320, 180), debug=True)
print(f'\n渲染图像形状: {image.shape}')
print(f'图像值范围: {image.min():.3f} ~ {image.max():.3f}')
print(f'图像均值: {image.mean():.3f}')
