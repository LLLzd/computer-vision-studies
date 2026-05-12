#!/usr/bin/env python3
"""
NuScenes 数据集可视化工具
- 图1: 6张环视相机图像（带3D box真值投影）
- 图2: BEV视角3D box真值（自车坐标系）
- 图3: BEV视角3D box真值（世界坐标系）
"""

import os
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Arrow
from PIL import Image, ImageDraw
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

from utils import (
    load_nuscenes,
    get_sensor_order,
    get_sample_data_path,
    get_ego_pose,
)


# 颜色配置（参考generate_gif.py）
COLORS = {
    'bg': (245/255, 245/255, 245/255),           # 背景色
    'trajectory': (144/255, 238/255, 144/255),    # 淡绿色轨迹
    'box_edge': (255/255, 68/255, 68/255),        # 红色边框（车辆）
    'box_fill': (255/255, 68/255, 68/255),        # 红色填充
    'box_edge_other': (68/255, 255/255, 68/255),   # 绿色边框（其他车辆）
    'arrow': (173/255, 216/255, 230/255),         # 淡蓝色箭头
    'ego': (255/255, 0/255, 0/255),               # 红色自车
    'text': (60/255, 60/255, 60/255),             # 文字颜色
    'bg_rgb': (245, 245, 245),                    # 背景色（RGB 0-255）
}


def project_3d_box_to_image(nusc, sample_token, sensor_channel):
    """
    将3D box投影到指定相机图像上（参考generate_gif.py）
    返回：投影后的box列表
    """
    sample = nusc.get('sample', sample_token)
    ego_pose = get_ego_pose(nusc, sample_token)
    
    # 获取相机校准数据
    calibrated_sensor = nusc.get('calibrated_sensor', 
        nusc.get('sample_data', sample['data'][sensor_channel])['calibrated_sensor_token'])
    
    # 获取所有标注
    boxes = []
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        
        # 只处理车辆类别
        if 'vehicle' not in ann['category_name']:
            continue
            
        box = Box(
            center=ann['translation'],
            size=ann['size'],
            orientation=Quaternion(ann['rotation'])
        )
        
        # 将box转换到相机坐标系
        box.translate(-np.array(ego_pose['translation']))
        box.rotate(Quaternion(ego_pose['rotation']).inverse)
        box.translate(-np.array(calibrated_sensor['translation']))
        box.rotate(Quaternion(calibrated_sensor['rotation']).inverse)
        
        # 投影到图像平面（需要将相机内参转换为numpy数组）
        camera_intrinsic = np.array(calibrated_sensor['camera_intrinsic']).reshape(3, 3)
        corners = view_points(box.corners(), camera_intrinsic, normalize=True)
        
        # 检查是否在图像范围内
        if np.all(corners[2, :] > 0):  # 所有点在相机前方
            # view_points返回的已经是像素坐标，不需要再乘以图像尺寸
            img_corners = corners[:2, :].T
            
            # 检查是否在图像范围内（允许少量溢出）
            if np.min(img_corners[:, 0]) > -100 and np.max(img_corners[:, 0]) < 1700 and \
               np.min(img_corners[:, 1]) > -100 and np.max(img_corners[:, 1]) < 1000:
                # 根据车辆类型选择颜色
                is_car = 'car' in ann['category_name']
                boxes.append({
                    'corners': img_corners,
                    'category': ann['category_name'],
                    'color': COLORS['box_edge'] if is_car else COLORS['box_edge_other'],
                    'is_car': is_car
                })
    
    return boxes


def visualize_surround_cameras(nusc, sample_token):
    """
    生成环视相机图（带3D box投影）
    相机布局：
    第一行（上）：左前 → 前 → 右前
    第二行（下）：左后 → 后 → 右后
    """
    sensor_order = get_sensor_order()
    fig = plt.figure(figsize=(16, 8), dpi=100)
    
    camera_idx = 0
    for row_idx, row in enumerate(sensor_order):
        for sensor_channel in row:
            # 获取图像路径并读取
            img_path = get_sample_data_path(nusc, sample_token, sensor_channel)
            img = plt.imread(img_path)
            
            # 计算子图位置
            display_row = 1 - row_idx
            col_idx = camera_idx % 3
            ax = fig.add_subplot(2, 3, camera_idx + 1)
            
            # 显示图像
            ax.imshow(img)
            
            # 投影3D box到当前相机图像
            projected_boxes = project_3d_box_to_image(nusc, sample_token, sensor_channel)
            for box in projected_boxes:
                corners = box['corners']
                # NuScenes Box角点顺序:
                # 0: front-left-top, 1: front-right-top, 2: front-right-bottom, 3: front-left-bottom
                # 4: back-left-top, 5: back-right-top, 6: back-right-bottom, 7: back-left-bottom
                
                # 绘制底部四边形 (front-left-bottom -> front-right-bottom -> front-right-top -> front-left-top)
                bottom_idx = [3, 2, 1, 0, 3]
                ax.plot(
                    corners[bottom_idx, 0],
                    corners[bottom_idx, 1],
                    color=box['color'],
                    linewidth=2,
                    alpha=0.8
                )
                
                # 绘制顶部四边形 (back-left-bottom -> back-right-bottom -> back-right-top -> back-left-top)
                top_idx = [7, 6, 5, 4, 7]
                ax.plot(
                    corners[top_idx, 0],
                    corners[top_idx, 1],
                    color=box['color'],
                    linewidth=2,
                    alpha=0.8
                )
                
                # 绘制4条垂直边
                for i in range(4):
                    ax.plot(
                        [corners[i, 0], corners[i+4, 0]],
                        [corners[i, 1], corners[i+4, 1]],
                        color=box['color'],
                        linewidth=2,
                        alpha=0.8
                    )
                
                # 填充底部区域（使用前四个点形成的四边形）
                ax.fill(
                    corners[[3, 2, 1, 0], 0],
                    corners[[3, 2, 1, 0], 1],
                    color=box['color'],
                    alpha=0.2
                )
            
            # 设置子图标题和关闭坐标轴
            ax.set_title(sensor_channel, fontsize=12, pad=2)
            ax.axis('off')
            
            camera_idx += 1
    
    fig.suptitle("Surround Cameras (with 3D Box Projection)", fontsize=16, fontweight='bold', y=0.98)
    return fig


def get_object_trajectory(nusc, instance_token, max_frames=50):
    """获取对象的轨迹（历史位置）"""
    positions = []
    
    # 获取该实例的所有标注
    annotations = []
    for ann in nusc.sample_annotation:
        if ann['instance_token'] == instance_token:
            annotations.append(ann)
    
    # 按时间排序
    annotations.sort(key=lambda x: x['token'])
    
    # 获取位置
    for ann in annotations[:max_frames]:
        positions.append(np.array(ann['translation'])[:2])
    
    return positions


def visualize_bev_ego(nusc, sample_token, bev_size=50.0):
    """生成BEV图（自车坐标系）"""
    ego_pose = get_ego_pose(nusc, sample_token)
    sample = nusc.get('sample', sample_token)
    
    fig = plt.figure(figsize=(12, 12), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    
    ax.set_xlim(-bev_size, bev_size)
    ax.set_ylim(-bev_size, bev_size)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, color='gray')
    ax.set_facecolor(COLORS['bg'])
    
    # 绘制自车
    ax.scatter(0, 0, c='red', s=150, marker='s', label='Ego Vehicle', zorder=10)
    
    # 绘制所有物体
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        
        box = Box(
            center=ann['translation'],
            size=ann['size'],
            orientation=Quaternion(ann['rotation']),
            velocity=ann.get('velocity', [0, 0, 0]),
        )
        
        # 转换到自车坐标系
        box.translate(-np.array(ego_pose['translation']))
        box.rotate(Quaternion(ego_pose['rotation']).inverse)
        
        corners = box.corners()[:2, :]
        bev_corners = corners[:, [2, 3, 7, 6, 2]]
        
        # 检查是否在范围内
        if np.min(bev_corners) < -bev_size or np.max(bev_corners) > bev_size:
            continue
        
        # 绘制轨迹（淡绿色）
        trajectory = get_object_trajectory(nusc, ann['instance_token'])
        if len(trajectory) >= 2:
            traj_ego = []
            for pos in trajectory:
                pos_global = np.array([pos[0], pos[1], 0])
                pos_ego = Quaternion(ego_pose['rotation']).inverse.rotate(pos_global - np.array(ego_pose['translation']))
                traj_ego.append(pos_ego[:2])
            
            traj_ego = np.array(traj_ego)
            ax.plot(traj_ego[:, 0], traj_ego[:, 1], color=COLORS['trajectory'],
                    alpha=0.6, linewidth=2, zorder=1)
        
        # 确定颜色
        is_car = 'car' in ann['category_name']
        edge_color = COLORS['box_edge'] if is_car else COLORS['box_edge_other']
        fill_color = edge_color
        
        # 绘制box
        poly = Polygon(bev_corners.T, 
                      facecolor=fill_color,
                      edgecolor=edge_color,
                      alpha=0.3, linewidth=2)
        ax.add_patch(poly)
        
        # 绘制速度箭头（淡蓝色）
        if ann.get('velocity'):
            vel = np.array(ann['velocity'])
            vel_ego = Quaternion(ego_pose['rotation']).inverse.rotate(vel)
            arrow_length = min(np.linalg.norm(vel_ego[:2]) * 2, 5.0)
            if arrow_length > 0.1:
                center = (bev_corners[:, 0] + bev_corners[:, 2]) / 2
                arrow = Arrow(
                    center[0], center[1],
                    vel_ego[0] / np.linalg.norm(vel_ego[:2]) * arrow_length,
                    vel_ego[1] / np.linalg.norm(vel_ego[:2]) * arrow_length,
                    width=0.4, 
                    color=COLORS['arrow'],
                    alpha=0.7
                )
                ax.add_patch(arrow)
        
        # 标签
        center = (bev_corners[:, 0] + bev_corners[:, 2]) / 2
        cat_name = ann['category_name'].split('.')[-1]
        label = f"{cat_name}\n{ann['instance_token'][:6]}"
        ax.text(center[0], center[1] + 1.5, label,
                ha='center', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), zorder=5)
    
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title("BEV View - Ego Coordinate System", fontsize=14, pad=10)
    
    return fig


def visualize_bev_world(nusc, sample_token, bev_size=50.0):
    """生成BEV图（世界坐标系）"""
    ego_pose = get_ego_pose(nusc, sample_token)
    sample = nusc.get('sample', sample_token)
    
    # 获取自车位置作为中心
    ego_pos = np.array(ego_pose['translation'])[:2]
    
    fig = plt.figure(figsize=(12, 12), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    
    ax.set_xlim(ego_pos[0] - bev_size, ego_pos[0] + bev_size)
    ax.set_ylim(ego_pos[1] - bev_size, ego_pos[1] + bev_size)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, color='gray')
    ax.set_facecolor(COLORS['bg'])
    
    # 绘制自车（红色）
    ax.scatter(ego_pos[0], ego_pos[1], c='red', s=150, marker='s', label='Ego Vehicle', zorder=10)
    
    # 绘制所有物体
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        
        box = Box(
            center=ann['translation'],
            size=ann['size'],
            orientation=Quaternion(ann['rotation']),
        )
        
        corners = box.corners()[:2, :]
        bev_corners = corners[:, [2, 3, 7, 6, 2]]
        
        # 检查是否在范围内
        if np.min(bev_corners) < ego_pos[0] - bev_size or np.max(bev_corners) > ego_pos[0] + bev_size:
            continue
        if np.min(bev_corners[1]) < ego_pos[1] - bev_size or np.max(bev_corners[1]) > ego_pos[1] + bev_size:
            continue
        
        # 绘制轨迹（淡绿色）
        trajectory = get_object_trajectory(nusc, ann['instance_token'])
        if len(trajectory) >= 2:
            traj_np = np.array(trajectory)
            ax.plot(traj_np[:, 0], traj_np[:, 1], color=COLORS['trajectory'],
                    alpha=0.6, linewidth=2, zorder=1)
        
        # 确定颜色
        is_car = 'car' in ann['category_name']
        edge_color = COLORS['box_edge'] if is_car else COLORS['box_edge_other']
        fill_color = edge_color
        
        # 绘制box
        poly = Polygon(bev_corners.T, 
                      facecolor=fill_color,
                      edgecolor=edge_color,
                      alpha=0.3, linewidth=2)
        ax.add_patch(poly)
        
        # 标签
        center = (bev_corners[:, 0] + bev_corners[:, 2]) / 2
        cat_name = ann['category_name'].split('.')[-1]
        label = f"{cat_name}\n{ann['instance_token'][:6]}"
        ax.text(center[0], center[1] + 1.5, label,
                ha='center', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), zorder=5)
    
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title("BEV View - World Coordinate System", fontsize=14, pad=10)
    
    return fig


def save_figure(fig, path, dpi=100):
    """保存图像并关闭"""
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close(fig)


def main():
    DATA_ROOT = "data/nuscenes"
    VERSION = "v1.0-mini"
    OUTPUT_DIR = "output"
    
    nusc = load_nuscenes(DATA_ROOT, VERSION)
    
    if len(nusc.sample) > 0:
        # 随机选择一个单帧
        random.seed(42)  # 设置随机种子保证可重复性
        sample_token = random.choice(nusc.sample)['token']
        print(f"Visualizing random sample: {sample_token}")
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # 生成环视相机图（带3D box投影）
        print("Generating surround cameras view (with 3D box projection)...")
        fig_cams = visualize_surround_cameras(nusc, sample_token)
        cams_path = os.path.join(OUTPUT_DIR, f"surround_cams_{sample_token[:12]}.png")
        save_figure(fig_cams, cams_path)
        print(f"Saved surround cameras to: {cams_path}")
        
        # 生成BEV自车坐标系图
        print("Generating BEV ego view...")
        fig_bev_ego = visualize_bev_ego(nusc, sample_token)
        bev_ego_path = os.path.join(OUTPUT_DIR, f"bev_ego_{sample_token[:12]}.png")
        save_figure(fig_bev_ego, bev_ego_path)
        print(f"Saved BEV ego view to: {bev_ego_path}")
        
        # 生成BEV世界坐标系图
        print("Generating BEV world view...")
        fig_bev_world = visualize_bev_world(nusc, sample_token)
        bev_world_path = os.path.join(OUTPUT_DIR, f"bev_world_{sample_token[:12]}.png")
        save_figure(fig_bev_world, bev_world_path)
        print(f"Saved BEV world view to: {bev_world_path}")
        
        # 拼接三张图
        print("Combining images...")
        img_cams = Image.open(cams_path)
        img_bev_ego = Image.open(bev_ego_path)
        img_bev_world = Image.open(bev_world_path)
        
        # 统一尺寸（保持原始比例）
        target_width = 1200
        img_cams = img_cams.resize((target_width, int(img_cams.height * target_width / img_cams.width)), Image.Resampling.LANCZOS)
        img_bev_ego = img_bev_ego.resize((int(target_width/2), int(target_width/2)), Image.Resampling.LANCZOS)
        img_bev_world = img_bev_world.resize((int(target_width/2), int(target_width/2)), Image.Resampling.LANCZOS)
        
        # 计算布局
        padding = 30
        separator = 20
        
        total_width = target_width + padding * 2
        top_height = img_cams.height + padding * 2
        bottom_height = max(img_bev_ego.height, img_bev_world.height) + padding * 2 + separator
        
        total_height = top_height + bottom_height + separator
        
        # 创建画布
        combined = Image.new('RGB', (total_width, total_height), color=COLORS['bg_rgb'])
        draw = ImageDraw.Draw(combined)
        
        # 粘贴环视相机图
        combined.paste(img_cams, (padding, padding))
        
        # 绘制分隔线
        draw.line([
            (padding, top_height - padding + separator//2),
            (total_width - padding, top_height - padding + separator//2)
        ], fill=(200, 200, 200), width=2)
        
        # 粘贴BEV图（并排）
        bev_y = top_height + separator
        combined.paste(img_bev_ego, (padding, bev_y))
        combined.paste(img_bev_world, (padding + img_bev_ego.width + separator, bev_y))
        
        # 添加标题
        draw.text((total_width//2 - 120, 10), "NuScenes Dataset Visualization", 
                  fill=(50, 50, 50), font_size=20)
        
        # 添加子图标题
        draw.text((padding, top_height - padding + separator + 10), 
                  "BEV View (Ego Coordinate)", fill=(60, 60, 60), font_size=14)
        draw.text((padding + img_bev_ego.width + separator, top_height - padding + separator + 10), 
                  "BEV View (World Coordinate)", fill=(60, 60, 60), font_size=14)
        
        # 保存组合图
        combined_path = os.path.join(OUTPUT_DIR, f"combined_{sample_token[:12]}.png")
        combined.save(combined_path)
        print(f"Saved combined view to: {combined_path}")
        
        print("\n✅ All visualizations completed!")
    else:
        print("No samples found in dataset!")


if __name__ == "__main__":
    main()
