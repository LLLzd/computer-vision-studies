#!/usr/bin/env python3
"""
生成场景GIF动图（环视相机或BEV视角）
"""

import os
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Arrow
import numpy as np
from PIL import Image

from utils import (
    load_nuscenes,
    get_sensor_order,
    project_box_to_bev,
    is_box_in_range,
    get_sample_data_path,
    get_ego_pose,
    iter_scene_samples,
)
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion


def project_3d_box_to_image(nusc, sample_token, sensor_channel):
    """
    将3D box投影到指定相机图像上
    返回：[(x1, y1), (x2, y2), ...] 图像坐标系中的box角点，None表示不在视野内
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
            
            # 检查是否在图像范围内
            if np.min(img_corners[:, 0]) > -100 and np.max(img_corners[:, 0]) < 1700 and \
               np.min(img_corners[:, 1]) > -100 and np.max(img_corners[:, 1]) < 1000:
                boxes.append({
                    'corners': img_corners,
                    'category': ann['category_name'],
                    'color': '#FF4444' if 'car' in ann['category_name'] else '#44FF44'
                })
    
    return boxes


def generate_surround_frame(nusc, sample_token, scene_name, frame_idx, output_dir):
    """
    生成环视相机单帧，包含3D box真值投影
    相机布局：
    第一行（上）：左前(CAM_FRONT_LEFT) → 前(CAM_FRONT) → 右前(CAM_FRONT_RIGHT)
    第二行（下）：左后(CAM_BACK_LEFT) → 后(CAM_BACK) → 右后(CAM_BACK_RIGHT)
    """
    sensor_order = get_sensor_order()
    
    # 创建大图，包含6个相机视图
    fig = plt.figure(figsize=(17.9, 8.93), dpi=100)
    camera_idx = 0
    
    # 遍历相机顺序
    for row_idx, row in enumerate(sensor_order):
        for sensor_channel in row:
            # 获取图像路径并读取
            img_path = get_sample_data_path(nusc, sample_token, sensor_channel)
            img = plt.imread(img_path)
            
            # 计算子图位置
            # display_row: 0=上排, 1=下排
            # col_idx: 0=左, 1=中, 2=右
            display_row = 1 - row_idx
            col_idx = camera_idx % 3
            
            # 创建子图区域
            ax = fig.add_axes([0.02 + col_idx * 0.32, 0.05 + display_row * 0.45, 0.30, 0.42])
            
            # 显示图像
            ax.imshow(img)
            
            # 投影3D box到当前相机图像
            projected_boxes = project_3d_box_to_image(nusc, sample_token, sensor_channel)
            for box in projected_boxes:
                corners = box['corners']
                # NuScenes Box角点顺序:
                # 0: front-left-top, 1: front-right-top, 2: front-right-bottom, 3: front-left-bottom
                # 4: back-left-top, 5: back-right-top, 6: back-right-bottom, 7: back-left-bottom
                
                # 绘制底部四边形
                bottom_idx = [3, 2, 1, 0, 3]
                ax.plot(
                    corners[bottom_idx, 0],
                    corners[bottom_idx, 1],
                    color=box['color'],
                    linewidth=2,
                    alpha=0.8
                )
                
                # 绘制顶部四边形
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
                
                # 填充底部区域
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
    
    # 设置大图标题
    fig.suptitle(f"{scene_name} - Frame {frame_idx}", fontsize=16, y=0.98)
    
    # 保存帧图像
    frame_path = os.path.join(output_dir, f"frame_{frame_idx:03d}.png")
    fig.savefig(frame_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    return frame_path


def generate_bev_frame(nusc, sample_token, scene_name, frame_idx, output_dir, show_annotation=True):
    """生成BEV单帧"""
    ego_pose = get_ego_pose(nusc, sample_token)
    sample = nusc.get('sample', sample_token)
    
    fig = plt.figure(figsize=(20, 20), dpi=100)
    ax = fig.add_axes([0.05, 0.05, 0.90, 0.90])
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.scatter(0, 0, c='red', s=200, marker='s', label='Ego Vehicle', zorder=10)
    
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        if ann['category_name'] == 'vehicle.car':
            box = Box(center=ann['translation'], size=ann['size'], orientation=Quaternion(ann['rotation']))
            bev_corners = project_box_to_bev(box, ego_pose)
            if is_box_in_range(bev_corners):
                # 画box
                poly = Polygon(bev_corners.T, facecolor='#00FFFF', alpha=0.3, edgecolor='#00FFFF', linewidth=2)
                ax.add_patch(poly)
                
                if show_annotation:
                    # 朝向箭头
                    center = (bev_corners[:, 0] + bev_corners[:, 2]) / 2
                    quat = Quaternion(ann['rotation'])
                    forward = quat.rotate(np.array([0, 1, 0]))[:2]
                    forward = Quaternion(ego_pose['rotation']).inverse.rotate(np.array([forward[0], forward[1], 0]))[:2]
                    arrow_length = 2.0
                    arrow = Arrow(center[0], center[1], forward[0] * arrow_length, forward[1] * arrow_length,
                                width=0.5, color='#00FFFF', alpha=0.8)
                    ax.add_patch(arrow)
                    
                    # 标签
                    label = f"car\n{ann['instance_token'][:6]}"
                    ax.text(center[0], center[1] + 1.8, label,
                           ha='center', va='bottom', fontsize=12,
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    ax.set_xlabel('X (meters)', fontsize=14)
    ax.set_ylabel('Y (meters)', fontsize=14)
    ax.set_title(f"BEV View - {scene_name} - Frame {frame_idx}", fontsize=16, pad=10)
    
    frame_path = os.path.join(output_dir, f"frame_{frame_idx:03d}.png")
    fig.savefig(frame_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    return frame_path


def generate_gif(nusc, output_path, mode='surround', scene_idx=0, max_frames=15, duration=500):
    """
    生成GIF动图
    mode: 'surround' - 环视相机(带3D box投影), 'bev' - BEV视角, 'bev_annotated' - BEV带标注
    """
    scene = nusc.scene[scene_idx]
    scene_name = scene['name']
    
    tmp_dir = f"output/tmp_frames_{mode}"
    os.makedirs(tmp_dir, exist_ok=True)
    
    print(f"Processing {scene_name} ({mode})...")
    
    frame_paths = []
    
    for sample, frame_idx in iter_scene_samples(nusc, scene_idx, max_frames):
        print(f"  Frame {frame_idx}...")
        
        if mode == 'surround':
            frame_path = generate_surround_frame(nusc, sample['token'], scene_name, frame_idx, tmp_dir)
        elif mode == 'bev':
            frame_path = generate_bev_frame(nusc, sample['token'], scene_name, frame_idx, tmp_dir, show_annotation=False)
        elif mode == 'bev_annotated':
            frame_path = generate_bev_frame(nusc, sample['token'], scene_name, frame_idx, tmp_dir, show_annotation=True)
        
        frame_paths.append(frame_path)
    
    # 生成GIF
    print("\nGenerating GIF...")
    images = []
    for fp in frame_paths:
        img = Image.open(fp)
        if mode == 'surround':
            img = img.resize((1790, 893), Image.Resampling.LANCZOS)
        else:
            img = img.resize((1790, 1790), Image.Resampling.LANCZOS)
        images.append(img)
    
    if images:
        images[0].save(output_path, save_all=True, append_images=images[1:], duration=duration, loop=0)
        print(f"GIF saved to: {output_path}")
    
    # 清理临时文件
    shutil.rmtree(tmp_dir)


def main():
    os.makedirs("output", exist_ok=True)
    nusc = load_nuscenes()
    
    # 生成环视相机GIF（带3D box投影）
    generate_gif(nusc, "output/scene0_surround.gif", mode='surround')
    
    # 生成BEV GIF
    generate_gif(nusc, "output/scene0_bev.gif", mode='bev')
    
    # 生成带标注的BEV GIF
    generate_gif(nusc, "output/scene0_bev_annotated.gif", mode='bev_annotated')


if __name__ == "__main__":
    main()
