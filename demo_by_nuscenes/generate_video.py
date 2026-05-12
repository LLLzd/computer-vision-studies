#!/usr/bin/env python3
"""
生成场景可视化视频
"""

import os
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
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
from pyquaternion import Quaternion


def draw_box_bev(ax: plt.Axes, corners: np.ndarray, color: str, linewidth: int = 2):
    """在BEV视图中绘制box"""
    poly = Polygon(corners.T, facecolor=color, alpha=0.3, edgecolor=color, linewidth=linewidth)
    ax.add_patch(poly)


def generate_frame(nusc, sample_token, scene_name, frame_idx):
    """生成单帧组合视图"""
    sensor_order = get_sensor_order()
    
    # 环视相机
    fig1 = plt.figure(figsize=(17.9, 8.93), dpi=100)
    camera_idx = 0
    for row_idx, row in enumerate(sensor_order):
        for sensor_channel in row:
            img_path = get_sample_data_path(nusc, sample_token, sensor_channel)
            img = plt.imread(img_path)
            display_row = 1 - row_idx
            col_idx = camera_idx % 3
            ax = fig1.add_axes([0.02 + col_idx * 0.32, 0.05 + display_row * 0.45, 0.30, 0.42])
            ax.imshow(img)
            ax.set_title(sensor_channel, fontsize=12, pad=2)
            ax.axis('off')
            camera_idx += 1
    fig1.suptitle(f"{scene_name} - Frame {frame_idx}", fontsize=16, y=0.98)
    fig1.savefig("/tmp/surround.png", dpi=100, bbox_inches='tight')
    plt.close(fig1)
    img_surround = Image.open("/tmp/surround.png").resize((1790, 893), Image.Resampling.LANCZOS)
    
    # BEV
    ego_pose = get_ego_pose(nusc, sample_token)
    sample = nusc.get('sample', sample_token)
    
    fig2 = plt.figure(figsize=(20, 20), dpi=100)
    ax = fig2.add_axes([0.05, 0.05, 0.90, 0.90])
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
                draw_box_bev(ax, bev_corners, '#00FFFF')
    
    ax.set_xlabel('X (meters)', fontsize=14)
    ax.set_ylabel('Y (meters)', fontsize=14)
    ax.set_title("BEV View (Cars Only)", fontsize=16, pad=10)
    
    fig2.savefig("/tmp/bev.png", dpi=100, bbox_inches='tight')
    plt.close(fig2)
    img_bev = Image.open("/tmp/bev.png").resize((1790, 1790), Image.Resampling.LANCZOS)
    
    # 拼接
    combined_height = 893 + 1790
    img_combined = Image.new('RGB', (1790, combined_height))
    img_combined.paste(img_surround, (0, 0))
    img_combined.paste(img_bev, (0, 893))
    
    return np.array(img_combined)


def generate_video(nusc, output_path):
    """生成视频"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 5
    out = None
    
    print("Generating video...")
    
    for scene_idx in range(len(nusc.scene)):
        scene = nusc.scene[scene_idx]
        scene_name = scene['name']
        print(f"  Processing {scene_name}...")
        
        frame_idx = 0
        max_frames = 10
        
        for sample, idx in iter_scene_samples(nusc, scene_idx, max_frames):
            if idx % 2 == 0:  # 每2帧抽1帧
                frame = generate_frame(nusc, sample['token'], scene_name, idx)
                
                if out is None:
                    height, width = frame.shape[:2]
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            frame_idx += 1
        
        print(f"    Done {scene_name}")
    
    if out is not None:
        out.release()
        print(f"\nVideo saved to: {output_path}")


def main():
    os.makedirs("output", exist_ok=True)
    nusc = load_nuscenes()
    generate_video(nusc, "output/scenes_video.mp4")


if __name__ == "__main__":
    main()
