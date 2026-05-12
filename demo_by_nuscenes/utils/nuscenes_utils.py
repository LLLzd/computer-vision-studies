#!/usr/bin/env python3
"""
NuScenes 数据集可视化公共工具模块
"""

import os
import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion


def load_nuscenes(data_root: str = "data/nuscenes", version: str = "v1.0-mini") -> NuScenes:
    """加载 NuScenes 数据集"""
    print(f"Loading NuScenes dataset from {data_root}...")
    nusc = NuScenes(version=version, dataroot=data_root, verbose=False)
    return nusc


def get_sensor_order() -> list:
    """获取相机在车辆上的布局顺序
    第一行从左到右：左前、前、右前
    第二行从左到右：左后、后、右后
    """
    return [
        ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"],
        ["CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"],
    ]


def project_box_to_bev(box: Box, ego_pose: dict) -> np.ndarray:
    """将box从全局坐标系投影到BEV坐标系"""
    box.translate(-np.array(ego_pose['translation']))
    box.rotate(Quaternion(ego_pose['rotation']).inverse)
    corners = box.corners()[:2, :]
    return corners[:, [2, 3, 7, 6, 2]]


def is_box_in_range(bev_corners: np.ndarray, bev_size: float = 50.0) -> bool:
    """检查box是否在BEV范围内"""
    min_x = np.min(bev_corners[0, :])
    max_x = np.max(bev_corners[0, :])
    min_y = np.min(bev_corners[1, :])
    max_y = np.max(bev_corners[1, :])
    return (min_x >= -bev_size and max_x <= bev_size and
            min_y >= -bev_size and max_y <= bev_size)


def get_category_color(category_name: str) -> str:
    """获取类别的颜色映射"""
    color_map = {
        'vehicle.car': '#00FFFF',      # 青色
        'vehicle.truck': '#FF00FF',    # 洋红
        'vehicle.bus': '#00FF00',      # 绿色
        'vehicle.motorcycle': '#FFA500', # 橙色
        'human.pedestrian.adult': '#0080FF',  # 蓝色
        'human.pedestrian.child': '#0080FF',  # 蓝色
        'movable_object.barrier': '#FFFF00',  # 黄色
        'movable_object.trafficcone': '#FF4500', # 橙红色
    }
    return color_map.get(category_name, '#FFFFFF')  # 默认白色


def get_sample_data_path(nusc: NuScenes, sample_token: str, sensor_channel: str) -> str:
    """获取传感器数据路径"""
    sample = nusc.get('sample', sample_token)
    cam_data = nusc.get('sample_data', sample['data'][sensor_channel])
    return os.path.join(nusc.dataroot, cam_data['filename'])


def get_ego_pose(nusc: NuScenes, sample_token: str) -> dict:
    """获取自车姿态"""
    sample = nusc.get('sample', sample_token)
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    return nusc.get('ego_pose', lidar_data['ego_pose_token'])


def iter_scene_samples(nusc: NuScenes, scene_idx: int = 0, max_frames: int = None):
    """迭代场景中的样本"""
    scene = nusc.scene[scene_idx]
    current_token = scene['first_sample_token']
    frame_idx = 0
    
    while current_token != '' and (max_frames is None or frame_idx < max_frames):
        sample = nusc.get('sample', current_token)
        yield sample, frame_idx
        current_token = sample['next']
        frame_idx += 1
