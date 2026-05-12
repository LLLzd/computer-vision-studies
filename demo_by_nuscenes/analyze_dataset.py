#!/usr/bin/env python3
"""
NuScenes 数据集分析工具
- 统计场景信息
- 分析类别分布
- 计算帧率和时长
- 展示对象属性特征
"""

import os
import numpy as np
from collections import Counter
from utils import load_nuscenes


def analyze_scenes(nusc):
    """分析场景信息"""
    print("\n" + "="*70)
    print("🎬 SCENE ANALYSIS")
    print("="*70)
    
    scenes_info = []
    total_frames = 0
    total_duration = 0.0
    
    for i, scene in enumerate(nusc.scene):
        first_token = scene['first_sample_token']
        last_token = scene['last_sample_token']
        nbr_samples = scene['nbr_samples']
        
        first_sample = nusc.get('sample', first_token)
        last_sample = nusc.get('sample', last_token)
        
        first_lidar = nusc.get('sample_data', first_sample['data']['LIDAR_TOP'])
        last_lidar = nusc.get('sample_data', last_sample['data']['LIDAR_TOP'])
        
        duration = (last_lidar['timestamp'] - first_lidar['timestamp']) / 1e6  # 秒
        fps = nbr_samples / duration if duration > 0 else 0
        
        total_frames += nbr_samples
        total_duration += duration
        
        scenes_info.append({
            'name': scene['name'],
            'index': i,
            'samples': nbr_samples,
            'duration': duration,
            'fps': fps
        })
        
        print(f"\nScene {i:2d}: {scene['name']}")
        print(f"  ├─ Frames: {nbr_samples}")
        print(f"  ├─ Duration: {duration:.2f} s")
        print(f"  └─ FPS: {fps:.2f}")
    
    print(f"\n" + "-"*70)
    print(f"Total Scenes: {len(nusc.scene)}")
    print(f"Total Frames: {total_frames}")
    print(f"Total Duration: {total_duration:.2f} s ({total_duration/60:.1f} min)")
    print(f"Overall FPS: {total_frames/total_duration:.2f}")
    
    return scenes_info


def analyze_categories(nusc):
    """分析类别分布"""
    print("\n" + "="*70)
    print("🏷️ CATEGORY ANALYSIS")
    print("="*70)
    
    category_counts = Counter()
    for ann in nusc.sample_annotation:
        category_counts[ann['category_name']] += 1
    
    print(f"\nTotal Categories: {len(nusc.category)}")
    print(f"\nCategory Distribution:")
    print("-" * 70)
    
    # 按数量排序
    sorted_categories = category_counts.most_common()
    max_name_len = max(len(cat) for cat, _ in sorted_categories)
    
    for i, (cat, count) in enumerate(sorted_categories):
        percentage = (count / sum(category_counts.values())) * 100
        bar_length = int(percentage / 2)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"{i+1:2d}. {cat:{max_name_len}} : {count:6d} [{percentage:5.2f}%] {bar}")
    
    # 类别分组统计
    print("\nCategory Groups:")
    print("-" * 70)
    
    groups = {
        "🚗 Vehicles": [c for c in category_counts.keys() if "vehicle" in c],
        "🚶 Pedestrians": [c for c in category_counts.keys() if "human.pedestrian" in c],
        "📦 Movable Objects": [c for c in category_counts.keys() if "movable_object" in c],
        "🏗️ Static Objects": [c for c in category_counts.keys() if "static_object" in c],
        "🐶 Animals": [c for c in category_counts.keys() if "animal" in c],
    }
    
    for group_name, categories in groups.items():
        if categories:
            total = sum(category_counts[c] for c in categories)
            print(f"{group_name}: {total} objects")
            for cat in sorted(categories):
                print(f"  └─ {cat}: {category_counts[cat]}")


def analyze_object_properties(nusc):
    """分析对象属性特征"""
    print("\n" + "="*70)
    print("📊 OBJECT PROPERTY ANALYSIS")
    print("="*70)
    
    sizes = []
    velocities = []
    instances = set()
    
    for ann in nusc.sample_annotation:
        # 尺寸信息
        sizes.append(ann['size'])
        
        # 速度信息
        if 'velocity' in ann and ann['velocity'] is not None:
            vel_mag = np.linalg.norm(ann['velocity'][:2])  # 平面速度
            velocities.append(vel_mag)
        
        # 实例ID
        instances.add(ann['instance_token'])
    
    # 尺寸统计
    sizes = np.array(sizes)
    print("\nSize Statistics (width, length, height):")
    print(f"  Objects: {len(sizes)}")
    print(f"  Width  - Mean: {np.mean(sizes[:, 0]):.2f}m, Std: {np.std(sizes[:, 0]):.2f}m, Range: [{np.min(sizes[:, 0]):.2f}, {np.max(sizes[:, 0]):.2f}]")
    print(f"  Length - Mean: {np.mean(sizes[:, 1]):.2f}m, Std: {np.std(sizes[:, 1]):.2f}m, Range: [{np.min(sizes[:, 1]):.2f}, {np.max(sizes[:, 1]):.2f}]")
    print(f"  Height - Mean: {np.mean(sizes[:, 2]):.2f}m, Std: {np.std(sizes[:, 2]):.2f}m, Range: [{np.min(sizes[:, 2]):.2f}, {np.max(sizes[:, 2]):.2f}]")
    
    # 速度统计
    velocities = np.array(velocities)
    print("\nVelocity Statistics:")
    print(f"  Objects with velocity: {len(velocities)}")
    if len(velocities) > 0:
        print(f"  Mean: {np.mean(velocities):.2f} m/s")
        print(f"  Std: {np.std(velocities):.2f} m/s")
        print(f"  Max: {np.max(velocities):.2f} m/s")
        print(f"  Min: {np.min(velocities):.2f} m/s")
    else:
        print("  (No velocity data available in annotations)")
    
    # 实例统计
    print(f"\nUnique Instances: {len(instances)}")
    
    # 每帧平均对象数
    avg_objects_per_frame = len(nusc.sample_annotation) / len(nusc.sample)
    print(f"Average objects per frame: {avg_objects_per_frame:.2f}")


def analyze_sensors(nusc):
    """分析传感器配置"""
    print("\n" + "="*70)
    print("📡 SENSOR ANALYSIS")
    print("="*70)
    
    sensor_types = Counter()
    for sensor in nusc.sensor:
        channel = sensor['channel']
        sensor_types[channel.split('_')[0]] += 1
    
    print(f"\nTotal Sensors: {len(nusc.sensor)}")
    print("\nSensor Types:")
    for sensor_type, count in sensor_types.items():
        print(f"  {sensor_type}: {count}")
    
    print("\nSensor Channels:")
    for sensor in nusc.sensor:
        print(f"  - {sensor['channel']}")


def main():
    """主函数"""
    nusc = load_nuscenes()
    
    print("\n" + "="*70)
    print("🚀 NuScenes Dataset Comprehensive Analysis")
    print("="*70)
    
    # 分析场景
    analyze_scenes(nusc)
    
    # 分析类别
    analyze_categories(nusc)
    
    # 分析对象属性
    analyze_object_properties(nusc)
    
    # 分析传感器
    analyze_sensors(nusc)
    
    print("\n" + "="*70)
    print("✅ Analysis Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
