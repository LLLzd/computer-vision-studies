"""
统计 Pascal VOC 2012 数据集中有分割标注的图像数量
"""

import os

# 数据目录
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# 图像列表文件路径
list_file = os.path.join(DATA_DIR, 'VOC2012', 'ImageSets', 'Main', 'trainval.txt')

# 分割标注目录
seg_class_dir = os.path.join(DATA_DIR, 'VOC2012', 'SegmentationClass')

# 统计变量
total_images = 0
has_annotation = 0
no_annotation = 0

# 读取图像列表
with open(list_file, 'r') as f:
    for line in f:
        img_id = line.strip()
        img_path = os.path.join(DATA_DIR, 'VOC2012', 'JPEGImages', f'{img_id}.jpg')
        
        # 只统计实际存在的图像
        if os.path.exists(img_path):
            total_images += 1
            
            # 检查是否有分割标注
            seg_path = os.path.join(seg_class_dir, f'{img_id}.png')
            if os.path.exists(seg_path):
                has_annotation += 1
            else:
                no_annotation += 1

# 计算占比
has_annotation_ratio = (has_annotation / total_images) * 100
no_annotation_ratio = (no_annotation / total_images) * 100

# 打印结果
print("Pascal VOC 2012 数据集分割标注统计:")
print(f"总图像数: {total_images}")
print(f"有分割标注的图像数: {has_annotation} ({has_annotation_ratio:.2f}%)")
print(f"无分割标注的图像数: {no_annotation} ({no_annotation_ratio:.2f}%)")
