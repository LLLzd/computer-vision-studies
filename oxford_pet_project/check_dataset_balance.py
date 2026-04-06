"""
检查数据集的前景/背景比例
"""

import os
import numpy as np
from PIL import Image
from torchvision.datasets import OxfordIIITPet
from config import DATA_DIR, transform, label_transform

# 加载数据集
dataset = OxfordIIITPet(
    root=DATA_DIR,
    split='trainval',
    target_types='segmentation',
    transform=transform,
    target_transform=label_transform,
    download=False
)

print(f"数据集大小: {len(dataset)}")

# 统计前景/背景比例
total_pixels = 0
foreground_pixels = 0

for i in range(min(100, len(dataset))):  # 只检查前100张图片
    if i % 10 == 0:
        print(f"处理第 {i} 张图片...")
    
    image, label = dataset[i]
    label_np = label.squeeze().numpy()
    
    # 计算像素总数
    img_pixels = label_np.size
    total_pixels += img_pixels
    
    # 计算前景像素数（值为1的像素）
    fg_pixels = np.sum(label_np == 1)
    foreground_pixels += fg_pixels

# 计算比例
foreground_ratio = foreground_pixels / total_pixels * 100
background_ratio = 100 - foreground_ratio

print(f"\n前景像素比例: {foreground_ratio:.2f}%")
print(f"背景像素比例: {background_ratio:.2f}%")

if foreground_ratio < 10:
    print("\n⚠️  警告：前景像素比例过低，可能导致模型偏向预测背景")
    print("建议：使用加权损失函数或数据增强来解决类别不平衡问题")