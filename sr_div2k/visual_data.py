import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# 数据集路径
TRAIN_HR = "data/DIV2K_train_HR"
TRAIN_LR = "data/DIV2K_train_LR_bicubic/X3"
VAL_HR = "data/DIV2K_valid_HR"
VAL_LR = "data/DIV2K_valid_LR_bicubic/X3"

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 随机选择2组图片（从train和valid中选择）
def visualize_data(num_samples=2):
    """可视化训练数据
    随机选择2组图片，从train和valid中选择，展示低分辨率输入和高分辨率标签
    """
    # 创建outputs目录
    os.makedirs("outputs", exist_ok=True)
    
    # 收集所有数据集
    datasets = [
        ("train", TRAIN_HR, TRAIN_LR),
        ("valid", VAL_HR, VAL_LR)
    ]
    
    # 收集所有可用的图像对
    all_image_pairs = []
    for dataset_name, hr_dir, lr_dir in datasets:
        hr_files = sorted([f for f in os.listdir(hr_dir) if f.endswith('.png')])
        lr_files = sorted([f for f in os.listdir(lr_dir) if f.endswith('.png')])
        
        for hr_file in hr_files:
            lr_file = hr_file.replace('.png', 'x3.png')
            hr_path = os.path.join(hr_dir, hr_file)
            lr_path = os.path.join(lr_dir, lr_file)
            
            if os.path.exists(lr_path):
                all_image_pairs.append((dataset_name, hr_path, lr_path))
    
    # 确保有足够的图像对
    if len(all_image_pairs) < num_samples:
        print(f"❌ 可用图像对数量不足，只有 {len(all_image_pairs)} 对")
        return
    
    # 随机选择2个样本（不固定种子，每次都随机）
    selected_pairs = random.sample(all_image_pairs, num_samples)
    
    # 创建可视化
    plt.figure(figsize=(10, 8))
    
    for i, (dataset_name, hr_path, lr_path) in enumerate(selected_pairs, 1):
        # 读取图像
        hr_img = Image.open(hr_path)
        lr_img = Image.open(lr_path)
        
        # 获取图像尺寸
        lr_width, lr_height = lr_img.size
        hr_width, hr_height = hr_img.size
        
        # 显示低分辨率输入（模型输入）
        plt.subplot(num_samples, 2, (i-1)*2 + 1)
        plt.imshow(np.array(lr_img))
        plt.title(f"{dataset_name} 输入: 低分辨率图像 (尺寸: {lr_width}x{lr_height})")
        plt.axis('off')
        
        # 显示高分辨率标签（目标输出）
        plt.subplot(num_samples, 2, (i-1)*2 + 2)
        plt.imshow(np.array(hr_img))
        plt.title(f"{dataset_name} 标签: 高分辨率图像 (尺寸: {hr_width}x{hr_height})")
        plt.axis('off')
    
    plt.tight_layout()
    output_path = "outputs/data_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 可视化结果已保存至: {output_path}")
    # 不显示弹窗

if __name__ == "__main__":
    print("📊 开始可视化训练数据...")
    visualize_data(num_samples=2)
