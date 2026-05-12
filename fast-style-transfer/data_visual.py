import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def visualize_images(content_dir, output_dir, num_images=4):
    """可视化内容图像和风格迁移结果
    
    Args:
        content_dir: 内容图像目录
        output_dir: 输出图像目录
        num_images: 要可视化的图像数量
    """
    # 获取内容图像列表
    content_images = []
    for root, _, files in os.walk(content_dir):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                content_images.append(os.path.join(root, file))
    
    # 随机选择图像
    selected_images = random.sample(content_images, min(num_images, len(content_images)))
    
    # 获取对应的输出图像
    output_images = []
    for content_path in selected_images:
        content_name = os.path.basename(content_path)
        output_name = f"styled_{content_name}"
        output_path = os.path.join(output_dir, output_name)
        if os.path.exists(output_path):
            output_images.append(output_path)
        else:
            output_images.append(None)
    
    # 创建子图
    fig, axes = plt.subplots(len(selected_images), 2, figsize=(12, 4 * len(selected_images)))
    
    for i, (content_path, output_path) in enumerate(zip(selected_images, output_images)):
        # 显示内容图像
        content_img = Image.open(content_path)
        axes[i, 0].imshow(content_img)
        axes[i, 0].set_title(f"输入图像: {os.path.basename(content_path)}")
        axes[i, 0].axis('off')
        
        # 显示输出图像
        if output_path and os.path.exists(output_path):
            output_img = Image.open(output_path)
            axes[i, 1].imshow(output_img)
            axes[i, 1].set_title(f"风格化结果: {os.path.basename(output_path)}")
        else:
            axes[i, 1].text(0.5, 0.5, "无对应输出图像", ha='center', va='center')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "visualization.png"), dpi=150, bbox_inches='tight')
    print(f"可视化结果保存到: {os.path.join(output_dir, 'visualization.png')}")

def compare_styles(content_path, output_paths, style_names):
    """比较不同风格的迁移结果
    
    Args:
        content_path: 内容图像路径
        output_paths: 不同风格的输出图像路径列表
        style_names: 风格名称列表
    """
    # 创建子图
    fig, axes = plt.subplots(1, len(output_paths) + 1, figsize=(4 * (len(output_paths) + 1), 6))
    
    # 显示内容图像
    content_img = Image.open(content_path)
    axes[0].imshow(content_img)
    axes[0].set_title("原始图像")
    axes[0].axis('off')
    
    # 显示不同风格的输出
    for i, (output_path, style_name) in enumerate(zip(output_paths, style_names)):
        if os.path.exists(output_path):
            output_img = Image.open(output_path)
            axes[i+1].imshow(output_img)
            axes[i+1].set_title(f"{style_name}风格")
        else:
            axes[i+1].text(0.5, 0.5, f"{style_name}风格\n无结果", ha='center', va='center')
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(output_paths[0]), "style_comparison.png"), dpi=150, bbox_inches='tight')
    print(f"风格比较结果保存到: {os.path.join(os.path.dirname(output_paths[0]), 'style_comparison.png')}")

if __name__ == "__main__":
    from config import CONTENT_DIR, OUTPUT_DIR
    
    # 可视化内容图像和输出结果
    visualize_images(CONTENT_DIR, OUTPUT_DIR)
    
    # 示例：比较不同风格
    # content_img = os.path.join(CONTENT_DIR, "your_photo.jpg")
    # style_outputs = [
    #     os.path.join(OUTPUT_DIR, "candy_style.jpg"),
    #     os.path.join(OUTPUT_DIR, "mosaic_style.jpg"),
    #     os.path.join(OUTPUT_DIR, "rain_princess_style.jpg"),
    #     os.path.join(OUTPUT_DIR, "udnie_style.jpg")
    # ]
    # style_names = ["Candy", "Mosaic", "Rain Princess", "Udnie"]
    # compare_styles(content_img, style_outputs, style_names)
