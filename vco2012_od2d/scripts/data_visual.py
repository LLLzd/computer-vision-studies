import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xml.etree.ElementTree as ET

# 添加项目根目录到Python路径
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess import Preprocessor
from dataset import VOCDataset

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']  # 用于显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用于显示负号

# 数据集路径
DATA_DIR = 'data'
JPEGIMAGES_DIR = os.path.join(DATA_DIR, 'VOC2012', 'JPEGImages')
ANNOTATIONS_DIR = os.path.join(DATA_DIR, 'VOC2012', 'Annotations')
OUTPUT_DIR = 'outputs'

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 初始化预处理器
preprocessor = Preprocessor(target_size=(512, 512))
# 初始化数据集
voc_dataset = VOCDataset(split='train')

def get_image_paths():
    """获取所有图像路径"""
    return [f for f in os.listdir(JPEGIMAGES_DIR) if f.endswith('.jpg')]

def visualize_original_image(image_path, annotation_path, ax):
    """可视化原始图像和标签"""
    # 读取原始图像
    image = Image.open(image_path)
    width, height = image.size
    
    # 解析原始标注
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    boxes = []
    for obj in root.findall('object'):
        cls = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))
        boxes.append((cls, xmin, ymin, xmax, ymax))
    
    # 显示图像
    ax.imshow(image)
    
    # 绘制边界框
    for cls, xmin, ymin, xmax, ymax in boxes:
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                               linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin - 5, cls, color='g', fontsize=8, 
               bbox=dict(facecolor='white', alpha=0.7))
    
    # 添加图像信息
    ax.set_title(f"原始图像 (尺寸: {width}x{height})")
    ax.axis('off')

def visualize_preprocessed_image(image_path, annotation_path, ax):
    """可视化预处理后的图像和标签"""
    # 使用预处理器处理图像和标注
    image, boxes = preprocessor.preprocess_sample(image_path, annotation_path)
    width, height = image.size
    
    # 显示图像
    ax.imshow(image)
    
    # 绘制边界框
    for cls, xmin, ymin, xmax, ymax in boxes:
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                               linewidth=2, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin - 5, cls, color='b', fontsize=8, 
               bbox=dict(facecolor='white', alpha=0.7))
    
    # 添加图像信息
    ax.set_title(f"预处理后图像 (尺寸: {width}x{height})")
    ax.axis('off')

def visualize_heatmap_with_boxes(heatmap, wh, ax, class_names, boxes):
    """可视化heatmap并在上面画边界框
    
    Args:
        heatmap: 热图数据
        wh: 宽高数据
        ax: matplotlib轴
        class_names: 类别名称列表
        boxes: 原始边界框列表 [(cls, xmin, ymin, xmax, ymax), ...]
    """
    # 叠加显示多个类别的热图
    combined_heatmap = np.zeros_like(heatmap[0])
    for i in range(heatmap.shape[0]):
        if np.max(heatmap[i]) > 0:
            combined_heatmap = np.maximum(combined_heatmap, heatmap[i])
    
    # 显示叠加后的热图
    im = ax.imshow(combined_heatmap, cmap='jet', vmin=0, vmax=1)
    
    # 添加颜色条
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 在热图上画边界框（黄色）
    # 计算缩放因子
    scale_x = heatmap.shape[2] / 512  # HEATMAP_SIZE[1] / IMAGE_SIZE[0]
    scale_y = heatmap.shape[1] / 512  # HEATMAP_SIZE[0] / IMAGE_SIZE[1]
    
    for cls, xmin, ymin, xmax, ymax in boxes:
        # 将原图坐标转换到热图坐标
        hm_xmin = int(xmin * scale_x)
        hm_ymin = int(ymin * scale_y)
        hm_xmax = int(xmax * scale_x)
        hm_ymax = int(ymax * scale_y)
        
        # 绘制边界框
        rect = patches.Rectangle((hm_xmin, hm_ymin), hm_xmax - hm_xmin, hm_ymax - hm_ymin, 
                               linewidth=2, edgecolor='yellow', facecolor='none')
        ax.add_patch(rect)
        ax.text(hm_xmin, hm_ymin - 2, cls, color='yellow', fontsize=6, 
               bbox=dict(facecolor='black', alpha=0.5))
    
    # 添加标题，显示所有类别
    class_names_str = ', '.join(class_names)
    ax.set_title(f"热图+边界框 (类别: {class_names_str[:30]}...)")
    ax.axis('off')

def get_image_classes(annotation_path):
    """获取图像中的所有类别"""
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    classes = []
    for obj in root.findall('object'):
        cls = obj.find('name').text
        if cls not in classes:
            classes.append(cls)
    return classes

def main():
    """主函数"""
    # 检查数据集是否存在
    if not os.path.exists(JPEGIMAGES_DIR):
        print(f"Error: JPEGImages directory not found at {JPEGIMAGES_DIR}")
        print("Please make sure the Pascal VOC 2012 dataset is downloaded")
        return
    
    if not os.path.exists(ANNOTATIONS_DIR):
        print(f"Error: Annotations directory not found at {ANNOTATIONS_DIR}")
        print("Please make sure the Pascal VOC 2012 dataset is downloaded")
        return
    
    # 从数据集中随机选择4张图片
    if len(voc_dataset) < 4:
        print("Error: Not enough images in dataset")
        return
    
    # 随机选择4个索引
    selected_indices = random.sample(range(len(voc_dataset)), 4)
    selected_images = [voc_dataset.image_paths[idx] for idx in selected_indices]
    
    # 计算需要的行数：每行4张图片
    # 第一行原始图像，第二行预处理后图像，第三行热图+边界框
    num_rows = 3
    
    # 创建子图
    fig, axes = plt.subplots(num_rows, len(selected_images) + 1, figsize=(20, 5 * num_rows))
    
    # 添加行描述
    row_descriptions = ["原始图像和标签", "预处理后图像和标签", "热图+边界框"]
    
    for i, desc in enumerate(row_descriptions):
        axes[i, 0].text(0.5, 0.5, desc, fontsize=12, ha='center', va='center')
        axes[i, 0].axis('off')
    
    # 处理每张图像
    for j, (idx, image_name) in enumerate(zip(selected_indices, selected_images)):
        # 构建图像和标注路径
        image_path = os.path.join(JPEGIMAGES_DIR, image_name)
        annotation_path = os.path.join(ANNOTATIONS_DIR, image_name.replace('.jpg', '.xml'))
        
        # 检查标注文件是否存在
        if not os.path.exists(annotation_path):
            print(f"Warning: Annotation file not found for {image_name}")
            continue
        
        # 获取图像中的所有类别
        image_classes = get_image_classes(annotation_path)
        
        # 第一行：原始图像和标签
        visualize_original_image(image_path, annotation_path, axes[0, j+1])
        
        # 第二行：预处理后图像和标签
        visualize_preprocessed_image(image_path, annotation_path, axes[1, j+1])
        
        # 第三行：热图+边界框
        try:
            # 直接使用选定的索引从数据集中获取样本
            _, heatmap, offsets, wh = voc_dataset[idx]
            
            # 获取预处理后的边界框
            _, boxes = preprocessor.preprocess_sample(image_path, annotation_path)
            
            # 显示热图并在上面画边界框
            if heatmap.shape[0] > 0:
                visualize_heatmap_with_boxes(heatmap.numpy(), wh.numpy(), axes[2, j+1], image_classes, boxes)
        except Exception as e:
            print(f"Error visualizing heatmap: {e}")
            # 如果出错，显示空图
            axes[2, j+1].axis('off')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存结果
    output_path = os.path.join(OUTPUT_DIR, 'data_visualization_result.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    # 关闭图表，避免弹窗
    plt.close()

if __name__ == '__main__':
    main()
