import os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

# 数据集路径
DATA_DIR = 'data'
JPEGIMAGES_DIR = os.path.join(DATA_DIR, 'VOC2012', 'JPEGImages')
ANNOTATIONS_DIR = os.path.join(DATA_DIR, 'VOC2012', 'Annotations')

class Preprocessor:
    def __init__(self, target_size=(512, 512), fill_color=(128, 128, 128)):
        """初始化预处理器
        
        Args:
            target_size: 目标图像尺寸 (width, height)
            fill_color: 填充颜色 (R, G, B)
        """
        self.target_size = target_size
        self.fill_color = fill_color
    
    def preprocess_image(self, image):
        """预处理单张图像
        
        Args:
            image: PIL图像对象或图像路径
            
        Returns:
            processed_image: 处理后的图像
            padding_info: 填充信息 (top, left, right, bottom)
        """
        # 如果是路径，读取图像
        if isinstance(image, str):
            image = Image.open(image)
        
        if image is None:
            raise ValueError("无法读取图像")
        
        # 获取原始尺寸
        w, h = image.size
        target_w, target_h = self.target_size
        
        # 计算填充量
        pad_top = (target_h - h) // 2
        pad_bottom = target_h - h - pad_top
        pad_left = (target_w - w) // 2
        pad_right = target_w - w - pad_left
        
        # 创建新图像并填充
        processed_image = Image.new('RGB', (target_w, target_h), self.fill_color)
        processed_image.paste(image, (pad_left, pad_top))
        
        return processed_image, (pad_top, pad_left, pad_right, pad_bottom)
    
    def parse_annotation(self, annotation_path):
        """解析标注文件
        
        Args:
            annotation_path: 标注文件路径
            
        Returns:
            boxes: 边界框列表，格式为 [(cls, xmin, ymin, xmax, ymax), ...]
        """
        # 解析标注文件
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        boxes = []
        for obj in root.findall('object'):
            cls = obj.find('name').text
            bndbox = obj.find('bndbox')
            # 处理浮点数坐标
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))
            boxes.append((cls, xmin, ymin, xmax, ymax))
        
        return boxes
    
    def update_boxes(self, boxes, padding_info):
        """更新边界框坐标
        
        Args:
            boxes: 原始边界框列表
            padding_info: 填充信息 (top, left, right, bottom)
            
        Returns:
            updated_boxes: 更新后的边界框列表
        """
        pad_top, pad_left, _, _ = padding_info
        updated_boxes = []
        
        for cls, xmin, ymin, xmax, ymax in boxes:
            # 添加填充偏移
            new_xmin = xmin + pad_left
            new_ymin = ymin + pad_top
            new_xmax = xmax + pad_left
            new_ymax = ymax + pad_top
            updated_boxes.append((cls, new_xmin, new_ymin, new_xmax, new_ymax))
        
        return updated_boxes
    
    def preprocess_sample(self, image_path, annotation_path):
        """预处理单个样本
        
        Args:
            image_path: 图像路径
            annotation_path: 标注文件路径
            
        Returns:
            processed_image: 处理后的图像
            updated_boxes: 更新后的边界框列表
        """
        # 检查文件是否存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"标注文件不存在: {annotation_path}")
        
        # 预处理图像
        processed_image, padding_info = self.preprocess_image(image_path)
        
        # 解析并更新标注
        original_boxes = self.parse_annotation(annotation_path)
        updated_boxes = self.update_boxes(original_boxes, padding_info)
        
        return processed_image, updated_boxes

def main():
    """主函数"""
    # 初始化预处理器
    preprocessor = Preprocessor(target_size=(512, 512))
    
    # 测试单个样本
    test_image_name = '2007_000027'  # 测试用的图像文件名
    image_path = os.path.join(JPEGIMAGES_DIR, f"{test_image_name}.jpg")
    annotation_path = os.path.join(ANNOTATIONS_DIR, f"{test_image_name}.xml")
    
    try:
        processed_image, updated_boxes = preprocessor.preprocess_sample(image_path, annotation_path)
        print(f"处理完成！")
        print(f"处理后图像尺寸: {processed_image.size}")
        print(f"原始边界框数量: {len(updated_boxes)}")
        print(f"更新后的边界框: {updated_boxes[:2]}...")  # 只显示前2个边界框
        
        # 显示处理前后的对比
        print("\n处理前后对比:")
        # 读取原始图像
        original_image = Image.open(image_path)
        
        print(f"原始尺寸: {original_image.size}")
        print(f"处理后尺寸: {processed_image.size}")
        
    except Exception as e:
        print(f"处理失败: {e}")

if __name__ == '__main__':
    main()