"""数据集模块

实现VOC2012数据集的加载、预处理和目标生成功能。
统一管理数据划分，90%训练，10%测试。
"""

import os
import numpy as np
from torch.utils.data import Dataset
from preprocess import Preprocessor
from config import JPEGIMAGES_DIR, ANNOTATIONS_DIR, IMAGE_SIZE, HEATMAP_SIZE, VOC_CLASSES, CLS_TO_IDX

class VOCDataset(Dataset):
    """VOC2012数据集类"""
    
    # 类变量，存储数据划分
    _train_images = None
    _val_images = None
    
    @classmethod
    def _split_dataset(cls):
        """划分数据集（90%训练，10%测试）
        
        只在第一次调用时划分，后续调用使用缓存的结果。
        """
        if cls._train_images is not None and cls._val_images is not None:
            return
        
        # 获取所有图像路径
        all_images = sorted([f for f in os.listdir(JPEGIMAGES_DIR) if f.endswith('.jpg')])
        
        # 划分训练集和验证集（90%训练，10%验证）
        split_idx = int(len(all_images) * 0.9)
        cls._train_images = all_images[:split_idx]
        cls._val_images = all_images[split_idx:]
        
        print(f"📊 Dataset split:")
        print(f"   Total images: {len(all_images)}")
        print(f"   Train images: {len(cls._train_images)} ({len(cls._train_images)/len(all_images)*100:.1f}%)")
        print(f"   Val images: {len(cls._val_images)} ({len(cls._val_images)/len(all_images)*100:.1f}%)")
    
    def __init__(self, split='train'):
        """初始化数据集
        
        Args:
            split: 数据集分割，'train'或'val'
        """
        self.split = split
        self.preprocessor = Preprocessor(target_size=IMAGE_SIZE)
        
        # 划分数据集
        VOCDataset._split_dataset()
        
        # 获取对应的图像列表
        if split == 'train':
            self.image_paths = VOCDataset._train_images
        else:
            self.image_paths = VOCDataset._val_images
    
    def __len__(self):
        """返回数据集长度
        
        Returns:
            数据集长度
        """
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            image: 预处理后的图像张量
            heatmap: 目标heatmap
            offsets: 偏移量
            wh: 宽高
        """
        # 获取图像名称
        image_name = self.image_paths[idx]
        
        # 构建图像和标注路径
        image_path = os.path.join(JPEGIMAGES_DIR, image_name)
        annotation_path = os.path.join(ANNOTATIONS_DIR, image_name.replace('.jpg', '.xml'))
        
        # 预处理图像和标注
        image, boxes = self.preprocessor.preprocess_sample(image_path, annotation_path)
        
        # 生成heatmap、偏移量和宽高
        heatmap, offsets, wh = self._generate_target(boxes)
        
        # 将图像转换为张量
        from torchvision import transforms
        image = transforms.ToTensor()(image)
        
        # 将numpy数组转换为张量
        import torch
        heatmap = torch.tensor(heatmap, dtype=torch.float32)
        offsets = torch.tensor(offsets, dtype=torch.float32)
        wh = torch.tensor(wh, dtype=torch.float32)
        
        return image, heatmap, offsets, wh
    
    def _generate_target(self, boxes):
        """生成heatmap、偏移量和宽高目标
        
        Args:
            boxes: 边界框列表
            
        Returns:
            heatmap: 目标heatmap
            offsets: 偏移量
            wh: 宽高（在原图尺度上的宽高）
        """
        # 初始化heatmap、偏移量和宽高
        heatmap = np.zeros((len(VOC_CLASSES), HEATMAP_SIZE[0], HEATMAP_SIZE[1]), dtype=np.float32)
        offsets = np.zeros((2, HEATMAP_SIZE[0], HEATMAP_SIZE[1]), dtype=np.float32)  # (x, y)偏移
        wh = np.zeros((2, HEATMAP_SIZE[0], HEATMAP_SIZE[1]), dtype=np.float32)  # (width, height)
        
        # 计算缩放因子
        scale_x = HEATMAP_SIZE[1] / IMAGE_SIZE[0]
        scale_y = HEATMAP_SIZE[0] / IMAGE_SIZE[1]
        
        # 遍历每个边界框
        for cls, xmin, ymin, xmax, ymax in boxes:
            # 计算目标中心
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            
            # 计算宽高（在原图尺度上）
            width = xmax - xmin
            height = ymax - ymin
            
            # 转换到heatmap坐标
            hm_x = int(center_x * scale_x)
            hm_y = int(center_y * scale_y)
            
            # 确保坐标在有效范围内
            hm_x = max(0, min(HEATMAP_SIZE[1]-1, hm_x))
            hm_y = max(0, min(HEATMAP_SIZE[0]-1, hm_y))
            
            # 获取类别索引
            cls_idx = CLS_TO_IDX.get(cls, 0)
            
            # 生成heatmap (使用高斯分布)
            self._draw_heatmap(heatmap[cls_idx], hm_x, hm_y, sigma=5)
            
            # 计算偏移量
            offsets[0, hm_y, hm_x] = (center_x * scale_x - hm_x)  # x偏移
            offsets[1, hm_y, hm_x] = (center_y * scale_y - hm_y)  # y偏移
            
            # 计算宽高（保持原图尺度）
            wh[0, hm_y, hm_x] = width   # 宽度
            wh[1, hm_y, hm_x] = height  # 高度
        
        return heatmap, offsets, wh
    
    def _draw_heatmap(self, heatmap, center_x, center_y, sigma=5):
        """绘制高斯heatmap
        
        Args:
            heatmap: heatmap数组
            center_x: 中心x坐标
            center_y: 中心y坐标
            sigma: 高斯分布的标准差
        """
        # 首先设置中心点为1.0（确保精确匹配）
        heatmap[center_y, center_x] = 1.0
        
        # 然后绘制高斯分布（跳过中心点）
        height, width = heatmap.shape
        for y in range(max(0, center_y - 3*sigma), min(height, center_y + 3*sigma + 1)):
            for x in range(max(0, center_x - 3*sigma), min(width, center_x + 3*sigma + 1)):
                if y == center_y and x == center_x:
                    continue  # 跳过中心点
                # 计算距离
                distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                # 计算高斯值
                heatmap[y, x] = max(heatmap[y, x], np.exp(-distance ** 2 / (2 * sigma ** 2)))

def get_test_image_paths(num_images=4):
    """获取测试集图像路径（用于推理）
    
    Args:
        num_images: 需要获取的图像数量
        
    Returns:
        图像路径列表
    """
    # 确保数据集已划分
    VOCDataset._split_dataset()
    
    # 从验证集中随机选择
    import random
    if len(VOCDataset._val_images) < num_images:
        num_images = len(VOCDataset._val_images)
    
    return random.sample(VOCDataset._val_images, num_images)
