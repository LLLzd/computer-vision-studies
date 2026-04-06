"""
数据集可视化脚本
用于可视化 Pascal VOC 2012 数据集中的样本和分割结果

数据格式说明：
- 图像：RGB 彩色图像，形状为 (3, 128, 128)（通道, 高度, 宽度）
- 标签：分割标注，形状为 (128, 128)，值为类别索引

可视化说明：
- 显示原始图像和对应的分割标注
- 每行显示一个样本的图像和其分割掩码
- 保存为 PNG 格式的可视化结果
"""

# 导入必要的库
import os  # 用于文件路径操作
import torch  # PyTorch 核心库
import matplotlib.pyplot as plt  # 用于可视化
import numpy as np  # 用于数组操作
from torch.utils.data import DataLoader  # 数据加载器

# 从配置文件导入必要的配置
from config import DATA_DIR, OUTPUT_DIR, transform, label_transform, NUM_CLASSES, IMAGE_SIZE


class PascalVOC2012Dataset:
    """
    Pascal VOC 2012 数据集类
    用于加载和处理 Pascal VOC 2012 数据集
    总图像数: 11540
    有分割标注的图像数: 2273 (19.70%)
    无分割标注的图像数: 9267 (80.30%)
    """
    def __init__(self, root, split='train', transform=None, target_transform=None, only_with_annotation=True):
        """
        初始化 Pascal VOC 2012 数据集
        
        Args:
            root: 数据集根目录
            split: 数据集分割，可选 'train', 'val', 'trainval'
            transform: 图像预处理
            target_transform: 标签预处理
            only_with_annotation: 是否只包含有分割标注的图像
        """
        # 初始化类变量
        self.root = root  # 数据集根目录
        self.split = split  # 数据集分割类型
        self.transform = transform  # 图像预处理函数
        self.target_transform = target_transform  # 标签预处理函数
        self.only_with_annotation = only_with_annotation  # 是否只包含有标注的图像
        
        # 初始化图像和标签列表
        self.images = []  # 存储图像路径
        self.targets = []  # 存储分割标注路径
        
        # 读取图像列表文件
        list_file = os.path.join(root, 'VOC2012', 'ImageSets', 'Main', f'{split}.txt')
        with open(list_file, 'r') as f:
            # 遍历文件中的每一行
            for line in f:
                img_id = line.strip()  # 获取图像ID
                # 构建图像文件路径
                img_path = os.path.join(root, 'VOC2012', 'JPEGImages', f'{img_id}.jpg')
                
                # 检查图像文件是否存在
                if os.path.exists(img_path):
                    # 构建分割标注文件路径
                    target_path = os.path.join(root, 'VOC2012', 'SegmentationClass', f'{img_id}.png')
                    
                    # 检查是否只需要有标注的图像
                    if self.only_with_annotation:
                        # 如果只需要有标注的图像且标注文件存在
                        if os.path.exists(target_path):
                            self.images.append(img_path)  # 添加图像路径
                            self.targets.append(target_path)  # 添加标注路径
                    else:
                        # 否则，无论是否有标注都添加
                        self.images.append(img_path)  # 添加图像路径
                        # 如果有标注，添加标注路径，否则添加 None
                        if os.path.exists(target_path):
                            self.targets.append(target_path)
                        else:
                            self.targets.append(None)
    
    def __len__(self):
        """
        返回数据集长度
        
        Returns:
            数据集样本数量
        """
        return len(self.images)  # 返回图像列表长度
    
    def __getitem__(self, idx):
        """
        获取数据集中的一个样本
        
        Args:
            idx: 样本索引
        
        Returns:
            图像和标签的元组
        """
        from PIL import Image  # 延迟导入 PIL，节省内存
        
        # 加载图像
        img_path = self.images[idx]  # 获取图像路径
        image = Image.open(img_path).convert('RGB')  # 打开图像并转换为 RGB 格式
        
        # 加载标签
        target_path = self.targets[idx]  # 获取标注路径
        if target_path and os.path.exists(target_path):
            target = Image.open(target_path)  # 打开标注图像
        else:
            # 如果没有分割标注，创建一个空的标签
            target = Image.new('L', image.size, 0)  # 创建全黑的标签图像
        
        # 应用变换
        if self.transform:
            image = self.transform(image)  # 应用图像预处理
        if self.target_transform:
            target = self.target_transform(target)  # 应用标签预处理
        
        return image, target  # 返回处理后的图像和标签


def get_color_map():
    """
    获取类别颜色映射
    
    Returns:
        颜色映射字典，键为类别索引，值为 RGB 颜色值
    """
    # 为 21 个 VOC2012 类别定义不同的颜色
    color_map = {
        0: [0, 0, 0],        # 背景 - 黑色
        1: [128, 0, 0],      # 飞机 - 深红色
        2: [0, 128, 0],      # 自行车 - 深绿色
        3: [128, 128, 0],    # 鸟 - 橄榄色
        4: [0, 0, 128],      # 船 - 深蓝色
        5: [128, 0, 128],    # 瓶子 - 紫色
        6: [0, 128, 128],    # 公交车 - 青色
        7: [128, 128, 128],  # 汽车 - 灰色
        8: [64, 0, 0],       # 猫 - 暗红色
        9: [192, 0, 0],      # 椅子 - 亮红色
        10: [64, 128, 0],    # 牛 - 深绿棕色
        11: [192, 128, 0],   # 餐桌 - 橙色
        12: [64, 0, 128],    # 狗 - 深紫色
        13: [192, 0, 128],   # 马 - 亮紫色
        14: [64, 128, 128],  # 摩托车 - 深青色
        15: [192, 128, 128], # 人 - 浅粉色
        16: [0, 64, 0],       # 盆栽植物 - 暗绿色
        17: [128, 64, 0],     # 羊 - 棕色
        18: [0, 192, 0],      # 沙发 - 亮绿色
        19: [128, 192, 0],    # 火车 - 黄绿色
        20: [0, 64, 128]      # 电视监视器 - 暗蓝色
    }
    return color_map  # 返回颜色映射字典


def denormalize_image(tensor):
    """
    反归一化图像张量
    
    因为训练时图像被归一化到 [-1, 1] 范围，需要反归一化回 [0, 1] 范围以便可视化
    
    Args:
        tensor: 归一化后的张量，形状为 (3, H, W)
    
    Returns:
        反归一化后的张量，形状为 (3, H, W)
    """
    # ImageNet 数据集的均值和标准差
    mean = torch.tensor([0.485, 0.456, 0.406])  # ImageNet 均值
    std = torch.tensor([0.229, 0.224, 0.225])  # ImageNet 标准差
    
    # 反归一化公式：原始值 = 归一化值 * 标准差 + 均值
    return tensor * std[:, None, None] + mean[:, None, None]


def process_label(label_tensor):
    """
    处理标签张量，转换为可视化格式
    
    Args:
        label_tensor: 标签张量，形状为 (128, 128)
    
    Returns:
        处理后的标签数组，形状为 (128, 128, 3)，RGB 格式
    """
    # 去除通道维度并转换为 numpy 数组
    label = label_tensor.squeeze().numpy()
    
    # 获取颜色映射
    color_map = get_color_map()
    
    # 创建 RGB 图像
    height, width = label.shape  # 获取标签的高度和宽度
    label_rgb = np.zeros((height, width, 3), dtype=np.uint8)  # 创建 RGB 图像数组
    
    # 遍历每个像素，根据类别索引设置颜色
    for i in range(height):
        for j in range(width):
            class_idx = int(label[i, j])  # 获取当前像素的类别索引
            # 确保类别索引在有效范围内
            if class_idx in color_map:
                label_rgb[i, j] = color_map[class_idx]  # 设置对应类别的颜色
            else:
                label_rgb[i, j] = [0, 0, 0]  # 默认为黑色
    
    return label_rgb  # 返回处理后的 RGB 标签图像


def visualize_and_save(images, targets, save_path):
    """
    可视化样本并保存结果
    
    Args:
        images: 图像张量，形状为 (num_samples, 3, 128, 128)
        targets: 标签张量，形状为 (num_samples, 128, 128)
        save_path: 保存路径
    """
    # 创建画布
    num_samples = len(images)  # 获取样本数量
    # 创建子图，每行显示一个样本的图像和标签
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5*num_samples))
    
    # 遍历每个样本
    for i in range(num_samples):
        # 处理图像
        image = denormalize_image(images[i])  # 反归一化图像
        image = image.permute(1, 2, 0).numpy()  # 转换为 (H, W, C) 格式
        image = (image * 255).astype(np.uint8)  # 转换为 0-255 范围
        
        # 处理标签
        target = process_label(targets[i])  # 处理标签为 RGB 格式
        
        # 显示图像
        axes[i, 0].imshow(image)  # 显示原始图像
        axes[i, 0].set_title(f'Sample {i+1}')  # 设置标题
        axes[i, 0].axis('off')  # 关闭坐标轴
        
        # 显示标签
        axes[i, 1].imshow(target)  # 显示分割标注，使用彩色
        axes[i, 1].set_title('Segmentation Mask')  # 设置标题
        axes[i, 1].axis('off')  # 关闭坐标轴
    
    # 调整布局，避免子图重叠
    plt.tight_layout()
    
    # 保存结果
    plt.savefig(save_path, dpi=150, bbox_inches='tight')  # 保存为 PNG 文件
    plt.close()  # 关闭图形，释放内存
    
    print(f"可视化结果已保存到: {save_path}")  # 打印保存路径


def get_dataset_samples(dataset, num_samples=4):
    """
    从数据集中获取指定数量的样本
    
    Args:
        dataset: Pascal VOC 2012 数据集对象，包含图像和分割标注
        num_samples: 要获取的样本数量
    
    Returns:
        images: 图像张量，形状为 (num_samples, 3, 128, 128)
        targets: 标签张量，形状为 (num_samples, 128, 128)
    """
    # 创建数据加载器，用于批量获取样本
    dataloader = DataLoader(
        dataset,  # 输入数据集
        batch_size=num_samples,  # 批次大小设置为要可视化的样本数量
        shuffle=True  # 随机打乱数据，每次运行都会显示不同的样本
    )
    
    # 获取第一个批次的样本
    images, targets = next(iter(dataloader))  # 获取第一个批次
    
    return images, targets  # 返回图像和标签张量


def load_dataset():
    """
    加载 Pascal VOC 2012 数据集
    
    Returns:
        dataset: Pascal VOC 2012 数据集实例，只包含有分割标注的图像
    """
    dataset = PascalVOC2012Dataset(
        root=DATA_DIR,  # 数据集保存路径
        split='trainval',  # 使用训练+验证集
        transform=transform,  # 图像预处理（调整大小、归一化等）
        target_transform=label_transform,  # 标签预处理（调整大小）
        only_with_annotation=True  # 只包含有分割标注的图像
    )
    return dataset  # 返回数据集实例


def main():
    """
    主可视化函数
    加载数据集，获取样本，保存可视化结果
    """
    print("开始可视化 Pascal VOC 2012 数据集...")  # 打印开始信息
    
    # 加载数据集
    print("加载数据集...")  # 打印加载信息
    dataset = load_dataset()  # 加载只包含有标注的数据集
    print(f"数据集加载完成，包含 {len(dataset)} 张有分割标注的图像")  # 打印数据集大小
    
    # 获取样本
    print("获取样本...")  # 打印获取样本信息
    images, targets = get_dataset_samples(dataset, num_samples=4)  # 获取4个样本
    
    # 保存可视化结果
    save_path = os.path.join(OUTPUT_DIR, 'dataset_visualization.png')
    visualize_and_save(images, targets, save_path)  # 可视化并保存结果
    
    print("\n🎉 可视化完成！")  # 打印完成信息


if __name__ == '__main__':
    # 调用主函数
    main()
