"""
数据集可视化脚本
用于可视化 Oxford-IIIT Pet 数据集中的样本和分割结果

数据格式说明：
- 图像：RGB 彩色图像，形状为 (3, 256, 256)（通道, 高度, 宽度）
- 标签：分割标注，形状为 (256, 256)，值为 1=前景，2=背景，3=未分类

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
from torchvision.datasets import OxfordIIITPet  # Oxford-IIIT Pet 数据集

# 从配置文件导入必要的配置
from config import DATA_DIR, OUTPUT_DIR, transform, label_transform


def get_dataset_samples(dataset, num_samples=4):
    """
    从数据集中获取指定数量的样本
    
    Args:
        dataset: OxfordIIITPet 数据集对象，包含图像和分割标注
        num_samples: 要获取的样本数量
    
    Returns:
        images: 图像张量，形状为 (num_samples, 3, 256, 256)
        targets: 标签张量，形状为 (num_samples, 256, 256)
    """
    # 创建数据加载器，用于批量获取样本
    dataloader = DataLoader(
        dataset,  # 输入数据集
        batch_size=num_samples,  # 批次大小设置为要可视化的样本数量
        shuffle=True  # 随机打乱数据，每次运行都会显示不同的样本
    )
    
    # 获取第一个批次的样本
    images, targets = next(iter(dataloader))  # 获取第一个批次
    
    return images, targets


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
        label_tensor: 标签张量，形状为 (1, 256, 256)
    
    Returns:
        处理后的标签数组，形状为 (256, 256)，值为 0-255
    """
    # 去除通道维度并转换为 numpy 数组
    label = label_tensor.squeeze().numpy()
    
    # 转换标签值：前景为255（白色），背景为0（黑色），未分类为127（灰色）
    label_processed = np.where(label == 1, 255, np.where(label == 2, 0, 127))
    
    return label_processed


def visualize_and_save(images, targets, save_path):
    """
    可视化样本并保存结果
    
    Args:
        images: 图像张量，形状为 (num_samples, 3, 256, 256)
        targets: 标签张量，形状为 (num_samples, 256, 256)
        save_path: 保存路径
    """
    # 创建画布
    num_samples = len(images)  # 获取样本数量
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5*num_samples))  # 每行显示图像和标签
    
    # 遍历每个样本
    for i in range(num_samples):
        # 处理图像
        image = denormalize_image(images[i])  # 反归一化
        image = image.permute(1, 2, 0).numpy()  # 转换为 (H, W, C) 格式
        image = (image * 255).astype(np.uint8)  # 转换为 0-255 范围
        
        # 处理标签
        target = process_label(targets[i])  # 处理标签
        
        # 显示图像
        axes[i, 0].imshow(image)  # 显示原始图像
        axes[i, 0].set_title(f'Sample {i+1}')  # 设置标题
        axes[i, 0].axis('off')  # 关闭坐标轴
        
        # 显示标签
        axes[i, 1].imshow(target, cmap='gray')  # 显示分割标注，使用灰度 colormap
        axes[i, 1].set_title(f'Segmentation Mask')  # 设置标题
        axes[i, 1].axis('off')  # 关闭坐标轴
    
    # 调整布局，避免子图重叠
    plt.tight_layout()
    
    # 保存结果
    plt.savefig(save_path, dpi=150, bbox_inches='tight')  # 保存为 PNG 文件
    plt.close()  # 关闭图形，释放内存
    
    print(f"可视化结果已保存到: {save_path}")  # 打印保存路径


def load_dataset():
    """
    加载 Oxford-IIIT Pet 数据集
    
    Returns:
        dataset: OxfordIIITPet 数据集实例
    """
    dataset = OxfordIIITPet(
        root=DATA_DIR,  # 数据集保存路径
        split='trainval',  # 使用训练+验证集
        target_types='segmentation',  # 使用分割标注
        transform=transform,  # 图像预处理（调整大小、归一化等）
        target_transform=label_transform,  # 标签预处理（调整大小）
        download=False  # 假设已经通过 download.py 下载
    )
    return dataset


def main():
    """
    主可视化函数
    加载数据集，获取样本，保存可视化结果
    """
    print("开始可视化 Oxford-IIIT Pet 数据集...")  # 打印开始信息
    
    # 加载数据集
    print("加载数据集...")  # 打印加载信息
    dataset = load_dataset()
    
    # 获取样本
    print("获取样本...")  # 打印获取样本信息
    images, targets = get_dataset_samples(dataset, num_samples=4)  # 获取4个样本
    
    # 保存可视化结果
    save_path = os.path.join(OUTPUT_DIR, 'dataset_visualization.png')
    visualize_and_save(images, targets, save_path)
    
    print("\n🎉 可视化完成！")  # 打印完成信息


if __name__ == '__main__':
    # 调用主函数
    main()