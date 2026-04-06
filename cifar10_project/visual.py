"""
可视化脚本
用于可视化数据集中的样本图片
"""
# 导入 PyTorch 库
import torch
# 导入 torchvision 库，用于加载数据集和图像处理
import torchvision
# 导入 matplotlib 库，用于可视化
import matplotlib.pyplot as plt
# 导入 numpy 库，用于数组操作
import numpy as np

# 从配置文件导入必要的配置
from config import DATA_DIR, VISUAL_PATH, TRANSFORM


def get_sample_images(num_samples=4):
    """
    从训练集中获取指定数量的样本
    Args:
        num_samples: 样本数量，默认为4
    Returns:
        images: 图像张量列表
        labels: 标签列表
    """
    # 加载 CIFAR-10 训练集
    trainset = torchvision.datasets.CIFAR10(
        root=DATA_DIR,  # 数据集保存路径
        train=True,  # 加载训练集（train=False 表示加载测试集）
        download=True,  # 如果数据集不存在则下载
        transform=TRANSFORM  # 应用数据预处理转换
    )
    
    # 创建数据加载器，获取指定数量的样本
    dataloader = torch.utils.data.DataLoader(
        trainset,  # 数据集
        batch_size=num_samples,  # 批次大小（等于样本数量）
        shuffle=True  # 随机打乱数据
    )
    # 获取第一个批次的图像和标签
    images, labels = next(iter(dataloader))
    # 返回图像和标签
    return images, labels


def denormalize(tensor):
    """
    反归一化张量，将值从[-1, 1]转换到[0, 1]
    Args:
        tensor: 归一化后的张量
    Returns:
        反归一化后的张量
    """
    # 反归一化：(x + 1) / 2
    return tensor / 2 + 0.5


def visualize_samples(images, labels, classes, save_path):
    """
    可视化样本图片
    Args:
        images: 图像张量，形状为 [N, C, H, W]
        labels: 标签列表
        classes: 类别名称元组
        save_path: 保存路径
    """
    # 反归一化图像
    images = denormalize(images)
    
    # 使用 make_grid 拼接图片
    # nrow=2 表示每行显示 2 张图片，padding=2 表示图片之间的间距
    grid = torchvision.utils.make_grid(images, nrow=2, padding=2)
    
    # 转换格式用于显示
    # 将张量转换为 numpy 数组
    npimg = grid.numpy()
    # 从 [C, H, W] 转换为 [H, W, C] 格式（用于 matplotlib 显示）
    npimg = np.transpose(npimg, (1, 2, 0))
    
    # 创建图形
    plt.figure(figsize=(10, 10))
    # 显示拼接后的图片
    plt.imshow(npimg)
    # 关闭坐标轴
    plt.axis("off")
    
    # 添加类别标签
    # 生成标签文本，格式为 "1. class | 2. class | ..."
    label_text = " | ".join([f"{i+1}. {classes[labels[i]]}" for i in range(len(labels))])
    # 设置标题，显示 "CIFAR-10 Samples" 和标签文本
    plt.title(f"CIFAR-10 Samples\n{label_text}", fontsize=12)
    
    # 调整布局
    plt.tight_layout()
    # 保存图片到指定路径，dpi=150 表示分辨率，bbox_inches='tight' 表示紧凑保存
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    # 打印保存路径
    print(f"可视化图片已保存到: {save_path}")


def main():
    """主可视化函数"""
    # 打印加载数据集的提示
    print("加载数据集...")
    
    # 获取4个样本
    num_samples = 4
    images, labels = get_sample_images(num_samples)
    
    # 打印获取的样本数量
    print(f"获取了 {num_samples} 个样本")
    
    # 定义类别名称
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # 可视化并保存
    visualize_samples(images, labels, classes, VISUAL_PATH)
    
    # 打印样本信息
    print("\n样本信息：")
    # 遍历每个样本，打印样本索引和类别
    for i in range(num_samples):
        print(f"  样本 {i+1}: {classes[labels[i]]}")


if __name__ == '__main__':
    # 调用主函数
    main()
