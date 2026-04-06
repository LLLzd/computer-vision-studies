"""
训练脚本
用于训练 MNIST 分割模型
"""

# 导入必要的库
import os  # 用于文件路径操作
import torch  # PyTorch 核心库
import torch.nn as nn  # 神经网络模块
import torch.optim as optim  # 优化器
from torch.utils.data import DataLoader  # 数据加载器
from torchvision.datasets import MNIST  # MNIST 数据集
from tqdm import tqdm  # 进度条

# 从配置文件导入必要的配置
from config import DATA_DIR, MODEL_DIR, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, device, transform

# 从模型文件导入 U-Net 模型
from net import UNet


import numpy as np
from PIL import Image

class MNISTSegmentationDataset(torch.utils.data.Dataset):
    """
    MNIST 分割数据集
    将 MNIST 图像转换为分割任务
    """
    
    def __init__(self, root, train=True, transform=None):
        """
        初始化数据集
        Args:
            root: 数据根目录
            train: 是否使用训练集
            transform: 图像转换
        """
        self.root = root
        self.train = train
        self.transform = transform
        
        # 加载 MNIST 数据
        self.images, self.labels = self._load_data()
    
    def _load_data(self):
        """
        加载 MNIST 数据
        Returns:
            (images, labels): 图像和标签
        """
        import struct
        
        # 确定文件路径
        if self.train:
            images_path = os.path.join(self.root, 'MNIST', 'raw', 'train-images-idx3-ubyte')
            labels_path = os.path.join(self.root, 'MNIST', 'raw', 'train-labels-idx1-ubyte')
        else:
            images_path = os.path.join(self.root, 'MNIST', 'raw', 't10k-images-idx3-ubyte')
            labels_path = os.path.join(self.root, 'MNIST', 'raw', 't10k-labels-idx1-ubyte')
        
        # 读取图像文件
        with open(images_path, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
        
        # 读取标签文件
        with open(labels_path, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            labels = np.fromfile(f, dtype=np.uint8)
        
        return images, labels
    
    def __len__(self):
        """
        获取数据集长度
        Returns:
            数据集长度
        """
        return len(self.images)
    
    def __getitem__(self, index):
        """
        获取数据集中的一个样本
        Args:
            index: 样本索引
        Returns:
            (image, target): 图像和标签
        """
        # 获取图像数据
        image = self.images[index]
        
        # 转换为 PIL 图像
        image = Image.fromarray(image, mode='L')
        
        # 应用转换
        if self.transform:
            image = self.transform(image)
        
        # 将图像作为标签（分割任务）
        # 二值化：大于 0.5 的像素为前景，否则为背景
        target = (image > 0.5).float()
        
        return image, target


def dice_loss(pred, target):
    """
    Dice 损失函数
    用于评估分割结果的质量
    Args:
        pred: 预测值，形状为 [batch_size, 1, H, W]
        target: 目标值，形状为 [batch_size, 1, H, W]
    Returns:
        Dice 损失值
    """
    smooth = 1e-6  # 防止除以零
    pred = torch.sigmoid(pred)  # 应用 sigmoid 激活
    intersection = (pred * target).sum(dim=(2, 3))  # 计算交集
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))  # 计算并集
    dice = (2. * intersection + smooth) / (union + smooth)  # 计算 Dice 系数
    return 1 - dice.mean()  # 返回 Dice 损失（1 - Dice 系数）


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    训练一个 epoch
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
    Returns:
        平均损失
    """
    model.train()  # 设置模型为训练模式
    running_loss = 0.0  # 累计损失
    
    # 遍历数据加载器
    for images, targets in tqdm(dataloader, desc="Training"):
        # 移动数据到设备
        images = images.to(device)
        targets = targets.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(images)
        
        # 计算损失
        loss = criterion(outputs, targets)
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        # 累计损失
        running_loss += loss.item() * images.size(0)
    
    # 计算平均损失
    avg_loss = running_loss / len(dataloader.dataset)
    return avg_loss


def main():
    """
    主训练函数
    """
    print("开始训练 MNIST 分割模型...")
    print(f"使用设备: {device}")
    
    # 加载数据集
    print("加载数据集...")
    # 直接使用 mnist_project 中的数据路径
    mnist_data_path = '/Users/rik/Downloads/workspace/study/mnist_project/data'
    print(f"使用的数据路径: {mnist_data_path}")
    print(f"路径是否存在: {os.path.exists(mnist_data_path)}")
    print(f"MNIST 目录是否存在: {os.path.exists(os.path.join(mnist_data_path, 'MNIST'))}")
    print(f"MNIST/raw 目录是否存在: {os.path.exists(os.path.join(mnist_data_path, 'MNIST', 'raw'))}")
    print(f"训练图像文件是否存在: {os.path.exists(os.path.join(mnist_data_path, 'MNIST', 'raw', 'train-images-idx3-ubyte'))}")
    
    train_dataset = MNISTSegmentationDataset(
        root=mnist_data_path,  # 数据集保存路径
        train=True,  # 使用训练集
        transform=transform  # 图像预处理
    )
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset,  # 数据集
        batch_size=BATCH_SIZE,  # 批次大小
        shuffle=True,  # 随机打乱数据
        num_workers=4  # 多线程加载数据
    )
    
    # 初始化模型
    model = UNet(in_channels=1, out_channels=1).to(device)  # 移动模型到设备
    
    # 定义损失函数（Dice 损失）
    criterion = dice_loss
    
    # 定义优化器（Adam 优化器）
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 训练过程
    print("开始训练...")
    best_loss = float('inf')  # 最佳损失值
    start_epoch = 0  # 开始的 epoch
    
    # 检查是否存在已保存的模型文件
    import glob
    model_files = glob.glob(os.path.join(MODEL_DIR, 'epoch*_unet_mnist.pth'))
    if model_files:
        # 按 epoch 号排序
        model_files.sort(key=lambda x: int(x.split('epoch')[1].split('_')[0]))
        latest_model = model_files[-1]
        # 加载模型权重和优化器状态
        checkpoint = torch.load(latest_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # 获取开始的 epoch
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
        print(f"✅ 加载最新模型: {latest_model}")
        print(f"📌 从 epoch {start_epoch} 开始训练")
        print(f"📌 上次训练损失: {checkpoint['loss']:.4f}")
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 30)
        
        # 训练一个 epoch
        train_loss = train_one_epoch(model, train_dataloader, criterion, optimizer, device)
        
        print(f"Train Loss: {train_loss:.4f}")
        
        # 保存当前 epoch 的模型（包含优化器状态）
        epoch_model_path = os.path.join(MODEL_DIR, f'epoch{epoch}_unet_mnist.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss
        }, epoch_model_path)
        print(f"✅ 保存 epoch {epoch} 模型: {epoch_model_path}")
        
        # 保存最佳模型
        if train_loss < best_loss:
            best_loss = train_loss
            best_model_path = os.path.join(MODEL_DIR, 'unet_mnist.pth')
            torch.save(model.state_dict(), best_model_path)  # 保存模型权重
            print(f"✅ 保存最佳模型: {best_model_path}")
    
    print("\n🎉 训练完成！")
    print(f"最佳损失: {best_loss:.4f}")


if __name__ == '__main__':
    # 调用主函数
    main()
