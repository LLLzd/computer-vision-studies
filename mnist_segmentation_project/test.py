"""
测试脚本
用于测试 MNIST 分割模型
"""

# 导入必要的库
import os  # 用于文件路径操作
import torch  # PyTorch 核心库
from torch.utils.data import DataLoader  # 数据加载器
from tqdm import tqdm  # 进度条

# 从配置文件导入必要的配置
from config import DATA_DIR, MODEL_DIR, BATCH_SIZE, device, transform

# 从模型文件导入 U-Net 模型
from net import UNet

# 从训练脚本导入数据集类
from train import MNISTSegmentationDataset


def calculate_iou(pred, target, threshold=0.5):
    """
    计算 IoU（Intersection over Union）
    用于评估分割结果的质量
    Args:
        pred: 预测值，形状为 [batch_size, 1, H, W]
        target: 目标值，形状为 [batch_size, 1, H, W]
        threshold: 二值化阈值
    Returns:
        IoU 值
    """
    pred = torch.sigmoid(pred) > threshold  # 应用 sigmoid 并二值化
    target = target > 0.5  # 二值化目标
    
    intersection = (pred & target).sum(dim=(2, 3)).float()  # 计算交集
    union = (pred | target).sum(dim=(2, 3)).float()  # 计算并集
    
    iou = (intersection + 1e-6) / (union + 1e-6)  # 计算 IoU，防止除以零
    return iou.mean().item()  # 返回平均 IoU


def test(model, dataloader, device):
    """
    测试模型
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
    Returns:
        平均 IoU
    """
    model.eval()  # 设置模型为评估模式
    running_iou = 0.0  # 累计 IoU
    
    with torch.no_grad():  # 禁用梯度计算
        for images, targets in tqdm(dataloader, desc="Testing"):
            # 移动数据到设备
            images = images.to(device)
            targets = targets.to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 计算 IoU
            iou = calculate_iou(outputs, targets)
            running_iou += iou * images.size(0)
    
    # 计算平均 IoU
    avg_iou = running_iou / len(dataloader.dataset)
    return avg_iou


def main():
    """
    主测试函数
    """
    print("开始测试 MNIST 分割模型...")
    print(f"使用设备: {device}")
    
    # 加载数据集
    print("加载数据集...")
    test_dataset = MNISTSegmentationDataset(
        root=DATA_DIR,  # 数据集保存路径
        train=False,  # 使用测试集
        transform=transform  # 图像预处理
    )
    
    # 创建数据加载器
    test_dataloader = DataLoader(
        test_dataset,  # 数据集
        batch_size=BATCH_SIZE,  # 批次大小
        shuffle=False,  # 不打乱数据
        num_workers=4  # 多线程加载数据
    )
    
    # 初始化模型
    model = UNet(in_channels=1, out_channels=1).to(device)  # 移动模型到设备
    
    # 加载预训练模型
    model_path = os.path.join(MODEL_DIR, 'unet_mnist.pth')
    if not os.path.exists(model_path):
        print("❌ 模型文件不存在，请先训练模型")
        return
    
    model.load_state_dict(torch.load(model_path))  # 加载模型权重
    print(f"✅ 加载模型: {model_path}")
    
    # 测试模型
    print("开始测试...")
    avg_iou = test(model, test_dataloader, device)
    
    print(f"\n🎉 测试完成！")
    print(f"平均 IoU: {avg_iou:.4f}")


if __name__ == '__main__':
    # 调用主函数
    main()
