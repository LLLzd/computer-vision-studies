"""
测试脚本
用于测试 Oxford-IIIT Pet 图像分割模型
"""

# 导入必要的库
import os  # 用于文件路径操作
import torch  # PyTorch 核心库
import torch.nn as nn  # 神经网络模块
from torch.utils.data import DataLoader  # 数据加载器
from torchvision.datasets import OxfordIIITPet  # Oxford-IIIT Pet 数据集
from tqdm import tqdm  # 进度条

# 从配置文件导入必要的配置
from config import DATA_DIR, MODEL_DIR, BATCH_SIZE, device, transform, label_transform

# 从模型文件导入 U-Net 模型
from net import UNet


def calculate_iou(pred, target):
    """
    计算 IoU（Intersection over Union）
    用于评估分割结果的质量
    Args:
        pred: 预测值，形状为 [batch_size, 3, H, W]
        target: 目标值，形状为 [batch_size, H, W]（三分类标签）
    Returns:
        IoU 值
    """
    # 对三分类输出应用 softmax 并获取预测类别
    pred = torch.softmax(pred, dim=1)  # 应用 softmax
    pred_class = torch.argmax(pred, dim=1) + 1  # 获取预测类别（加1是因为类别从1开始）
    
    # 计算每个类别的 IoU
    iou_sum = 0.0
    class_count = 0
    
    # 计算前景（类别1）的 IoU
    pred_foreground = (pred_class == 1)
    target_foreground = (target == 1)
    intersection = (pred_foreground & target_foreground).sum(dim=(1, 2)).float()
    union = (pred_foreground | target_foreground).sum(dim=(1, 2)).float()
    iou_foreground = (intersection + 1e-6) / (union + 1e-6)
    iou_sum += iou_foreground.mean().item()
    class_count += 1
    
    # 计算背景（类别2）的 IoU
    pred_background = (pred_class == 2)
    target_background = (target == 2)
    intersection = (pred_background & target_background).sum(dim=(1, 2)).float()
    union = (pred_background | target_background).sum(dim=(1, 2)).float()
    iou_background = (intersection + 1e-6) / (union + 1e-6)
    iou_sum += iou_background.mean().item()
    class_count += 1
    
    # 计算未分类（类别3）的 IoU
    pred_unknown = (pred_class == 3)
    target_unknown = (target == 3)
    intersection = (pred_unknown & target_unknown).sum(dim=(1, 2)).float()
    union = (pred_unknown | target_unknown).sum(dim=(1, 2)).float()
    iou_unknown = (intersection + 1e-6) / (union + 1e-6)
    iou_sum += iou_unknown.mean().item()
    class_count += 1
    
    # 返回平均 IoU
    return iou_sum / class_count


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
    print("开始测试 Oxford-IIIT Pet 图像分割模型...")
    print(f"使用设备: {device}")
    
    # 加载数据集
    print("加载数据集...")
    test_dataset = OxfordIIITPet(
        root=DATA_DIR,  # 数据集保存路径
        split='test',  # 使用测试集
        target_types='segmentation',  # 使用分割标注
        transform=transform,  # 图像预处理
        target_transform=label_transform,  # 标签预处理
        download=False  # 假设已经通过 download.py 下载
    )
    
    # 创建数据加载器
    test_dataloader = DataLoader(
        test_dataset,  # 数据集
        batch_size=BATCH_SIZE,  # 批次大小
        shuffle=False,  # 不打乱数据
        num_workers=0  # 单线程加载数据，避免 pickle 序列化问题
    )
    
    # 初始化模型
    # 输入通道数为 3（RGB 图像），输出通道数为 3（三分类分割）
    model = UNet(in_channels=3, out_channels=3).to(device)  # 移动模型到设备
    
    # 加载预训练模型
    model_path = os.path.join(MODEL_DIR, 'unet_pet_best.pth')
    if not os.path.exists(model_path):
        print("❌ 模型文件不存在，请先训练模型")
        return
    
    # 加载检查点文件
    checkpoint = torch.load(model_path, map_location=device)
    
    # 检查是否是完整的检查点（包含 model_state_dict）
    if 'model_state_dict' in checkpoint:
        # 从完整的检查点中加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'N/A')
        loss = checkpoint.get('loss', 'N/A')
        print(f"✅ 加载检查点: {model_path}")
        print(f"   - Epoch: {epoch}")
        print(f"   - Loss: {loss:.4f}")
    else:
        # 直接加载模型权重（兼容旧格式）
        model.load_state_dict(checkpoint)
        print(f"✅ 加载模型: {model_path}")
    
    # 测试模型
    print("开始测试...")
    avg_iou = test(model, test_dataloader, device)
    
    print(f"\n🎉 测试完成！")
    print(f"平均 IoU: {avg_iou:.4f}")


if __name__ == '__main__':
    # 调用主函数
    main()
