"""
训练脚本
用于训练 Oxford-IIIT Pet 图像分割模型

模型架构：U-Net
任务类型：三分类图像分割（前景、背景、未分类）
"""

# 导入必要的库
import os  # 用于文件路径操作
import torch  # PyTorch 核心库
import torch.nn as nn  # 神经网络模块
import torch.optim as optim  # 优化器
from torch.utils.data import DataLoader  # 数据加载器
from torchvision.datasets import OxfordIIITPet  # Oxford-IIIT Pet 数据集
from tqdm import tqdm  # 进度条

# 从配置文件导入必要的配置
from config import DATA_DIR, MODEL_DIR, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, device, transform, label_transform

# 从模型文件导入 U-Net 模型
from net import UNet


def dice_loss(pred, target):
    """
    Dice 损失函数
    用于评估分割结果的质量
    
    Args:
        pred: 预测值，形状为 [batch_size, 3, H, W]
        target: 目标值，形状为 [batch_size, H, W]（0-based 索引）
    
    Returns:
        Dice 损失值
    """
    smooth = 1e-6  # 防止除以零
    
    # 获取批次大小和通道数
    batch_size, num_classes, height, width = pred.shape
    
    # 将预测值转换为概率分布
    pred = torch.softmax(pred, dim=1)  # 应用 softmax 激活
    
    # 将目标值转换为 one-hot 编码
    # 标签已经是 0-based 索引（0=前景，1=背景，2=未分类）
    target_one_hot = torch.zeros(batch_size, num_classes, height, width, device=pred.device)
    target_one_hot.scatter_(1, target.unsqueeze(1), 1.0)  # 形状变为 [batch_size, 3, H, W]
    
    # 计算每个类别的 Dice 系数
    intersection = (pred * target_one_hot).sum(dim=(2, 3))  # 计算交集
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))  # 计算并集
    dice = (2. * intersection + smooth) / (union + smooth)  # 计算 Dice 系数
    
    return 1 - dice.mean()  # 返回 Dice 损失（1 - Dice 系数）


def process_labels(targets):
    """
    处理标签，确保标签值正确
    
    Args:
        targets: 标签张量，形状为 [batch_size, H, W]，值为 1、2、3
        - 1: 前景（宠物）
        - 2: 背景
        - 3: 未分类
    
    Returns:
        处理后的标签张量，形状为 [batch_size, H, W]，值为 0、1、2（0-based 索引）
        - 0: 前景（宠物）
        - 1: 背景
        - 2: 未分类
    """
    # 将标签值从 1、2、3 转换为 0、1、2（0-based 索引）
    # 因为 CrossEntropyLoss 期望标签是 0-based 的类别索引
    return targets - 1


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
    for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc="Training")):
        # 移动数据到设备
        # images 形状: [batch_size, 3, 256, 256]
        # targets 形状: [batch_size, 256, 256]
        images = images.to(device)
        targets = targets.to(device)
        
        # 处理标签
        # 将标签值从 1、2、3 转换为 0、1、2（0-based 索引）
        # 因为 CrossEntropyLoss 期望标签是 0-based 的类别索引
        targets = process_labels(targets)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        # outputs 形状: [batch_size, 3, 256, 256]
        outputs = model(images)
        
        # 计算损失
        # CrossEntropyLoss 输入：
        # - outputs: 形状 [batch_size, 3, H, W]，模型预测的 logits
        # - targets: 形状 [batch_size, H, W]，0-based 类别索引（0、1、2）
        ce_loss = criterion(outputs, targets)
        
        # 计算 Dice 损失
        # DiceLoss 输入：
        # - outputs: 形状 [batch_size, 3, H, W]，模型预测的 logits
        # - targets: 形状 [batch_size, H, W]，0-based 类别索引（0、1、2）
        dice_loss_value = dice_loss(outputs, targets)
        
        # 组合损失：CrossEntropyLoss + DiceLoss
        # 权重可以根据任务调整，这里使用 1:1 的权重
        loss = ce_loss + dice_loss_value
        print(f"CE Loss: {ce_loss.item():.4f}, Dice Loss: {dice_loss_value.item():.4f}, Total Loss: {loss.item():.4f}")
        
        # 反向传播
        loss.backward()  # 计算梯度
        optimizer.step()  # 执行优化器步骤
        
        # 累计损失
        running_loss += loss.item() * images.size(0)
    
    # 计算平均损失
    avg_loss = running_loss / len(dataloader.dataset)
    return avg_loss


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
    """
    保存检查点，包含模型权重、优化器状态、epoch和损失
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前 epoch
        loss: 当前损失
        checkpoint_path: 检查点保存路径
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"✅ 保存检查点: {checkpoint_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    加载检查点
    
    Args:
        checkpoint_path: 检查点路径
        model: 模型
        optimizer: 优化器（可选）
    
    Returns:
        epoch: 保存的 epoch
        loss: 保存的损失
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
    print(f"📂 加载检查点: {checkpoint_path}")
    print(f"   - Epoch: {epoch}")
    print(f"   - Loss: {loss:.4f}")
    return epoch, loss


def load_dataset():
    """
    加载 Oxford-IIIT Pet 数据集
    
    Returns:
        train_dataset: 训练数据集
    """
    # 创建 OxfordIIITPet 数据集实例
    train_dataset = OxfordIIITPet(
        root=DATA_DIR,  # 数据集保存路径
        split='trainval',  # 使用训练+验证集
        target_types='segmentation',  # 使用分割标注
        transform=transform,  # 图像预处理
        target_transform=label_transform,  # 标签预处理
        download=False  # 假设已经通过 download.py 下载
    )
    return train_dataset


def create_dataloader(dataset, batch_size=BATCH_SIZE):
    """
    创建数据加载器
    
    Args:
        dataset: 数据集
        batch_size: 批次大小
    
    Returns:
        dataloader: 数据加载器
    """
    dataloader = DataLoader(
        dataset,  # 数据集
        batch_size=batch_size,  # 批次大小
        shuffle=True,  # 随机打乱数据
        num_workers=4,  # 多线程加载数据
        pin_memory=False  # M1 芯片不支持 pin_memory
    )
    return dataloader


def main(resume_from=None):
    """
    主训练函数
    
    Args:
        resume_from: 从指定检查点路径恢复训练，如果为 None 则从头开始
    """
    print("开始训练 Oxford-IIIT Pet 图像分割模型...")
    print(f"使用设备: {device}")
    
    # 加载数据集
    print("加载数据集...")
    train_dataset = load_dataset()
    
    # 创建数据加载器
    train_dataloader = create_dataloader(train_dataset)
    
    # 初始化模型
    # 输出通道数设置为 3，用于三分类（前景、背景、未分类）
    model = UNet(in_channels=3, out_channels=3).to(device)  # 移动模型到设备
    
    # 定义损失函数（交叉熵损失）
    # 用于三分类任务
    criterion = nn.CrossEntropyLoss()
    
    # 定义优化器（AdamW 优化器）
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # 初始化训练参数
    start_epoch = 0
    best_loss = float('inf')
    early_stop_counter = 0
    patience = 5  # 早停耐心值
    
    # 如果指定了 resume_from，则加载检查点
    if resume_from is not None:
        if os.path.exists(resume_from):
            start_epoch, best_loss = load_checkpoint(resume_from, model, optimizer)
            start_epoch += 1  # 从下一个 epoch 开始
        else:
            print(f"⚠️ 检查点文件不存在: {resume_from}，将从头开始训练")
    
    # 训练过程
    print("开始训练...")
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 30)
        
        # 训练一个 epoch
        train_loss = train_one_epoch(model, train_dataloader, criterion, optimizer, device)
        
        print(f"Train Loss: {train_loss:.4f}")
        
        # 保存当前 epoch 的检查点
        epoch_checkpoint_path = os.path.join(MODEL_DIR, f'epoch{epoch}_unet_pet.pth')
        save_checkpoint(model, optimizer, epoch, train_loss, epoch_checkpoint_path)
        
        # 保存最佳模型
        if train_loss < best_loss:
            best_loss = train_loss
            best_model_path = os.path.join(MODEL_DIR, 'unet_pet_best.pth')
            save_checkpoint(model, optimizer, epoch, best_loss, best_model_path)
            print(f"   这是最佳模型！")
            early_stop_counter = 0  # 重置早停计数器
        else:
            early_stop_counter += 1
            print(f"   早停计数器: {early_stop_counter}/{patience}")
            # 检查是否触发早停
            if early_stop_counter >= patience:
                print("\n⚠️ 早停触发，停止训练")
                break
    
    print("\n🎉 训练完成！")
    print(f"最佳损失: {best_loss:.4f}")


if __name__ == '__main__':
    import argparse
    
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='训练 Oxford-IIIT Pet 图像分割模型')
    parser.add_argument('--resume', type=str, default=None, help='从指定检查点路径恢复训练')
    
    # 解析参数
    args = parser.parse_args()
    
    # 调用主函数
    main(resume_from=args.resume)
