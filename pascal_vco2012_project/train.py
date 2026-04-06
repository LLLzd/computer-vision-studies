"""
训练脚本
用于训练 Pascal VOC 2012 图像分割模型

模型架构：完全体 U-Net
任务类型：21类图像分割（包括背景）
"""

# 导入必要的库
import os  # 用于文件路径操作
import torch  # PyTorch 核心库
import torch.nn as nn  # 神经网络模块
import torch.optim as optim  # 优化器
from torch.utils.data import DataLoader  # 数据加载器
from tqdm import tqdm  # 进度条

# 从配置文件导入必要的配置
from config import DATA_DIR, MODEL_DIR, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, device, transform, label_transform, NUM_CLASSES

# 从模型文件导入 U-Net 模型
from net import UNet
from visual_dataset import PascalVOC2012Dataset


def dice_loss(pred, target, ignore_index=0):
    """
    Dice 损失函数
    用于评估分割结果的质量
    
    Args:
        pred: 预测值，形状为 [batch_size, 21, H, W]
        target: 目标值，形状为 [batch_size, H, W]（0-based 索引）
        ignore_index: 忽略的类别索引（默认0，即背景）
    
    Returns:
        Dice 损失值
    """
    smooth = 1e-6  # 防止除以零
    
    # 获取批次大小和通道数
    batch_size, num_classes, height, width = pred.shape
    
    # 将预测值转换为概率分布
    pred = torch.softmax(pred, dim=1)  # 应用 softmax 激活
    
    # 将目标值转换为 one-hot 编码
    # 标签已经是 0-based 索引（0-20）
    target_one_hot = torch.zeros(batch_size, num_classes, height, width, device=pred.device)
    target_one_hot.scatter_(1, target.unsqueeze(1), 1.0)  # 形状变为 [batch_size, 21, H, W]
    
    # 创建掩码，忽略背景类
    # 背景类（0）通常占大部分像素，会导致 Dice Loss 偏向背景
    mask = (target != ignore_index).float().unsqueeze(1)  # [batch_size, 1, H, W]
    
    # 计算每个类别的 Dice 系数（忽略背景）
    intersection = (pred * target_one_hot * mask).sum(dim=(2, 3))  # 计算交集
    union = (pred * mask).sum(dim=(2, 3)) + (target_one_hot * mask).sum(dim=(2, 3))  # 计算并集
    dice = (2. * intersection + smooth) / (union + smooth)  # 计算 Dice 系数
    
    # 只计算非背景类别的平均 Dice
    # 排除背景类（索引0）
    dice_non_bg = dice[:, 1:]  # [batch_size, 20]
    return 1 - dice_non_bg.mean()  # 返回 Dice 损失（1 - Dice 系数）


def focal_loss(pred, target, alpha=0.25, gamma=2.0, ignore_index=0):
    """
    Focal Loss 函数
    用于解决类别不平衡和难易样本不平衡问题
    
    Args:
        pred: 预测值，形状为 [batch_size, 21, H, W]
        target: 目标值，形状为 [batch_size, H, W]（0-based 索引）
        alpha: 类别平衡因子
        gamma: 难样本挖掘因子
        ignore_index: 忽略的类别索引（默认0，即背景）
    
    Returns:
        Focal Loss 值
    """
    # 获取批次大小和通道数
    batch_size, num_classes, height, width = pred.shape
    
    # 计算交叉熵损失
    ce_loss = nn.CrossEntropyLoss(reduction='none')(pred, target)
    
    # 计算概率
    pred_softmax = torch.softmax(pred, dim=1)
    # 取出目标类别的概率
    target_one_hot = torch.zeros(batch_size, num_classes, height, width, device=pred.device)
    target_one_hot.scatter_(1, target.unsqueeze(1), 1.0)
    pt = (pred_softmax * target_one_hot).sum(dim=1)
    
    # 计算 Focal Loss
    focal = alpha * (1 - pt) ** gamma * ce_loss
    
    # 忽略背景类
    mask = (target != ignore_index).float()
    focal = focal * mask
    
    return focal.mean()


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    """
    训练一个 epoch
    
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        epoch: 当前 epoch 编号
        total_epochs: 总训练轮数
    
    Returns:
        平均损失
    """
    model.train()  # 设置模型为训练模式
    running_loss = 0.0  # 累计损失
    running_ce_loss = 0.0  # 累计 CE Loss
    running_dice_loss = 0.0  # 累计 Dice Loss
    running_focal_loss = 0.0  # 累计 Focal Loss
    
    # 动态权重策略
    # 前期（前 30% 轮数）：CE Loss 权重高，帮助快速收敛
    # 中期（30%-60% 轮数）：引入 Focal Loss，解决难易样本不平衡
    # 后期（后 40% 轮数）：Dice Loss 权重增加，优化分割质量
    progress = epoch / total_epochs
    if progress < 0.3:
        # 前期：CE Loss 权重 1.0，Dice Loss 权重 0.1，Focal Loss 权重 0.5
        ce_weight = 1.0
        dice_weight = 0.1
        focal_weight = 0.5
    elif progress < 0.6:
        # 中期：逐渐平衡
        ce_weight = 0.8
        dice_weight = 0.3
        focal_weight = 1.0
    else:
        # 后期：Dice Loss 权重增加
        ce_weight = 0.5
        dice_weight = 1.0
        focal_weight = 0.5
    
    print(f"Loss Weights - CE: {ce_weight:.1f}, Dice: {dice_weight:.1f}, Focal: {focal_weight:.1f}")
    
    # 遍历数据加载器
    for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc="Training")):
        # 移动数据到设备
        # images 形状: [batch_size, 3, 160, 160]
        # targets 形状: [batch_size, 160, 160]
        images = images.to(device)
        targets = targets.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        # outputs 形状: [batch_size, 21, 160, 160]
        outputs = model(images)
        
        # 计算损失
        # CrossEntropyLoss 输入：
        # - outputs: 形状 [batch_size, 21, H, W]，模型预测的 logits
        # - targets: 形状 [batch_size, H, W]，0-based 类别索引（0-20）
        ce_loss = criterion(outputs, targets)
        
        # 计算 Dice 损失
        # DiceLoss 输入：
        # - outputs: 形状 [batch_size, 21, H, W]，模型预测的 logits
        # - targets: 形状 [batch_size, H, W]，0-based 类别索引（0-20）
        dice_loss_value = dice_loss(outputs, targets)
        
        # 计算 Focal Loss
        # FocalLoss 输入：
        # - outputs: 形状 [batch_size, 21, H, W]，模型预测的 logits
        # - targets: 形状 [batch_size, H, W]，0-based 类别索引（0-20）
        focal_loss_value = focal_loss(outputs, targets)
        
        # 组合损失：动态权重
        # CE Loss 关注像素级分类准确性
        # Dice Loss 关注区域重叠度（忽略背景）
        # Focal Loss 解决类别不平衡和难易样本不平衡
        loss = ce_weight * ce_loss + dice_weight * dice_loss_value + focal_weight * focal_loss_value
        
        # 每 50 个批次打印一次损失
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}: CE Loss: {ce_loss.item():.4f}, Dice Loss: {dice_loss_value.item():.4f}, Focal Loss: {focal_loss_value.item():.4f}, Total Loss: {loss.item():.4f}")
        
        # 反向传播
        loss.backward()  # 计算梯度
        optimizer.step()  # 执行优化器步骤
        
        # 累计损失
        running_loss += loss.item() * images.size(0)
        running_ce_loss += ce_loss.item() * images.size(0)
        running_dice_loss += dice_loss_value.item() * images.size(0)
        running_focal_loss += focal_loss_value.item() * images.size(0)
    
    # 计算平均损失
    avg_loss = running_loss / len(dataloader.dataset)
    avg_ce_loss = running_ce_loss / len(dataloader.dataset)
    avg_dice_loss = running_dice_loss / len(dataloader.dataset)
    avg_focal_loss = running_focal_loss / len(dataloader.dataset)
    
    print(f"Epoch Avg - CE Loss: {avg_ce_loss:.4f}, Dice Loss: {avg_dice_loss:.4f}, Focal Loss: {avg_focal_loss:.4f}, Total Loss: {avg_loss:.4f}")
    
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
    加载 Pascal VOC 2012 数据集
    
    Returns:
        train_dataset: 训练数据集
    """
    # 创建 PascalVOC2012Dataset 数据集实例
    train_dataset = PascalVOC2012Dataset(
        root=DATA_DIR,  # 数据集保存路径
        split='trainval',  # 使用训练+验证集
        transform=transform,  # 图像预处理
        target_transform=label_transform,  # 标签预处理
        only_with_annotation=True  # 只包含有分割标注的图像
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
    print("开始训练 Pascal VOC 2012 图像分割模型...")
    print(f"使用设备: {device}")
    print(f"类别数: {NUM_CLASSES}")
    
    # 加载数据集
    print("加载数据集...")
    train_dataset = load_dataset()
    print(f"数据集大小: {len(train_dataset)}")
    
    # 创建数据加载器
    train_dataloader = create_dataloader(train_dataset)
    
    # 初始化模型
    # 输出通道数设置为 21，用于 21 类分割
    model = UNet(in_channels=3, out_channels=NUM_CLASSES).to(device)  # 移动模型到设备
    
    # 定义损失函数（交叉熵损失）
    # 用于 21 类分类任务
    # 类别权重：背景类（0）权重较低，前景类权重较高
    # 这有助于解决类别不平衡问题（背景像素远多于前景）
    class_weights = torch.ones(NUM_CLASSES, device=device)
    class_weights[0] = 0.5  # 背景类权重降低
    class_weights[1:] = 2.0  # 前景类权重增加
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
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
            print(f"将从 epoch {start_epoch} 继续训练...")
        else:
            print(f"⚠️ 检查点文件不存在: {resume_from}，将从头开始训练")
    
    # 训练过程
    print("开始训练...")
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        # epoch 已经是从 start_epoch 开始的（例如 11），所以直接显示 epoch
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print("-" * 30)
        
        # 训练一个 epoch
        train_loss = train_one_epoch(model, train_dataloader, criterion, optimizer, device, epoch, NUM_EPOCHS)
        
        print(f"Train Loss: {train_loss:.4f}")
        
        # 保存当前 epoch 的检查点
        epoch_checkpoint_path = os.path.join(MODEL_DIR, f'epoch{epoch}_unet_voc2012.pth')
        save_checkpoint(model, optimizer, epoch, train_loss, epoch_checkpoint_path)
        
        # 保存最佳模型
        if train_loss < best_loss:
            best_loss = train_loss
            best_model_path = os.path.join(MODEL_DIR, 'unet_voc2012_best.pth')
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
    parser = argparse.ArgumentParser(description='训练 Pascal VOC 2012 图像分割模型')
    parser.add_argument('--resume', type=str, default=None, 
                     help='从指定检查点路径恢复训练。'
                          '示例: python train.py --resume models/epoch10_unet_voc2012.pth')
    
    # 解析参数
    args = parser.parse_args()
    
    # 调用主函数
    main(resume_from=args.resume)
