import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn

# 导入配置和模型
from config import DEVICE, UPSCALE_FACTOR, NUM_CHANNELS, BATCH_SIZE, EPOCHS, LR, MODEL_NAME
from config import TRAIN_HR, TRAIN_LR, VAL_HR, VAL_LR, MODEL_PATH
from models import ESPCN, EDSR

# ===================== 1. 数据集加载器 =====================
class SRDataset(Dataset):
    """超分辨率数据集加载器
    用于加载低分辨率和高分辨率图像对
    """
    def __init__(self, hr_dir, lr_dir):
        """初始化数据集
        Args:
            hr_dir: 高分辨率图像目录
            lr_dir: 低分辨率图像目录
        """
        # 加载高分辨率图像路径
        self.hr_files = sorted([os.path.join(hr_dir, f) for f in os.listdir(hr_dir) if f.endswith(('png', 'jpg'))])
        # 加载低分辨率图像路径
        self.lr_files = sorted([os.path.join(lr_dir, f) for f in os.listdir(lr_dir) if f.endswith(('png', 'jpg'))])

    def __len__(self):
        """返回数据集大小"""
        return len(self.hr_files)

    def __getitem__(self, idx):
        """获取数据项
        Args:
            idx: 数据索引
        Returns:
            lr: 低分辨率图像张量
            hr: 高分辨率图像张量
        """
        # 读取高分辨率图像并转换为RGB
        hr = Image.open(self.hr_files[idx]).convert("RGB")
        # 读取低分辨率图像并转换为RGB
        lr = Image.open(self.lr_files[idx]).convert("RGB")
        
        # 转换为numpy数组，然后转为tensor
        # 形状从 (H, W, C) 转为 (C, H, W)，并归一化到 [0, 1]
        hr = torch.from_numpy(np.array(hr)).permute(2, 0, 1).float() / 255.0
        lr = torch.from_numpy(np.array(lr)).permute(2, 0, 1).float() / 255.0
        
        return lr, hr

# 自定义collate_fn，处理不同大小的图像
def collate_fn(batch):
    """自定义数据收集函数
    处理不同大小的图像，将它们作为列表返回
    Args:
        batch: 数据批次
    Returns:
        lrs: 低分辨率图像列表
        hrs: 高分辨率图像列表
    """
    lrs, hrs = zip(*batch)
    return list(lrs), list(hrs)

# ===================== 2. 训练函数 =====================
def train():
    """训练主函数"""
    # 加载训练数据集
    train_dataset = SRDataset(TRAIN_HR, TRAIN_LR)
    # 创建数据加载器，批量大小为BATCH_SIZE，打乱数据，使用自定义collate_fn
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    # 初始化模型
    if MODEL_NAME == 'espcn':
        model = ESPCN(UPSCALE_FACTOR, NUM_CHANNELS).to(DEVICE)
    elif MODEL_NAME == 'edsr':
        model = EDSR(UPSCALE_FACTOR, NUM_CHANNELS).to(DEVICE)
    else:
        raise ValueError(f"不支持的模型名称: {MODEL_NAME}")
    
    # 定义损失函数：均方误差（MSE）
    criterion = nn.MSELoss()
    # 定义优化器：Adam
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print(f"🚀 开始训练，设备：{DEVICE}")
    print(f"📊 训练参数：EPOCHS={EPOCHS}, BATCH_SIZE={BATCH_SIZE}, LR={LR}")
    print(f"📦 模型：{MODEL_NAME}, 超分倍数：{UPSCALE_FACTOR}x")
    print(f"💾 模型将保存至：{MODEL_PATH}")
    
    # 训练循环
    for epoch in range(EPOCHS):
        # 设置模型为训练模式
        model.train()
        # 累计损失
        total_loss = 0
        
        # 进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        # 遍历训练数据
        for lrs, hrs in pbar:
            batch_loss = 0
            # 处理每个样本
            for lr, hr in zip(lrs, hrs):
                # 将数据移至指定设备
                lr = lr.to(DEVICE)
                hr = hr.to(DEVICE)
                
                # 前向传播：生成超分辨率图像
                sr = model(lr.unsqueeze(0))  # 添加batch维度
                # 计算损失
                loss = criterion(sr, hr.unsqueeze(0))
                
                # 反向传播
                # 清空梯度
                optimizer.zero_grad()
                # 计算梯度
                loss.backward()
                # 更新参数
                optimizer.step()
                
                # 累计损失
                batch_loss += loss.item()
            
            # 计算批次平均损失
            avg_batch_loss = batch_loss / len(lrs)
            total_loss += avg_batch_loss
            # 更新进度条
            pbar.set_postfix({"loss": f"{avg_batch_loss:.4f}"})
        
        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} 平均损失：{avg_loss:.4f}")
        
        # 保存模型
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"✅ 模型已保存至：{MODEL_PATH}")

    print(f"✅ 训练完成！模型保存在 {MODEL_PATH}")

if __name__ == "__main__":
    train()
