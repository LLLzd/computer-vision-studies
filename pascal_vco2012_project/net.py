"""
分割模型定义
使用完全体的 U-Net 架构进行图像分割
适用于 Pascal VOC 2012 数据集
"""

# 导入必要的库
import torch  # PyTorch 核心库
import torch.nn as nn  # 神经网络模块


class DoubleConv(nn.Module):
    """
    两次卷积操作：Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d -> ReLU
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """
    完全体的 U-Net 分割模型
    编码器-解码器架构，带有跳跃连接
    适用于图像分割任务
    
    结构：
    - 编码器：4层 + 1层瓶颈层
    - 解码器：4层
    - 通道数：64 → 128 → 256 → 512 → 1024 → 512 → 256 → 128 → 64
    """
    
    def __init__(self, in_channels=3, out_channels=21):
        """
        初始化模型
        
        Args:
            in_channels: 输入通道数（RGB图像为3）
            out_channels: 输出通道数（VOC2012为21类）
        """
        super(UNet, self).__init__()  # 调用父类初始化方法
        
        # 编码器（下采样）部分
        # 用于提取图像特征，每一层都进行卷积和下采样
        self.encoder1 = DoubleConv(in_channels, 64)
        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(64, 128)
        )
        self.encoder3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(128, 256)
        )
        self.encoder4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(256, 512)
        )
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(512, 1024)
        )
        
        # 解码器（上采样）部分
        # 用于恢复空间维度，每一层都进行上采样和卷积
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            DoubleConv(1024, 512)
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            DoubleConv(512, 256)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            DoubleConv(256, 128)
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            DoubleConv(128, 64)
        )
        
        # 输出层
        # 将 64 通道转换为 21 通道（VOC2012 分割）
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, 3, H, W]
        
        Returns:
            输出张量，形状为 [batch_size, 21, H, W]
        """
        # 编码器前向传播，保存中间特征图
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        bottleneck = self.bottleneck(enc4)
        
        # 解码器前向传播，使用跳跃连接
        dec4 = self.decoder4[:1](bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.decoder4[1:](dec4)
        
        dec3 = self.decoder3[:1](dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.decoder3[1:](dec3)
        
        dec2 = self.decoder2[:1](dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2[1:](dec2)
        
        dec1 = self.decoder1[:1](dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1[1:](dec1)
        
        # 输出层
        out = self.out(dec1)
        
        return out
