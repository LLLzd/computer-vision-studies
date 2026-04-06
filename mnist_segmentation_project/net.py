"""
模型定义文件
定义用于 MNIST 分割的 U-Net 模型
"""

# 导入必要的库
import torch  # PyTorch 核心库
import torch.nn as nn  # 神经网络模块


class UNet(nn.Module):
    """
    U-Net 模型
    用于图像分割任务
    """
    
    def __init__(self, in_channels=1, out_channels=1):
        """
        初始化 U-Net 模型
        Args:
            in_channels: 输入通道数，MNIST 为 1（灰度图像）
            out_channels: 输出通道数，分割任务为 1（二值分割）
        """
        super(UNet, self).__init__()  # 调用父类的初始化方法
        
        # 编码器（下采样）
        self.encoder = nn.ModuleList([
            # 第一层：输入 1x28x28
            nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),  # 16x28x28
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, kernel_size=3, padding=1),  # 16x28x28
                nn.ReLU(inplace=True)
            ),
            # 第二层：输入 16x28x28
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),  # 16x14x14
                nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 32x14x14
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),  # 32x14x14
                nn.ReLU(inplace=True)
            ),
            # 第三层：输入 32x14x14
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),  # 32x7x7
                nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64x7x7
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 64x7x7
                nn.ReLU(inplace=True)
            )
        ])
        
        # 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x3x3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 128x3x3
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 128x3x3
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, output_padding=1)  # 64x7x7
        )
        
        # 解码器（上采样）
        self.decoder = nn.ModuleList([
            # 第一层：输入 64x7x7 + 64x7x7（跳跃连接）
            nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 64x7x7
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 64x7x7
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # 32x14x14
            ),
            # 第二层：输入 32x14x14 + 32x14x14（跳跃连接）
            nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=3, padding=1),  # 32x14x14
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),  # 32x14x14
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)  # 16x28x28
            ),
            # 第三层：输入 16x28x28 + 16x28x28（跳跃连接）
            nn.Sequential(
                nn.Conv2d(32, 16, kernel_size=3, padding=1),  # 16x28x28
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, kernel_size=3, padding=1),  # 16x28x28
                nn.ReLU(inplace=True),
                nn.Conv2d(16, out_channels, kernel_size=1)  # 1x28x28
            )
        ])
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量，形状为 [batch_size, in_channels, H, W]
        Returns:
            输出张量，形状为 [batch_size, out_channels, H, W]
        """
        # 存储编码器的特征图，用于跳跃连接
        encoder_features = []
        
        # 编码器前向传播
        for block in self.encoder:
            x = block(x)
            encoder_features.append(x)
        
        # 瓶颈层前向传播
        x = self.bottleneck(x)
        
        # 解码器前向传播，结合跳跃连接
        for i, block in enumerate(self.decoder):
            # 获取对应编码器的特征图
            encoder_feature = encoder_features[-(i+1)]
            
            # 拼接解码器输入和编码器特征图（跳跃连接）
            x = torch.cat([x, encoder_feature], dim=1)
            # 解码器块前向传播
            x = block(x)
        
        return x
