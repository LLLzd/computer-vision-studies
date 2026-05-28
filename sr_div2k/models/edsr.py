import torch
import torch.nn as nn

class EDSR(nn.Module):
    """EDSR模型实现
    基于论文：Enhanced Deep Residual Networks for Single Image Super-Resolution
    """
    def __init__(self, upscale_factor=3, num_channels=3, num_resblocks=16, num_features=64):
        """初始化模型
        Args:
            upscale_factor: 超分倍数
            num_channels: 输入图像通道数（RGB为3）
            num_resblocks: 残差块数量
            num_features: 特征通道数
        """
        super(EDSR, self).__init__()
        
        # 初始卷积层：提取特征
        self.conv1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=1)
        
        # 残差块
        self.resblocks = nn.Sequential(*[
            ResidualBlock(num_features) for _ in range(num_resblocks)
        ])
        
        # 残差连接后的卷积层
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        
        # 上采样层
        if upscale_factor == 2:
            self.upsampler = nn.Sequential(
                nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2)
            )
        elif upscale_factor == 3:
            self.upsampler = nn.Sequential(
                nn.Conv2d(num_features, num_features * 9, kernel_size=3, padding=1),
                nn.PixelShuffle(3)
            )
        elif upscale_factor == 4:
            self.upsampler = nn.Sequential(
                nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2)
            )
        else:
            raise ValueError(f"不支持的超分倍数: {upscale_factor}")
        
        # 输出卷积层
        self.conv3 = nn.Conv2d(num_features, num_channels, kernel_size=3, padding=1)
        
        # 激活函数：ReLU
        self.relu = nn.ReLU()
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播
        Args:
            x: 输入低分辨率图像，形状为 [batch_size, channels, height, width]
        Returns:
            输出高分辨率图像，形状为 [batch_size, channels, height*upscale_factor, width*upscale_factor]
        """
        # 初始特征提取
        x = self.relu(self.conv1(x))
        
        # 残差块处理
        residual = x
        x = self.resblocks(x)
        x = self.conv2(x)
        x += residual  # 残差连接
        
        # 上采样
        x = self.upsampler(x)
        
        # 输出
        x = self.conv3(x)
        return x

class ResidualBlock(nn.Module):
    """残差块
    """
    def __init__(self, num_features):
        """初始化残差块
        Args:
            num_features: 特征通道数
        """
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化残差块权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播
        Args:
            x: 输入特征
        Returns:
            输出特征
        """
        residual = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        x += residual  # 残差连接
        return x
