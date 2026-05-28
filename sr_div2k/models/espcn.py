import torch
import torch.nn as nn

class ESPCN(nn.Module):
    """ESPCN模型实现
    基于论文：Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
    """
    def __init__(self, upscale_factor=3, num_channels=3):
        """初始化模型
        Args:
            upscale_factor: 超分倍数
            num_channels: 输入图像通道数（RGB为3）
        """
        super(ESPCN, self).__init__()
        # 特征提取层1：5x5卷积核，64个通道
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=5, padding=2)
        # 特征提取层2：3x3卷积核，64个通道
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # 亚像素卷积层：3x3卷积核，通道数为 num_channels * (upscale_factor^2)
        self.conv3 = nn.Conv2d(64, num_channels * (upscale_factor ** 2), kernel_size=3, padding=1)
        # 亚像素洗牌操作：将通道维度的信息转换为空间维度
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        # 激活函数：ReLU
        self.relu = nn.ReLU()

    def forward(self, x):
        """前向传播
        Args:
            x: 输入低分辨率图像，形状为 [batch_size, channels, height, width]
        Returns:
            输出高分辨率图像，形状为 [batch_size, channels, height*upscale_factor, width*upscale_factor]
        """
        # 第一层卷积 + ReLU激活
        x = self.relu(self.conv1(x))
        # 第二层卷积 + ReLU激活
        x = self.relu(self.conv2(x))
        # 第三层卷积 + 亚像素洗牌操作
        x = self.pixel_shuffle(self.conv3(x))
        return x
