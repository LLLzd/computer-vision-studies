"""
分割模型定义
使用 U-Net 架构进行图像分割
"""

# 导入必要的库
import torch  # PyTorch 核心库
import torch.nn as nn  # 神经网络模块


class UNet(nn.Module):
    """
    U-Net 分割模型
    编码器-解码器架构，带有跳跃连接
    适用于图像分割任务
    """
    
    def __init__(self, in_channels=3, out_channels=1):
        """
        初始化模型
        Args:
            in_channels: 输入通道数（RGB图像为3）
            out_channels: 输出通道数（3分类分割为1）
        """
        super(UNet, self).__init__()  # 调用父类初始化方法
        
        # 编码器（下采样）部分
        # 用于提取图像特征，每一层都进行卷积和下采样
        self.encoder = nn.ModuleList([
            # 第一层：输入 -> 16通道
            # 输入形状: [batch_size, 3, 256, 256]
            # 输出形状: [batch_size, 16, 256, 256]
            nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),  # 3x3 卷积，保持尺寸
                nn.BatchNorm2d(16),  # 批归一化，加速训练
                nn.ReLU(inplace=True),  # ReLU 激活函数
                nn.Conv2d(16, 16, kernel_size=3, padding=1),  # 3x3 卷积，保持尺寸
                nn.BatchNorm2d(16),  # 批归一化
                nn.ReLU(inplace=True)  # ReLU 激活函数
            ),
            # 第二层：16 -> 32通道
            # 输入形状: [batch_size, 16, 256, 256]
            # 输出形状: [batch_size, 32, 128, 128]
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),  # 2x2 最大池化，尺寸减半
                nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 3x3 卷积
                nn.BatchNorm2d(32),  # 批归一化
                nn.ReLU(inplace=True),  # ReLU 激活函数
                nn.Conv2d(32, 32, kernel_size=3, padding=1),  # 3x3 卷积
                nn.BatchNorm2d(32),  # 批归一化
                nn.ReLU(inplace=True)  # ReLU 激活函数
            ),
            # 第三层：32 -> 64通道
            # 输入形状: [batch_size, 32, 128, 128]
            # 输出形状: [batch_size, 64, 64, 64]
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),  # 2x2 最大池化，尺寸减半
                nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 3x3 卷积
                nn.BatchNorm2d(64),  # 批归一化
                nn.ReLU(inplace=True),  # ReLU 激活函数
                nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 3x3 卷积
                nn.BatchNorm2d(64),  # 批归一化
                nn.ReLU(inplace=True)  # ReLU 激活函数
            ),
            # 第四层：64 -> 128通道（瓶颈层）
            # 输入形状: [batch_size, 64, 64, 64]
            # 输出形状: [batch_size, 128, 32, 32]
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),  # 2x2 最大池化，尺寸减半
                nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 3x3 卷积
                nn.BatchNorm2d(128),  # 批归一化
                nn.ReLU(inplace=True),  # ReLU 激活函数
                nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 3x3 卷积
                nn.BatchNorm2d(128),  # 批归一化
                nn.ReLU(inplace=True)  # ReLU 激活函数
            )
        ])
        
        # 解码器（上采样）部分
        # 用于恢复空间维度，每一层都进行上采样和卷积
        self.decoder = nn.ModuleList([
            # 第一层：128 -> 64通道
            # 输入形状: [batch_size, 128, 32, 32]
            # 输出形状: [batch_size, 64, 64, 64]
            nn.Sequential(
                # 2x2 转置卷积，用于上采样，尺寸加倍
                # 原理：通过在输入特征图之间插入零填充，然后应用普通卷积
                # 计算过程：输入尺寸为 H x W，stride=2 时，输出尺寸为 (H-1)*stride + kernel_size = (32-1)*2 + 2 = 64
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # 2x2 转置卷积，尺寸加倍
                # 3x3 卷积，用于特征融合（与编码器跳跃连接拼接）
                # 输入通道数 = 转置卷积输出通道数 + 编码器对应层的通道数 = 64 + 64 = 128
                # 输出通道数 = 64
                # 计算过程：每个输出通道是所有输入通道与对应卷积核的加权和
                # 参数量：output_channels * (input_channels * kernel_size^2 + 1) = 64 * (128 * 3^2 + 1) = 64 * 1153 = 73792
                nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 3x3 卷积（与编码器跳跃连接拼接）
                nn.BatchNorm2d(64),  # 批归一化
                nn.ReLU(inplace=True),  # ReLU 激活函数
                nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 3x3 卷积
                nn.BatchNorm2d(64),  # 批归一化
                nn.ReLU(inplace=True)  # ReLU 激活函数
            ),
            # 第二层：64 -> 32通道
            # 输入形状: [batch_size, 64, 64, 64]
            # 输出形状: [batch_size, 32, 128, 128]
            nn.Sequential(
                # 2x2 转置卷积，用于上采样，尺寸加倍
                # 输出尺寸：(64-1)*2 + 2 = 128
                nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # 2x2 转置卷积，尺寸加倍
                # 3x3 卷积，用于特征融合（与编码器跳跃连接拼接）
                # 输入通道数 = 转置卷积输出通道数 + 编码器对应层的通道数 = 32 + 32 = 64
                # 输出通道数 = 32
                # 参数量：32 * (64 * 3^2 + 1) = 32 * 577 = 18464
                nn.Conv2d(64, 32, kernel_size=3, padding=1),  # 3x3 卷积（与编码器跳跃连接拼接）
                nn.BatchNorm2d(32),  # 批归一化
                nn.ReLU(inplace=True),  # ReLU 激活函数
                nn.Conv2d(32, 32, kernel_size=3, padding=1),  # 3x3 卷积
                nn.BatchNorm2d(32),  # 批归一化
                nn.ReLU(inplace=True)  # ReLU 激活函数
            ),
            # 第三层：32 -> 16通道
            # 输入形状: [batch_size, 32, 128, 128]
            # 输出形状: [batch_size, 16, 256, 256]
            nn.Sequential(
                # 2x2 转置卷积，用于上采样，尺寸加倍
                # 输出尺寸：(128-1)*2 + 2 = 256
                nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # 2x2 转置卷积，尺寸加倍
                # 3x3 卷积，用于特征融合（与编码器跳跃连接拼接）
                # 输入通道数 = 转置卷积输出通道数 + 编码器对应层的通道数 = 16 + 16 = 32
                # 输出通道数 = 16
                # 参数量：16 * (32 * 3^2 + 1) = 16 * 289 = 4624
                nn.Conv2d(32, 16, kernel_size=3, padding=1),  # 3x3 卷积（与编码器跳跃连接拼接）
                nn.BatchNorm2d(16),  # 批归一化
                nn.ReLU(inplace=True),  # ReLU 激活函数
                nn.Conv2d(16, 16, kernel_size=3, padding=1),  # 3x3 卷积
                nn.BatchNorm2d(16),  # 批归一化
                nn.ReLU(inplace=True)  # ReLU 激活函数
            )
        ])
        
        # 输出层
        # 将 16 通道转换为 3 通道（三分类分割）
        # 输入形状: [batch_size, 16, 256, 256]
        # 输出形状: [batch_size, 3, 256, 256]
        # 使用 1x1 卷积的原因：
        # 1. 保持空间尺寸不变（padding=0，stride=1）
        # 2. 高效调整通道数
        # 3. 计算量小，参数量少
        # 参数量：output_channels * (input_channels * 1^2 + 1) = 3 * (16 * 1 + 1) = 3 * 17 = 51
        # 与 Linear 层的区别：Linear 层会将特征展平为一维，而 1x1 卷积保持空间结构
        self.out = nn.Conv2d(16, out_channels, kernel_size=1)  # 1x1 卷积，不改变尺寸
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量，形状为 [batch_size, 3, 256, 256]
        Returns:
            输出张量，形状为 [batch_size, 1, 256, 256]
        """
        # 编码器前向传播，保存中间特征图
        encoder_features = []
        for block in self.encoder:
            x = block(x)  # 通过编码器块
            encoder_features.append(x)  # 保存特征图用于跳跃连接
        
        # 移除最后一个特征图（瓶颈层），因为它不需要跳跃连接
        encoder_features = encoder_features[:-1]
        
        # 解码器前向传播，使用跳跃连接
        for i, block in enumerate(self.decoder):
            # 上采样
            x = block[:1](x)  # 只取 ConvTranspose2d 部分
            # 跳跃连接：拼接编码器对应层的特征图
            x = torch.cat([x, encoder_features[-(i+1)]], dim=1)  # 沿通道维度拼接
            # 剩余操作
            x = block[1:](x)  # 执行剩余的卷积和激活操作
        
        # 输出层
        x = self.out(x)  # 1x1 卷积得到最终输出
        
        return x
