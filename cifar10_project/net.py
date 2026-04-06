"""
神经网络模型定义
包含CIFAR-10分类网络的架构
"""
# 导入 PyTorch 库
import torch
# 导入 PyTorch 的神经网络模块
import torch.nn as nn


class Net(nn.Module):
    """
    CIFAR-10分类网络
    架构：Conv -> BatchNorm -> ReLU -> Pool -> Conv -> BatchNorm -> ReLU -> Pool -> FC
    """
    def __init__(self):
        # 调用父类 nn.Module 的初始化方法
        super().__init__()
        # 第一层卷积层：输入通道3（RGB图像），输出通道16，卷积核大小3x3
        # 卷积核大小默认为 3x3，步长默认为 1，填充默认为 0
        self.conv1 = nn.Conv2d(3, 16, 3)
        # 批归一化层：对卷积1的输出进行归一化，加速训练收敛
        # 每个通道独立计算均值和方差，包含可学习的缩放因子(γ)和平移因子(β)
        # 参数 16 表示输入通道数
        self.bn1 = nn.BatchNorm2d(16)
        # 最大池化层：池化核大小2x2，步长2
        # 池化核大小 2x2，步长 2，填充 0
        self.pool = nn.MaxPool2d(2, 2)
        # 第二层卷积层：输入通道16，输出通道32，卷积核大小3x3
        self.conv2 = nn.Conv2d(16, 32, 3)
        # 批归一化层：对卷积2的输出进行归一化
        # 参数 32 表示输入通道数
        self.bn2 = nn.BatchNorm2d(32)
        # 第一个全连接层：输入特征数32*6*6（经过两次池化后的特征图大小），输出120
        # 计算过程：32x32输入 → 卷积后30x30 → 池化后15x15 → 卷积后13x13 → 池化后6x6
        self.fc1 = nn.Linear(32 * 6 * 6, 120)
        # 第二个全连接层：输入120，输出84
        self.fc2 = nn.Linear(120, 84)
        # 第三个全连接层：输入84，输出10（CIFAR-10的类别数）
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """
        前向传播方法，定义数据在网络中的流动路径
        Args:
            x: 输入张量，形状为 [batch_size, 3, 32, 32]
        Returns:
            输出张量，形状为 [batch_size, 10]
        """
        # 卷积1 -> 批归一化 -> ReLU激活 -> 池化
        # 批归一化在激活函数之前，有助于保持输入分布的稳定性
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        # 卷积2 -> 批归一化 -> ReLU激活 -> 池化
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        # 将特征图展平为一维向量，从第1维开始（第0维是批次维度）
        # 展平后形状：[batch_size, 32*6*6]
        x = torch.flatten(x, 1)
        # 全连接1 -> ReLU激活
        x = torch.relu(self.fc1(x))
        # 全连接2 -> ReLU激活
        x = torch.relu(self.fc2(x))
        # 全连接3，输出最终结果（未经过softmax，CrossEntropyLoss会自动处理）
        x = self.fc3(x)
        # 返回输出张量
        return x
