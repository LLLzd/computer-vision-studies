# CIFAR-10 图像分类项目

## 项目简介

本项目使用卷积神经网络（CNN）对 CIFAR-10 数据集进行图像分类。CIFAR-10 包含 10 个类别的 60000 张 32x32 彩色图像。

## 项目结构

```
cifar10_project/
├── config.py      # 配置文件（路径、超参数、数据预处理）
├── net.py         # 神经网络模型定义（包含 BatchNorm）
├── train.py       # 训练脚本
├── test.py        # 测试/评估脚本
├── infer.py       # 推理脚本
├── visual.py      # 可视化脚本
├── data/          # 数据集目录
├── models/        # 模型保存目录
└── outputs/       # 输出目录
```

## 数据集

CIFAR-10 包含以下 10 个类别：
- 飞机（plane）
- 汽车（car）
- 鸟类（bird）
- 猫（cat）
- 鹿（deer）
- 狗（dog）
- 青蛙（frog）
- 马（horse）
- 船（ship）
- 卡车（truck）

## 快速开始

### 1. 训练模型

```bash
python train.py
```

首次运行会自动下载 CIFAR-10 数据集（约 170MB）。

### 2. 测试模型

```bash
python test.py
```

### 3. 推理单张图片

```bash
python infer.py
```

### 4. 可视化样本

```bash
python visual.py
```

## 网络架构

```
输入: [batch_size, 3, 32, 32]
  ↓
Conv2d(3, 16, 3x3) → BatchNorm2d(16) → ReLU → MaxPool2d(2x2)
  ↓
Conv2d(16, 32, 3x3) → BatchNorm2d(32) → ReLU → MaxPool2d(2x2)
  ↓
Flatten: [batch_size, 32*6*6]
  ↓
Linear(32*6*6, 120) → ReLU
  ↓
Linear(120, 84) → ReLU
  ↓
Linear(84, 10)
  ↓
输出: [batch_size, 10]
```

## 超参数配置

- **批次大小**: 64
- **学习率**: 0.001
- **训练轮数**: 10
- **优化器**: Adam
- **损失函数**: CrossEntropyLoss

## 技术要点

- **数据预处理**: 归一化到 [-1, 1]
- **批归一化**: 加速训练收敛，提高模型稳定性
- **数据增强**: 训练时随机打乱数据

## 预期结果

训练 10 个 epoch 后，测试集准确率约为 70-75%。
