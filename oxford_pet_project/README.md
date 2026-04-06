# Oxford-IIIT Pet 图像分割项目

## 项目简介

本项目使用卷积神经网络（CNN）对 Oxford-IIIT Pet 数据集进行图像分割。Oxford-IIIT Pet 数据集包含 37 个类别的宠物图像，每个图像都有对应的分割标注。

## 项目结构

```
oxford_pet_project/
├── config.py      # 配置文件（路径、超参数、数据预处理）
├── net.py         # 分割模型定义（U-Net 架构）
├── train.py       # 训练脚本
├── test.py        # 测试/评估脚本
├── infer.py       # 推理脚本
├── visual.py      # 可视化脚本
├── download.py    # 数据集下载脚本
├── data/          # 数据集目录
├── models/        # 模型保存目录
└── outputs/       # 输出目录
```

## 数据集

Oxford-IIIT Pet 数据集包含：
- 37 个宠物类别（12 种猫，25 种狗）
- 总共 7349 张图像
- 每个图像都有分割标注（前景/背景）

**信息来源**：[Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)

## 快速开始

### 1. 下载数据集

```bash
python download.py
```

### 2. 训练模型

```bash
python train.py
```

### 3. 测试模型

```bash
python test.py
```

### 4. 推理单张图片

```bash
python infer.py
```

### 5. 可视化结果

```bash
python visual.py
```

## 模型架构

使用 U-Net 架构：
- 编码器（下采样）：提取特征
- 解码器（上采样）：恢复空间维度
- 跳跃连接：保留细节信息

## 技术要点

- **数据预处理**：图像和标签的缩放、归一化
- **损失函数**：交叉熵损失 + Dice 损失
- **数据增强**：随机翻转、旋转、亮度调整
- **评估指标**：IoU（Intersection over Union）

## 预期结果

训练后，模型应该能够准确分割出宠物的轮廓，IoU 指标在测试集上达到 80% 以上。
