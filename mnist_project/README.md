# MNIST 手写数字识别项目

## 项目结构

```
mnist_project/
├── config.py      # 配置文件（路径、超参数、数据预处理）
├── net.py         # 神经网络模型定义
├── train.py       # 训练脚本
├── test.py        # 测试/评估脚本
├── infer.py       # 推理脚本
├── visual.py      # 可视化脚本
├── data/          # 数据集目录
├── models/        # 模型保存目录
└── outputs/       # 输出目录
```

## 快速开始

### 1. 下载数据集

```bash
python download_data.py
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

### 5. 可视化样本

```bash
python visual.py
```

## 技术要点

- **输入尺寸**：MNIST 图像为 28x28 灰度图像
- **网络结构**：Conv -> BatchNorm -> ReLU -> Pool -> FC
- **损失函数**：交叉熵损失
- **优化器**：Adam 优化器
