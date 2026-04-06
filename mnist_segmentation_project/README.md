# MNIST 分割项目

这是一个基于 U-Net 模型的 MNIST 手写数字分割项目。

## 项目结构

```
mnist_segmentation_project/
├── config.py          # 配置文件
├── net.py             # 模型定义
├── train.py           # 训练脚本
├── test.py            # 测试脚本
├── infer.py           # 推理脚本
├── data/              # 数据集目录
├── models/            # 模型保存目录
├── outputs/           # 输出目录
└── README.md          # 项目说明
```

## 功能说明

1. **配置文件 (config.py)**
   - 定义路径、超参数和数据预处理
   - 支持自动创建必要的目录

2. **模型定义 (net.py)**
   - 实现了一个适用于 MNIST 分割的 U-Net 模型
   - 包含编码器、瓶颈层和解码器
   - 支持跳跃连接

3. **训练脚本 (train.py)**
   - 定义了 MNISTSegmentationDataset 类，将 MNIST 转换为分割任务
   - 使用 Dice 损失函数
   - 支持自动下载数据集

4. **测试脚本 (test.py)**
   - 计算 IoU 评估指标
   - 测试模型性能

5. **推理脚本 (infer.py)**
   - 对单张图片进行分割预测
   - 可视化分割结果

## 如何使用

1. **训练模型**
   ```bash
   python train.py
   ```

2. **测试模型**
   ```bash
   python test.py
   ```

3. **推理单张图片**
   ```bash
   python infer.py
   ```

## 技术细节

- **模型**：U-Net
- **损失函数**：Dice Loss
- **优化器**：Adam
- **批次大小**：16
- **学习率**：0.001
- **训练轮数**：10

## 注意事项

- 首次运行时会自动下载 MNIST 数据集
- 训练完成后，模型会保存到 `models/unet_mnist.pth`
- 推理结果会保存到 `outputs/infer_result.png`

## 预期结果

训练完成后，模型应该能够：
- 准确分割 MNIST 手写数字
- 生成清晰的分割掩码
- 在测试集上获得较高的 IoU 分数
