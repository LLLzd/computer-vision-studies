# ESPCN 超分辨率项目

基于 ESPCN (Efficient Sub-Pixel Convolutional Neural Network) 的图像超分辨率项目，适用于 M1 Mac。

## 📁 项目结构

```
sr_DIV2K_ESPCN/
├── config.py              # 配置文件（超参数、路径等）
├── models/                # 模型目录
│   ├── __init__.py
│   └── espcn.py          # ESPCN 模型定义
├── train_espcn.py        # 训练脚本
├── evaluate.py           # 评估脚本
├── inference.py          # 推理脚本
├── visual_data.py        # 数据可视化脚本
├── download_div2k.py     # 数据集下载脚本
├── data/                 # 数据集目录
│   ├── DIV2K_train_HR/
│   ├── DIV2K_train_LR_bicubic/X3/
│   ├── DIV2K_valid_HR/
│   ├── DIV2K_valid_LR_bicubic/X3/
│   └── ...
├── weights/              # 模型权重目录
│   └── espcn_best.pth
└── outputs/              # 输出目录
    ├── evaluation_results.png
    ├── evaluation_metrics.txt
    └── inference/
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install torch torchvision pillow numpy tqdm matplotlib scikit-image
```

### 2. 下载数据集

```bash
python download_div2k.py
```

### 3. 训练模型

```bash
python train_espcn.py
```

### 4. 评估模型

```bash
python evaluate.py
```

### 5. 推理测试

```bash
# 单张图像
python inference.py --input path/to/image.png

# 批量处理
python inference.py --input path/to/images/
```

## ⚙️ 配置说明

所有配置参数都在 `config.py` 中：

### 设备配置
- `DEVICE`: 设备选择（自动检测 MPS/CPU）

### 模型配置
- `UPSCALE_FACTOR`: 超分倍数（默认：3）
- `NUM_CHANNELS`: 输入图像通道数（默认：3）

### 训练配置
- `BATCH_SIZE`: 批次大小（默认：8）
- `EPOCHS`: 训练轮次（默认：100）
- `LR`: 学习率（默认：1e-3）

### 数据集路径
- `TRAIN_HR/LR`: 训练集路径
- `VAL_HR/LR`: 验证集路径
- `TEST_HR/LR`: 测试集路径

### 输出路径
- `MODEL_PATH`: 模型保存路径
- `OUTPUT_DIR`: 输出目录

## 📊 模型性能

当前模型（训练 1 个 epoch）：
- **平均 PSNR**: 26.59 dB
- **平均 SSIM**: 0.7921

## 🔧 模型架构

ESPCN 模型包含：
1. **特征提取层 1**: 5x5 卷积，64 通道
2. **特征提取层 2**: 3x3 卷积，64 通道
3. **亚像素卷积层**: 3x3 卷积，亚像素洗牌操作

## 📝 使用示例

### 可视化训练数据

```bash
python visual_data.py
```

### 修改训练参数

编辑 `config.py` 文件：

```python
# 训练配置
BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-3
```

### 自定义推理输出

```bash
python inference.py --input image.png --output custom_output/
```

## 📚 参考文献

- Shi, W., Caballero, J., Huszár, F., Totz, J., Aitken, A. P., Bishop, R., ... & Wang, Z. (2016). Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1874-1883).

## 🎯 改进建议

1. **增加训练轮次**: 将 `EPOCHS` 设置为 100 或更多
2. **调整学习率**: 使用学习率调度器
3. **数据增强**: 添加旋转、翻转等数据增强
4. **模型改进**: 尝试更深的网络结构

## 📧 联系方式

如有问题，请提交 Issue 或 Pull Request。
