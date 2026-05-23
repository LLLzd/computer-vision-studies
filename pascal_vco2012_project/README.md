# Pascal VOC 2012 图像分割项目

基于 PyTorch 的 Pascal VOC 2012 语义分割练习项目，使用 U-Net 结构对 21 类（含背景）进行像素级分类。

## 项目结构

```text
pascal_vco2012_project/
├── config.py
├── download.py
├── train.py
├── infer.py
├── net.py
├── visual_dataset.py
├── count_annotations.py
├── data/
├── models/
└── outputs/
```

## 环境要求

- Python 3.9+
- PyTorch / torchvision
- numpy、Pillow、matplotlib、tqdm、requests

## 快速开始

```bash
cd pascal_vco2012_project
python download.py
python train.py
python infer.py
```

## 功能说明

- `download.py`：下载并校验 VOC2012 数据集目录结构
- `train.py`：训练分割模型（含组合损失与训练过程记录）
- `infer.py`：加载权重并对样本进行可视化推理
- `visual_dataset.py`：数据集加载与可视化相关逻辑

## 输出目录

- `models/`：训练权重与检查点
- `outputs/`：推理可视化和训练中间结果

## 常见问题

- 下载慢或失败：检查网络后重试 `python download.py`
- 训练内存不足：调小 `config.py` 中 `BATCH_SIZE` 或 `IMAGE_SIZE`
- 推理结果异常：确认模型权重路径与类别配置一致
