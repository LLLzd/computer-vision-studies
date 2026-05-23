# VOC2012 Anchor-Free 2D 目标检测项目

该项目基于 Pascal VOC 2012 实现 Anchor-Free 检测训练与评估流程，采用 heatmap + offset + wh 的预测头设计。

## 项目结构

```text
vco2012_od2d/
├── config.yaml
├── config.py
├── dataset.py
├── net.py
├── train.py
├── test.py
├── infer.py
├── preprocess.py
├── scripts/
└── outputs/
```

## 环境要求

- Python 3.9+
- PyTorch / torchvision
- numpy、matplotlib、tqdm、pyyaml

## 快速开始

```bash
cd vco2012_od2d
python train.py --config config.yaml
python test.py --config config.yaml
python infer.py --config config.yaml
```

## 功能说明

- `train.py`：训练流程，支持 warmup + 学习率调度 + early stopping
- `test.py`：评估流程，包含解码、NMS、mAP/Precision/Recall 等指标
- `infer.py`：单图或样例推理与结果可视化
- `config.yaml`：统一管理类别、输入尺寸、阈值、训练超参

## 输出目录

- `outputs/weights/`：模型权重
- `outputs/`：评测结果、可视化输出等

## 常见问题

- 配置不生效：确认命令带了 `--config config.yaml`
- 显存不足：降低 `BATCH_SIZE` 或输入分辨率
- 检测框偏少：适当降低 `THRESHOLD`，并检查类别过滤配置
