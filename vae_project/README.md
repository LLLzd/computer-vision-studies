# VAE 变分自编码器项目

## 项目简介

本项目实现了一个标准的 Variational Autoencoder (VAE) PyTorch 工程，默认面向 MNIST 数据集，支持：

- 图像重构（Reconstruction）
- 随机生成（Generation）
- 2D 隐空间可视化（t-SNE）
- 快速训练与标准训练两种模式

工程结构遵循科研项目组织方式，适合研究生学习和实验复现。

## 项目结构

```text
vae_project/
├── data/
│   └── MNIST -> ../../mnist_project/data/MNIST  # 软链接共享数据
├── configs/
│   └── default.yaml
├── src/
│   ├── datasets/
│   │   └── mnist_dataset.py
│   ├── losses/
│   │   └── elbo_loss.py
│   ├── models/
│   │   └── vae.py
│   ├── trainer/
│   │   ├── quick_trainer.py
│   │   └── standard_trainer.py
│   └── utils/
│       ├── config_utils.py
│       ├── device_utils.py
│       ├── logger.py
│       ├── metrics.py
│       ├── seed.py
│       └── visualizer.py
├── train_quick.py
├── train_standard.py
├── infer.py
├── visualize.py
├── requirements.txt
└── outputs/   # 运行时自动创建
```

## 环境安装

```bash
cd /Users/rik/workspace/study/vae_project
pip install -r requirements.txt
```

## 快速开始

### 0) 共享 `mnist_project` 的数据（已就绪）

本项目支持直接复用 `study/mnist_project/data/MNIST`，无需重复下载。

已创建软链接：

```bash
vae_project/data/MNIST -> ../../mnist_project/data/MNIST
```

### 1) Quick 模式（快速验证）

```bash
python train_quick.py
```

默认配置：
- `epochs=10`
- `batch_size=128`
- 保存最终模型 `outputs/checkpoints/vae_final.pt`
- 输出关键可视化（重构图、生成图、t-SNE）

### 2) Standard 模式（正式实验）

```bash
python train_standard.py
```

默认配置：
- `epochs=50`
- `batch_size=64`
- 每个 epoch 保存 checkpoint
- 记录 train/val loss，自动生成 loss 曲线

### 3) Infer（推理）

```bash
python infer.py --checkpoint outputs/checkpoints/best.pt
```

输出：
- `outputs/visuals/infer_reconstruction.png`（单样本原图/重构对比）
- `outputs/visuals/infer_generation.png`（随机生成样本）

### 4) Visualize（独立可视化）

```bash
python visualize.py --checkpoint outputs/checkpoints/best.pt
```

输出：
- `outputs/visuals/reconstruction.png`
- `outputs/visuals/generation.png`
- `outputs/visuals/latent_tsne.png`

## 输出目录说明

训练运行后会自动创建：

- `outputs/checkpoints/`：模型权重（每轮、最优、最终）
- `outputs/logs/`：日志与 metrics（json/csv）
- `outputs/visuals/`：
  - `reconstruction.png`
  - `generation.png`
  - `latent_tsne.png`
  - `loss_curve.png`（standard 模式）
  - `infer_reconstruction.png`（infer 脚本）
  - `infer_generation.png`（infer 脚本）

## 配置说明

核心超参数集中在 `configs/default.yaml`：

- 模型：`latent_dim`, `hidden_dims`, `in_channels`
- 训练：`lr`, `beta`
- 模式：`quick` / `standard`
- 数据：`dataset.name`（`mnist` / `fashion_mnist` / `celeba`）
- 设备：`device.type`（`auto`/`cpu`/`cuda`/`mps`）

## 数据集切换示例

将 `configs/default.yaml` 中：

```yaml
dataset:
  name: fashion_mnist
  image_size: 28
```

即可切换到 FashionMNIST。

切换 CelebA 时建议同步设置：

```yaml
dataset:
  name: celeba
  image_size: 64

model:
  in_channels: 3
```

## 训练日志打印

训练中会按迭代打印：

```text
Epoch [e/E] Iter [i/I] loss=... recon_loss=... kl_loss=...
```

## 可复现性

项目默认固定随机种子，并设置：

- `random / numpy / torch` 随机种子
- CUDA 场景下 `cudnn.deterministic=True`

## 论文参考

- Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes. [https://arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114)
