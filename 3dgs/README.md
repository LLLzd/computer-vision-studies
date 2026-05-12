# 3D Gaussian Splatting (3DGS) 项目

一个基于3D高斯分布的场景重建和渲染项目，支持从视频序列重建3D场景并进行多视角渲染。

## 项目结构

```
3dgs/
├── main.py                 # 主入口脚本，整合所有功能
├── input/                  # 输入数据目录
│   ├── videos/            # 原始视频文件
│   └── frames/            # 提取的帧图像
├── preprocess/            # 预处理脚本
│   ├── extract_frames.py  # 从视频中提取帧
│   └── initialize.py      # 初始化3D高斯（SfM）
├── utils/                 # 工具模块
│   └── core.py           # 核心类（Gaussian3D, Camera, GaussianRenderer）
├── train/                 # 训练脚本
│   ├── train.py          # NumPy训练（带优化器类）
│   ├── train_simple.py   # NumPy简化版训练（快速原型）
│   ├── train_torch.py    # PyTorch训练演示（教学用）
│   └── train_3dgs.py     # PyTorch完整版训练（GPU加速）
├── render/               # 渲染脚本
│   ├── render.py        # PyTorch渲染器（GPU加速）
│   └── render_simple.py # NumPy渲染器（简化版）
└── output/              # 输出目录
    ├── gaussians_init.pkl      # 初始化的高斯
    ├── gaussians_trained.pkl   # 训练后的高斯
    ├── comparison_*.png        # 训练过程对比图
    ├── loss_curve.png         # 损失曲线
    └── renders/               # 渲染结果
        └── *.png              # 渲染图像
```

## 脚本说明

### 预处理脚本 (preprocess/)

#### `extract_frames.py`
从视频中提取帧图像
- 功能：读取视频文件，按指定间隔提取帧
- 参数：
  - `--video`: 输入视频路径
  - `--output`: 输出目录（默认：frames）
  - `--sample_rate`: 采样间隔（默认：5）
  - `--max_frames`: 最大帧数（默认：100）
  - `--resize`: 缩放因子（默认：0.5）

#### `initialize.py`
使用SfM初始化3D高斯
- 功能：从帧序列重建3D点云并初始化高斯
- 参数：
  - `--frames`: 帧目录
  - `--output`: 输出目录（默认：output）
  - `--num_gaussians`: 高斯数量（默认：50000）
  - `--max_frames`: 最大处理帧数（默认：30）

### 工具模块 (utils/)

#### `core.py`
核心类定义
- `Gaussian3D`: 3D高斯类（位置、协方差、颜色、不透明度）
- `Camera`: 相机类（位置、朝向、内参）
- `GaussianRenderer`: 高斯渲染器（3D到2D投影）

### 训练脚本 (train/)

#### `train_simple.py` ⭐ 推荐
NumPy简化版训练
- 特点：快速原型，优化版，适合M1/M2芯片
- 参数：
  - `--frames`: 帧目录
  - `--init`: 初始化文件
  - `--output`: 输出目录
  - `--iterations`: 迭代次数（默认：100）
  - `--max-gaussians`: 最大高斯数（默认：2000）
  - `--save-interval`: 保存间隔（默认：10）

#### `train_torch.py`
PyTorch训练演示
- 特点：详细注释，教学用，展示完整原理
- 参数：
  - 图像尺寸、高斯数量、训练轮数、学习率

#### `train_3dgs.py`
PyTorch完整版训练
- 特点：GPU加速，真正的可微渲染，支持自动求导
- 参数：
  - `--frames`: 帧目录
  - `--init`: 初始化文件
  - `--output`: 输出目录
  - `--iterations`: 迭代次数（默认：500）
  - `--lr`: 学习率（默认：0.01）

#### `train.py`
NumPy训练（带优化器类）
- 特点：包含优化器类，结构化训练流程
- 功能：创建合成目标图像进行训练演示

### 渲染脚本 (render/)

#### `render_simple.py` ⭐ 推荐
NumPy简化版渲染器
- 特点：快速渲染，支持多种模式
- 参数：
  - `--gaussians`: 高斯文件
  - `--frames`: 帧目录
  - `--output`: 输出目录
  - `--mode`: 渲染模式（360, compare, 3d）

#### `render.py`
PyTorch渲染器
- 特点：GPU加速，支持交互式查看
- 参数：
  - `--gaussians`: 高斯文件
  - `--frames`: 帧目录
  - `--output`: 输出目录
  - `--mode`: 渲染模式（360, compare, 3d, interactive）

## 使用方法

### 方法1：使用主入口脚本（推荐）

```bash
# 完整流程（一步到位）
python main.py full --video input/videos/IMG_7834.MOV --train-mode simple

# 分步执行
# 1. 提取帧
python main.py extract --video input/videos/IMG_7834.MOV

# 2. 初始化高斯
python main.py initialize --frames input/frames

# 3. 训练
python main.py train --mode simple --iterations 100

# 4. 渲染
python main.py render --mode simple --render-mode 360
```

### 方法2：直接运行各模块脚本

```bash
# 1. 提取帧
python preprocess/extract_frames.py --video input/videos/IMG_7834.MOV --output input/frames

# 2. 初始化高斯
python preprocess/initialize.py --frames input/frames --output output

# 3. 训练（选择合适的训练脚本）
python train/train_simple.py --frames input/frames --init output/gaussians_init.pkl --output output --iterations 100

# 4. 渲染
python render/render_simple.py --gaussians output/gaussians_trained.pkl --frames input/frames --output output/renders --mode 360
```

## 训练模式对比

| 模式 | 脚本 | 特点 | 适用场景 |
|------|------|------|----------|
| simple | `train_simple.py` | NumPy，快速，优化版 | M1/M2芯片，快速原型 |
| torch | `train_torch.py` | PyTorch，详细注释 | 学习原理，教学演示 |
| 3dgs | `train_3dgs.py` | PyTorch，GPU加速 | 有GPU，追求质量 |
| full | `train.py` | NumPy，结构化 | 理解优化流程 |

## 渲染模式

- `360`: 渲染360度旋转视频帧
- `compare`: 与原图对比
- `3d`: 3D高斯分布可视化
- `interactive`: 交互式查看（仅PyTorch版本）

## 依赖项

```bash
pip install numpy opencv-python matplotlib torch scipy tqdm
```

## 快速开始

```bash
# 1. 将视频放入 input/videos/
# 2. 运行完整流程
python main.py full --video input/videos/your_video.MOV --train-mode simple

# 3. 查看结果
# - output/comparison_*.png: 训练过程对比
# - output/loss_curve.png: 损失曲线
# - output/renders/: 渲染结果
```

## 注意事项

1. **训练模式选择**：
   - 在M1/M2芯片上推荐使用 `train_simple.py`
   - 有GPU时推荐使用 `train_3dgs.py`
   - 想学习原理查看 `train_torch.py`

2. **参数调整**：
   - 高斯数量：`num_gaussians`（默认50000）
   - 训练迭代：`iterations`（默认100-500）
   - 学习率：`lr`（默认0.01）

3. **输出文件**：
   - `gaussians_init.pkl`: 初始化的高斯
   - `gaussians_trained.pkl`: 训练后的高斯
   - `gaussians_checkpoint_*.pkl`: 训练检查点

## 工作流程

```
视频 → 提取帧 → SfM初始化 → 训练优化 → 渲染可视化
  ↓         ↓          ↓          ↓          ↓
videos   frames   gaussians  trained    renders
```

## 常见问题

**Q: 训练很慢怎么办？**
A: 使用 `train_simple.py`，减少高斯数量（`--max-gaussians 2000`）

**Q: 如何提高重建质量？**
A: 增加训练迭代次数（`--iterations 500`），使用 `train_3dgs.py`

**Q: 渲染结果不理想？**
A: 检查帧数量和质量，调整相机参数，增加训练时间
