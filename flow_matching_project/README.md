# Flow Matching 教学项目

一个用于学习和理解 **Flow Matching** 生成模型的完整教学项目。每一行代码都包含详细的中文注释，适合初学者深入理解 Flow Matching 的原理和实现。

---

## 📚 Flow Matching 简介

### 什么是 Flow Matching？

Flow Matching 是一种新兴的生成模型技术，通过学习一个**连续的流（flow）**来将简单分布（如高斯噪声）转换为复杂的数据分布。

### 核心思想

```
噪声分布 (t=0)  ──────流──────>  数据分布 (t=1)
    x₀ ~ N(0,I)    v(x,t)      x₁ ~ p_data
```

1. **定义概率路径**：从噪声到数据的连续路径 `x(t) = (1-t)·x₀ + t·x₁`
2. **学习向量场**：神经网络预测每个点在每个时刻的移动方向 `v(x, t)`
3. **生成样本**：通过求解 ODE `dx/dt = v(x, t)` 从噪声生成数据

### 与扩散模型的对比

| 特性 | 扩散模型 (DDPM) | Flow Matching |
|------|----------------|---------------|
| 前向过程 | 逐步添加噪声 | 线性插值路径 |
| 反向过程 | 学习去噪 | 学习向量场 |
| 生成方式 | 多步迭代去噪 | 求解 ODE |
| 训练目标 | 预测噪声 | 预测速度场 |
| 灵活性 | 固定步数 | 可调节步数 |

---

## 🗂️ 项目结构

```
flow_matching_project/
├── config.py       # 配置文件（所有超参数和路径设置）
├── net.py          # 网络模型（向量场网络、时间嵌入）
├── train.py        # 训练脚本（完整训练流程）
├── infer.py        # 推理脚本（生成新样本）
├── visualize.py    # 可视化脚本（多种可视化功能）
├── README.md       # 项目说明（本文件）
└── output/         # 输出目录
    ├── models/     # 保存的模型检查点
    ├── samples/    # 生成的样本图像
    └── logs/       # 训练日志和可视化
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install torch torchvision matplotlib numpy tqdm
```

### 2. 训练模型

```bash
# 基础训练（50 个 epoch）
python train.py

# 从检查点恢复训练
python train.py --resume output/models/checkpoint_epoch_010.pt
```

### 3. 生成样本

```bash
# 使用训练好的模型生成样本
python infer.py --model output/models/model_final.pt

# 指定生成参数
python infer.py --model output/models/model_final.pt \
                --num-samples 100 \
                --num-steps 200 \
                --method rk4 \
                --visualize-process
```

### 4. 可视化

```bash
# 执行所有可视化
python visualize.py --model output/models/model_final.pt

# 特定可视化
python visualize.py --model output/models/model_final.pt --type vector_field
python visualize.py --model output/models/model_final.pt --type trajectory
python visualize.py --model output/models/model_final.pt --type compare_steps
```

---

## 📖 代码详解

### config.py - 配置文件

包含所有可配置参数，每个参数都有详细注释：

```python
# 数据配置
IMAGE_SIZE = 28        # MNIST 图像尺寸
DATA_DIM = 784         # 展平后的维度 (28*28)

# 网络配置
HIDDEN_DIM = 256       # 隐藏层维度
NUM_HIDDEN_LAYERS = 4  # 隐藏层数量
TIME_EMBED_DIM = 64    # 时间嵌入维度

# 训练配置
BATCH_SIZE = 128       # 批次大小
LEARNING_RATE = 1e-3   # 学习率
NUM_EPOCHS = 50        # 训练轮数
```

### net.py - 网络模型

#### 1. 时间嵌入 (SinusoidalTimeEmbedding)

将标量时间 `t ∈ [0, 1]` 编码为高维向量：

```python
embed[2i]   = sin(t * 10000^(2i/d))
embed[2i+1] = cos(t * 10000^(2i/d))
```

类似于 Transformer 的位置编码，让网络理解"时间"概念。

#### 2. 向量场网络 (TimeAwareMLP)

预测向量场 `v(x, t)`：

```
输入: [x, time_embed(t)]  (拼接)
  ↓
Linear → Activation → ... (多层)
  ↓
输出: v(x, t)  (向量场)
```

#### 3. Flow Matching 模型 (FlowMatchingModel)

核心方法：

- `compute_loss(x_1)`: 计算训练损失
- `sample(num_samples, num_steps)`: 生成样本

### train.py - 训练脚本

训练流程：

```python
for epoch in range(NUM_EPOCHS):
    for batch in dataloader:
        x_1 = batch  # 真实数据
        
        # 1. 采样噪声和时间
        x_0 = randn_like(x_1)
        t = uniform(0, 1)
        
        # 2. 计算插值点
        x_t = (1-t) * x_0 + t * x_1
        
        # 3. 计算损失
        v_target = x_1 - x_0
        v_pred = model(x_t, t)
        loss = MSE(v_pred, v_target)
        
        # 4. 反向传播
        loss.backward()
        optimizer.step()
```

### infer.py - 推理脚本

生成过程（ODE 求解）：

```python
# 欧拉方法
x = randn(num_samples, data_dim)  # 初始噪声
dt = 1.0 / num_steps

for step in range(num_steps):
    t = step / num_steps
    v = model(x, t)  # 预测向量场
    x = x + dt * v   # 更新位置

return x  # 生成样本
```

---

## 🎓 数学原理

### 训练目标

Flow Matching 的训练目标是学习向量场 `v_θ(x, t)`，使得它能将噪声分布变换为数据分布。

**损失函数**：

```
L = E_{t, x_0, x_1} [ ||v_θ(x_t, t) - (x_1 - x_0)||² ]
```

其中：
- `x_0 ~ N(0, I)` 是随机噪声
- `x_1` 是真实数据样本
- `x_t = (1-t)·x_0 + t·x_1` 是插值路径上的点
- `x_1 - x_0` 是目标向量场（从噪声到数据的方向）

### 生成过程

从噪声 `x_0 ~ N(0, I)` 开始，求解 ODE：

```
dx/dt = v_θ(x, t),  x(0) = x_0
```

最终得到的 `x(1)` 就是从数据分布中采样的结果。

### ODE 求解方法

#### 欧拉方法（简单快速）

```
x(t + dt) = x(t) + dt · v(x(t), t)
```

#### 四阶龙格-库塔（更精确）

```
k1 = v(x, t)
k2 = v(x + 0.5·dt·k1, t + 0.5·dt)
k3 = v(x + 0.5·dt·k2, t + 0.5·dt)
k4 = v(x + dt·k3, t + dt)
x(t + dt) = x(t) + dt/6 · (k1 + 2·k2 + 2·k3 + k4)
```

---

## 📊 可视化功能

### 1. 向量场可视化

展示学习到的向量场在不同时间点的形态：

```bash
python visualize.py --model output/models/model_final.pt --type vector_field
```

### 2. 生成轨迹可视化

展示从噪声到生成样本的完整路径：

```bash
python visualize.py --model output/models/model_final.pt --type trajectory
```

### 3. 不同步数对比

对比不同 ODE 步数对生成质量的影响：

```bash
python visualize.py --model output/models/model_final.pt --type compare_steps
```

### 4. 不同方法对比

对比欧拉方法和 RK4 方法的生成结果：

```bash
python visualize.py --model output/models/model_final.pt --type compare_methods
```

### 5. 时间嵌入可视化

展示时间嵌入的结构：

```bash
python visualize.py --model output/models/model_final.pt --type time_embedding
```

---

## ⚙️ 参数调优

### 提高生成质量

| 参数 | 建议 | 说明 |
|------|------|------|
| `NUM_TIMESTEPS` | 100-500 | 更多步数 = 更精确的 ODE 求解 |
| `INTEGRATION_METHOD` | 'rk4' | RK4 比 Euler 更精确 |
| `HIDDEN_DIM` | 256-512 | 更大的网络 = 更强的表达能力 |
| `NUM_EPOCHS` | 50-100 | 更多训练 = 更好的收敛 |

### 加速生成

| 参数 | 建议 | 说明 |
|------|------|------|
| `NUM_TIMESTEPS` | 20-50 | 更少步数 = 更快的生成 |
| `INTEGRATION_METHOD` | 'euler' | Euler 比 RK4 更快 |

### 减少内存使用

| 参数 | 建议 | 说明 |
|------|------|------|
| `BATCH_SIZE` | 32-64 | 更小的批次 = 更少的内存 |
| `HIDDEN_DIM` | 128-256 | 更小的网络 = 更少的参数 |
| `NUM_SAMPLES` | 16-32 | 生成更少的样本 |

---

## 🔬 实验建议

### 实验 1：理解 Flow Matching

1. 训练模型并观察损失下降
2. 可视化向量场，理解流的形态
3. 可视化生成轨迹，理解从噪声到数据的过程

### 实验 2：ODE 步数的影响

1. 使用不同步数生成样本（10, 50, 100, 200）
2. 对比生成质量和速度
3. 找到质量与速度的平衡点

### 实验 3：积分方法的影响

1. 对比 Euler 和 RK4 方法
2. 观察相同步数下的质量差异
3. 理解数值精度的重要性

### 实验 4：网络架构的影响

1. 尝试不同的隐藏层维度（128, 256, 512）
2. 尝试不同的网络深度（2, 4, 6 层）
3. 观察模型容量对生成质量的影响

---

## 📚 参考资料

### 论文

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) - Lipman et al., ICLR 2023
- [Rectified Flow](https://arxiv.org/abs/2209.03003) - Liu et al.
- [Flow Straight and Fast](https://arxiv.org/abs/2305.08622) - Huang et al.

### 教程

- [Flow Matching: A Tutorial](https://mlg.eng.cam.ac.uk/blog/2023/01/20/flow-matching.html)
- [Understanding Flow Matching](https://sander.ai/2023/07/20/flow_matching.html)

### 相关项目

- [torchcfm](https://github.com/atong01/conditional-flow-matching) - 官方 Flow Matching 实现
- [rectified-flow](https://github.com/gnobitab/Rectified-flow-pytorch) - Rectified Flow 实现

---

## ❓ 常见问题

### Q1: 为什么我的生成结果很模糊？

**A**: 可能的原因：
- 训练不够充分，增加 `NUM_EPOCHS`
- ODE 求解不够精确，增加 `NUM_TIMESTEPS` 或使用 'rk4' 方法
- 网络容量不足，增加 `HIDDEN_DIM`

### Q2: 训练损失不下降怎么办？

**A**: 检查：
- 学习率是否合适（尝试 1e-4 到 1e-3）
- 数据是否正确加载
- 模型是否正确初始化

### Q3: 生成速度太慢怎么办？

**A**: 优化方法：
- 减少 `NUM_TIMESTEPS`（如 20-50）
- 使用 'euler' 方法代替 'rk4'
- 使用 GPU 加速

### Q4: 如何在自己的数据集上训练？

**A**: 修改 `train.py` 中的数据加载部分：

```python
# 替换 MNIST 为你的数据集
dataset = YourCustomDataset(root='path/to/data', transform=transform)
```

---

## 🤝 贡献

这是一个教学项目，欢迎：
- 提出问题和建议
- 报告 bug
- 提交改进

---

## 📝 许可证

本项目仅供学习和教学使用。

---

*最后更新: 2026-05-22*
