# Fast Style Transfer

基于PyTorch实现的快速风格迁移项目，支持使用预训练模型进行风格迁移，也可以训练自己的风格模型。

## 项目结构

```
fast-style-transfer/
├── images/                # 图像目录
│   ├── content/           # 内容图像
│   └── style/             # 风格图像
├── models/                # 模型目录
├── outputs/               # 输出目录
├── config.yaml            # 配置文件
├── config.py              # 配置加载器
├── dataset.py             # 数据集处理
├── data_visual.py         # 数据可视化
├── infer.py               # 快速推理脚本
├── main.py                # 主脚本（训练和迁移）
├── net.py                 # 网络结构
└── README.md              # 项目说明
```

## 快速开始

### 1. 下载预训练模型

直接下载这4个预训练模型，然后放入`models/`文件夹：

- **candy.pth** - 梵高风格
- **mosaic.pth** - 马赛克风格
- **rain_princess.pth** - 油画公主风格
- **udnie.pth** - 毕加索风格

下载链接：[PyTorch官方示例模型](https://github.com/pytorch/examples/tree/master/fast_neural_style/models)

### 2. 准备内容图像

将你要进行风格迁移的照片放入`images/content/`文件夹，例如：
- `images/content/your_photo.jpg`
- `images/content/cat.jpg`
- `images/content/landscape.jpg`

### 3. 配置文件

修改`config.yaml`文件中的配置：

```yaml
# 数据配置
DATA_DIR: "images"
CONTENT_DIR: "images/content"
STYLE_DIR: "images/style"
OUTPUT_DIR: "outputs"
MODEL_DIR: "models"

# 图像配置
IMAGE_SIZE: 256  # 训练图像尺寸
TRANSFER_SIZE: 512  # 风格迁移图像尺寸

# 推理配置
DEFAULT_CONTENT_IMAGE: "images/content/your_photo.jpg"
DEFAULT_STYLE_MODEL: "models/candy.pth"
DEFAULT_OUTPUT_IMAGE: "outputs/output_styled.jpg"

# 设备配置
DEVICE: "auto"  # auto, cpu, cuda, mps
```

### 4. 运行风格迁移

#### 使用main.py（推荐）

```bash
# 使用默认配置
python main.py transfer

# 指定参数
python main.py transfer --content images/content/cat.jpg --model models/candy.pth --output outputs/cat_styled.jpg
```

#### 使用infer.py（快速推理）

```bash
python infer.py
```

### 5. 训练自己的模型

```bash
# 使用风格图像训练模型
python main.py train --style images/style/starry_night.jpg
```

### 6. 可视化结果

```bash
python main.py visualize
```

## 技术说明

- **网络结构**：使用TransformerNetwork作为风格迁移网络，VGG19作为特征提取网络
- **损失函数**：内容损失 + 风格损失 + 总变分损失
- **设备支持**：自动检测并使用MPS（Apple Silicon）、CUDA或CPU
- **模块化设计**：网络结构、数据处理、配置管理分离

## 依赖

- PyTorch 1.8+
- torchvision
- Pillow
- NumPy
- Matplotlib
- tqdm

## 安装依赖

```bash
pip install torch torchvision Pillow numpy matplotlib tqdm pyyaml
```

## 示例

### 输入输出示例

#### 内容图像：
![Content Image](images/content/your_photo.jpg)

#### 风格图像（梵高风格）：
![Style Image](images/style/starry_night.jpg)

#### 输出图像：
![Output Image](outputs/output_styled.jpg)

## 参考

- [PyTorch官方示例：Fast Neural Style](https://github.com/pytorch/examples/tree/master/fast_neural_style)
- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)
