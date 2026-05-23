# Qwen-VL 多模态模型

集成 Qwen2.5-VL-3B-Instruct 多模态模型的项目，支持图像和视频的深度分析。

## 功能特性

- **图像分析**: 理解图像内容、描述场景
- **文本交互**: 基于图像进行问答对话
- **视频理解**: 分析视频内容、提取关键信息
- **模型下载**: 自动下载 HuggingFace 模型

## 项目文件

```
qwenvl_project/
├── frames/                    # 视频帧图像
├── inputs/                   # 输入视频
├── download_model_ali.py    # 阿里云模型下载
├── download_model_hf.py    # HuggingFace 模型下载
├── login_hf.py             # HuggingFace 登录
├── quick_detect.py         # 快速检测
├── run_image_analysis.py   # 图像分析
├── run_text_interaction.py # 文本交互
├── run_video_action.py     # 视频动作分析
├── run_video_analysis.py   # 视频分析
├── run_video_analysis_fast.py  # 快速视频分析
├── verify_model.py         # 模型验证
└── video_action_detector.py # 视频动作检测
```

## 使用方法

### 1. 安装依赖

```bash
pip install torch torchvision transformers qwen-vl-utils
```

### 2. 下载模型

```bash
# 使用阿里云下载
python download_model_ali.py

# 或使用 HuggingFace
python download_model_hf.py
```

### 3. 图像分析

```bash
python run_image_analysis.py --image path/to/image.jpg
```

### 4. 文本交互

```bash
python run_text_interaction.py --image path/to/image.jpg --question "描述这张图片"
```

### 5. 视频分析

```bash
python run_video_analysis.py --video path/to/video.mp4
```

## 模型说明

**Qwen2.5-VL-3B-Instruct**

- 参数量: 3B
- 能力: 图像理解、视频理解、文本生成
- 支持: 中文、英文等多语言

## 应用场景

- 视觉问答系统
- 图像描述生成
- 视频内容理解
- 文档图像分析
- 多模态对话系统

## 参考资料

- [Qwen2.5-VL](https://arxiv.org/abs/2407.11382)
- [HuggingFace Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [Qwen 官方文档](https://qwenlm.github.io/)

---

*最后更新: 2026-05-13*

## 环境与运行建议

- Python 3.9+（建议 3.10/3.11）
- Apple Silicon 可优先使用 `mps`，NVIDIA 环境可使用 `cuda`
- 首次运行建议先执行 `verify_model.py` 检查模型与依赖状态

## 最小可跑流程（建议）

```bash
pip install torch torchvision transformers qwen-vl-utils
python download_model_hf.py
python verify_model.py
python run_image_analysis.py --image path/to/image.jpg
```

## 常见问题

- 模型下载慢：优先使用 `download_model_ali.py` 或配置镜像
- 显存/内存不足：降低输入分辨率，优先跑单图分析而非长视频
- 视频分析耗时长：使用 `run_video_analysis_fast.py` 先做快速验证
