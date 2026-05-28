# 计算机视觉学习项目集合

## 📁 项目简介

这是一个计算机视觉学习项目集合，包含多个不同类型的计算机视觉任务实现，从基础的图像分类到更复杂的图像分割和超分辨率。所有项目都设计为适合在 M1 Mac 上运行，代码结构清晰，易于理解和扩展。

## 🗂️ 当前目录总览（已同步）

以下为 `study/` 目录下当前可见子项目（按方向分组）：

### 基础视觉任务

- [MNIST 手写数字识别](mnist_project/)：CNN 分类入门（训练/测试/推理/可视化）
- [CIFAR-10 图像分类](cifar10_project/)：10 类彩色图像分类
- [MNIST 图像分割](mnist_segmentation_project/)：U-Net 像素级分割
- [Oxford-IIIT Pet 图像分割](oxford_pet_project/)：宠物前景分割
- [Pascal VOC 2012 图像分割](pascal_vco2012_project/)：21 类语义分割
- [VOC2012 Anchor-Free 检测](vco2012_od2d/)：热力图式 2D 目标检测

### 生成与增强

- [DIV2K 超分辨率](sr_div2k/)：ESPCN/EDSR 超分辨率实验
- [Fast Style Transfer](fast-style-transfer/)：风格迁移训练与推理
- [VAE 变分自编码器](vae_project/)：MNIST 重构/生成/隐空间可视化，双模式训练
- [Flow Matching 生成模型](flow_matching_project/)：流匹配生成模型教学（详细注释）

### 多模态与大模型

- [Qwen-VL 多模态模型](qwenvl_project/)：图像/视频理解与问答
- [人脸偏好打分排序工具](face_ranking_project/)：本地离线 Elo 偏好排序

### 自动驾驶与 BEV

- [NuScenes 可视化 Demo](demo_by_nuscenes/)：环视 + BEV 可视化与统计
- [NuScenes 本地网页平台](nuscenes_web_platform/)：FastAPI + 前端交互浏览
- [BEV Box 插值评测](bev_box_interp_eval/)：关键帧补帧算法评测与可视化
- [Occupancy Grid 学习实验](occupancy/)：贝叶斯 OGM 与神经网络建图示例
- [里程计轨迹优化](odom/)：关键帧锚点约束下轨迹平滑

### 3D 重建 / 3DGS

- [3D Gaussian Splatting 学习版](3dgs/)：3DGS 流程拆解与训练渲染
- [gsplat M1 重建流程](gsplat_m1_recon/)：Apple Silicon 友好的一体化重建管线

### 图像处理基础实验

- [EDT 距离变换实验](edt/)：边缘提取 + 欧几里得距离变换

## 📋 项目列表

### 1. [MNIST 手写数字识别](mnist_project/)
- **任务**: 手写数字分类
- **模型**: 卷积神经网络 (CNN)
- **数据集**: MNIST
- **功能**: 训练、测试、推理、数据可视化
- **论文参考**: [LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
- **项目介绍**: 实现了一个基础的卷积神经网络，用于识别 MNIST 手写数字数据集。包含完整的训练、测试和推理流程，适合作为深度学习入门项目。

### 2. [CIFAR-10 图像分类](cifar10_project/)
- **任务**: 10类图像分类
- **模型**: 卷积神经网络 (CNN) 带批归一化
- **数据集**: CIFAR-10
- **功能**: 训练、测试、推理、数据可视化
- **论文参考**: [Krizhevsky, A., Hinton, G. E., & others. (2009). Learning multiple layers of features from tiny images.](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
- **项目介绍**: 实现了一个带有批归一化的卷积神经网络，用于对 CIFAR-10 数据集进行分类。CIFAR-10 包含 10 个类别的 60000 张 32x32 彩色图像，是图像分类的经典数据集。

### 3. [MNIST 图像分割](mnist_segmentation_project/)
- **任务**: 手写数字分割
- **模型**: U-Net
- **数据集**: MNIST
- **功能**: 训练、测试、推理、分割可视化
- **论文参考**: [Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 234-241). Springer, Cham.](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28)
- **项目介绍**: 使用 U-Net 模型对 MNIST 手写数字进行分割。将分类任务转换为分割任务，通过生成像素级的分割掩码来标识数字区域。

### 4. [Oxford-IIIT Pet 图像分割](oxford_pet_project/)
- **任务**: 宠物图像分割
- **模型**: U-Net
- **数据集**: Oxford-IIIT Pet
- **功能**: 训练、测试、推理、数据可视化
- **论文参考**: [Parkhi, O. M., Vedaldi, A., Zisserman, A., & Jawahar, C. V. (2012). Cats and dogs. In 2012 IEEE Conference on Computer Vision and Pattern Recognition (pp. 3498-3505).](https://ieeexplore.ieee.org/document/6248091)
- **项目介绍**: 使用 U-Net 模型对 Oxford-IIIT Pet 数据集进行图像分割。该数据集包含 37 个类别的宠物图像，每个图像都有对应的分割标注，用于训练模型识别宠物的轮廓。

### 5. [Pascal VOC 2012 图像分割](pascal_vco2012_project/)
- **任务**: 语义分割
- **模型**: U-Net
- **数据集**: Pascal VOC 2012
- **功能**: 训练、测试、推理、数据可视化
- **论文参考**: [Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J., & Zisserman, A. (2010). The pascal visual object classes (voc) challenge. International journal of computer vision, 88(2), 303-338.](https://link.springer.com/article/10.1007/s11263-009-0275-4)
- **项目介绍**: 使用 U-Net 模型对 Pascal VOC 2012 数据集进行语义分割。该数据集包含 20 个类别的物体，模型需要对图像中的每个像素进行分类，实现像素级的语义分割。

### 6. [Qwen-VL 多模态模型](qwenvl_project/)
- **任务**: 图像分析与文本交互
- **模型**: Qwen2.5-VL-3B-Instruct
- **功能**: 模型下载、图像分析、文本交互
- **论文参考**: [Qwen Team. (2024). Qwen2.5: Improved Baselines for Multimodal Understanding and Generation.](https://arxiv.org/abs/2407.11382)
- **项目介绍**: 集成了 Qwen2.5-VL-3B-Instruct 多模态模型，支持图像分析和文本交互。包含模型下载脚本和运行示例，可用于图像描述、视觉问答等任务。

### 7. [DIV2K 超分辨率](sr_div2k/)
- **任务**: 图像超分辨率
- **模型**: ESPCN、EDSR
- **数据集**: DIV2K
- **功能**: 训练、评估、推理、数据可视化
- **论文参考**:
  - ESPCN: [Shi, W., Caballero, J., Huszár, F., Totz, J., Aitken, A. P., Bishop, R., ... & Wang, Z. (2016). Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1874-1883).](https://arxiv.org/abs/1609.05158)
  - EDSR: [Lim, B., Son, S., Kim, H., Nah, S., & Lee, K. M. (2017). Enhanced deep residual networks for single image super-resolution. In Proceedings of the IEEE conference on computer vision and pattern recognition workshops (pp. 136-144).](https://arxiv.org/abs/1707.02921)
- **项目介绍**: 实现了 ESPCN 和 EDSR 两种超分辨率模型，使用 DIV2K 数据集进行训练和评估。支持 2x、3x、4x 超分倍数，可通过配置文件轻松切换模型和超分倍数。

### 8. [人脸偏好打分排序](face_ranking_project/)
- **任务**: 人脸图片偏好排序
- **核心方法**: Elo 评分 + 成对比较
- **功能**: Web 界面对比、结果导出 CSV、评分重置
- **项目介绍**: 本地离线运行，图片不上传网络，适合快速做审美偏好排序实验。

### 9. [NuScenes 可视化 Demo](demo_by_nuscenes/)
- **任务**: 自动驾驶多传感器数据可视化
- **数据集**: NuScenes（建议先用 `v1.0-mini`）
- **功能**: 环视相机展示、BEV 标注展示、GIF/视频导出

### 10. [NuScenes 本地网页平台](nuscenes_web_platform/)
- **任务**: NuScenes 数据浏览平台
- **技术栈**: FastAPI + 原生前端
- **功能**: 场景检索、逐帧查看、相机图层叠加、LiDAR BEV 渲染

### 11. [BEV Box 插值评测](bev_box_interp_eval/)
- **任务**: 关键帧到中间帧的 2D Box 递推评测
- **方法**: Linear / Poly / Kalman / Spline
- **功能**: 指标评测、可视化视频、对比图与评测报告输出

### 12. [3D Gaussian Splatting 学习版](3dgs/)
- **任务**: 3DGS 场景重建与渲染学习
- **功能**: 视频抽帧、初始化、训练、渲染、效果对比

### 13. [gsplat M1 重建流程](gsplat_m1_recon/)
- **任务**: 面向 Apple Silicon 的 3D 小场景重建
- **功能**: 一键流水线、COLMAP 位姿估计、MPS 训练与模型导出

### 14. [Fast Style Transfer](fast-style-transfer/)
- **任务**: 图像风格迁移
- **功能**: 预训练模型推理、自定义风格训练、结果可视化

### 15. [EDT 距离变换实验](edt/)
- **任务**: 边缘到距离场的图像处理实验
- **方法**: Canny/Sobel/Laplacian + EDT（BFS/两遍法）
- **功能**: 单图/批处理、结果可视化、距离场导出

### 16. [Occupancy Grid 学习实验](occupancy/)
- **任务**: 占用栅格估计与建图
- **方法**: 贝叶斯 OGM + 简化神经网络估计
- **功能**: 概率栅格更新、传感器模拟、可视化演示

### 17. [里程计轨迹优化实验](odom/)
- **任务**: 轨迹平滑与漂移校正
- **方法**: 关键帧锚点约束 + 全局平滑优化
- **功能**: 误差统计、优化趋势图、过程动画

### 18. [VOC2012 Anchor-Free 2D 检测](vco2012_od2d/)
- **任务**: Pascal VOC 多类别目标检测
- **方法**: Heatmap + Offset + WH（Anchor-Free）
- **功能**: 训练、评测（mAP/PR）、推理与可视化

### 19. [VAE 变分自编码器](vae_project/)
- **任务**: 生成建模与图像重构
- **模型**: Variational Autoencoder (VAE)
- **数据集**: MNIST（可切换 FashionMNIST/CelebA）
- **功能**: 快速/标准训练、ELBO 训练、重构可视化、随机生成、2D 隐空间 t-SNE
- **论文参考**: [Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes.](https://arxiv.org/abs/1312.6114)
- **项目介绍**: 提供标准科研工程化实现，支持配置化训练、checkpoint 管理、日志记录与可视化结果导出，便于课程学习和实验复现。

## 🚀 快速开始

### 环境配置

```bash
# 安装基础依赖
pip install torch torchvision pillow numpy tqdm matplotlib

# 安装额外依赖（根据项目需要）
pip install scikit-image huggingface_hub
```

### 运行项目

每个项目都有独立的 README.md 文件，详细说明了如何运行。以下是通用步骤：

1. **进入项目目录**
   ```bash
   cd <project_directory>
   ```

2. **下载数据集**（如果需要）
   ```bash
   python download_data.py  # 或类似脚本
   ```

3. **训练模型**
   ```bash
   python train.py
   ```

4. **测试模型**
   ```bash
   python test.py
   ```

5. **推理测试**
   ```bash
   python infer.py
   ```

## 📁 项目结构

所有项目都采用类似的结构：

```
<project_name>/
├── config.py          # 配置文件（超参数、路径等）
├── net.py             # 模型定义
├── train.py           # 训练脚本
├── test.py            # 测试脚本
├── infer.py           # 推理脚本
├── visual.py          # 可视化脚本
├── download.py        # 数据集下载脚本（如果需要）
├── data/              # 数据集目录
├── models/            # 模型保存目录
└── outputs/           # 输出目录
```

### 各项目详细结构

- **[MNIST 手写数字识别](mnist_project/)**：基础分类任务，包含完整的训练和推理流程
- **[CIFAR-10 图像分类](cifar10_project/)**：多类别分类任务，使用批归一化提高模型性能
- **[MNIST 图像分割](mnist_segmentation_project/)**：基础分割任务，将分类转换为分割
- **[Oxford-IIIT Pet 图像分割](oxford_pet_project/)**：宠物图像分割，使用 U-Net 模型
- **[Pascal VOC 2012 图像分割](pascal_vco2012_project/)**：21 类语义分割训练与推理
- **[Qwen-VL 多模态模型](qwenvl_project/)**：多模态任务，支持图像分析和文本交互
- **[DIV2K 超分辨率](sr_div2k/)**：超分辨率任务，实现 ESPCN 和 EDSR 模型
- **[人脸偏好打分排序](face_ranking_project/)**：基于 Elo 的本地离线偏好排序工具
- **[NuScenes 可视化 Demo](demo_by_nuscenes/)**：自动驾驶数据集静态可视化与导出
- **[NuScenes 本地网页平台](nuscenes_web_platform/)**：NuScenes 浏览、检索和渲染平台
- **[BEV Box 插值评测](bev_box_interp_eval/)**：关键帧补帧算法评测与可视化
- **[Occupancy Grid 学习实验](occupancy/)**：贝叶斯 OGM 与学习式占用估计
- **[里程计轨迹优化实验](odom/)**：关键帧锚点约束下的轨迹平滑
- **[3DGS 学习版](3dgs/)**：3D Gaussian Splatting 基础流程实现
- **[gsplat M1 重建流程](gsplat_m1_recon/)**：Apple Silicon 友好的 3DGS 小场景重建
- **[Fast Style Transfer](fast-style-transfer/)**：风格迁移训练与推理
- **[EDT 距离变换实验](edt/)**：边缘提取与欧几里得距离变换
- **[VOC2012 Anchor-Free 2D 检测](vco2012_od2d/)**：Anchor-Free 目标检测训练评测

## 🔧 技术栈

- **框架**: PyTorch
- **图像处理**: Pillow, OpenCV
- **数据处理**: NumPy
- **可视化**: Matplotlib
- **评估指标**: scikit-image
- **模型下载**: Hugging Face Hub

## 📚 学习资源

- **PyTorch 官方文档**: [https://pytorch.org/docs/stable/](https://pytorch.org/docs/stable/)
- **计算机视觉课程**: Stanford CS231n
- **深度学习课程**: fast.ai

## 🎯 学习路径

1. **基础任务**: 从 MNIST 和 CIFAR-10 分类开始
2. **进阶任务**: 尝试图像分割项目
3. **高级任务**: 探索超分辨率和多模态模型

## 📝 注意事项

- **数据集大小**: 部分数据集（如 Pascal VOC 2012）较大，请确保有足够的磁盘空间
- **训练时间**: 复杂模型可能需要较长时间训练，请耐心等待
- **硬件要求**: 建议使用 GPU 加速训练，特别是对于较大的模型
- **模型下载**: Qwen-VL 模型较大，需要稳定的网络连接

## 🧾 一级子项目全量清单

`study/` 当前一级子项目目录（共 18 个）：

- `3dgs/`
- `bev_box_interp_eval/`
- `cifar10_project/`
- `demo_by_nuscenes/`
- `edt/`
- `face_ranking_project/`
- `fast-style-transfer/`
- `gsplat_m1_recon/`
- `mnist_project/`
- `mnist_segmentation_project/`
- `nuscenes_web_platform/`
- `occupancy/`
- `odom/`
- `oxford_pet_project/`
- `pascal_vco2012_project/`
- `qwenvl_project/`
- `sr_div2k/`
- `vco2012_od2d/`

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这些项目！

## 📄 许可证

本项目采用 MIT 许可证。

## 📧 联系方式

如有问题，请提交 Issue 或联系项目维护者。
