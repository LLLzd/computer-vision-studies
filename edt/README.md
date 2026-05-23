# EDT 距离变换实验

该项目用于演示从 RGB 图像边缘提取到欧几里得距离变换（EDT）的完整流程，适合学习图像预处理与距离场构建。

## 项目结构

```text
edt/
├── edge_line.py      # 多种边缘检测方法（Canny/Sobel/Laplacian 等）
├── edt.py            # EDT 算法实现（BFS、两遍法、SciPy 对比等）
├── rgb_to_edt.py     # RGB -> 边缘 -> EDT 的端到端流程
├── input/
└── output/
```

## 环境要求

- Python 3.9+
- numpy
- opencv-python
- matplotlib

## 快速开始

```bash
cd edt
python rgb_to_edt.py -i input/your_image.png -o output/
```

批量处理：

```bash
python rgb_to_edt.py -b -o output/
```

## 输出结果

- `edges.png`：边缘检测结果
- `edt_output.png`：灰度 EDT 结果
- `edt_colored.png`：伪彩色 EDT 结果

## 适用场景

- 距离场可视化
- 骨架提取前处理
- 路径规划中的代价场构建
- 目标检测/分割中的几何辅助特征
