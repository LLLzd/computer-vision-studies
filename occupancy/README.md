# Occupancy Grid 学习实验

该目录包含两类占用栅格建图示例：  
- 基于贝叶斯更新的经典 OGM（`ogm.py`）  
- 基于简单神经网络的 occupancy 估计（`occ.py`）

## 项目结构

```text
occupancy/
├── ogm.py   # 贝叶斯 Occupancy Grid Mapping 演示
└── occ.py   # 神经网络 Occupancy Grid Estimation 演示
```

## 环境要求

- Python 3.9+
- numpy
- matplotlib

## 快速开始

```bash
cd occupancy
python ogm.py
python occ.py
```

## 脚本说明

- `ogm.py`
  - 使用 log-odds 形式进行占用概率更新
  - 包含 Bresenham 射线经过栅格更新逻辑
- `occ.py`
  - 用简化网络从模拟激光扫描数据回归占用概率图
  - 展示了数据生成、训练与可视化全流程

## 适合用途

- 自动驾驶/机器人 SLAM 基础学习
- Occupancy 建图算法教学演示
- 传统方法与学习方法的对照入门
