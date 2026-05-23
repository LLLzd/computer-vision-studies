# 里程计轨迹平滑优化实验

该项目通过二维轨迹示例演示：在固定关键帧锚点条件下，对中间帧漂移轨迹进行全局平滑优化。

## 项目结构

```text
odom/
├── filter.py                    # 轨迹优化主脚本
├── trajectory_optimization.png  # 轨迹对比图（示例输出）
├── optimization_trend.png       # 误差收敛趋势图（示例输出）
└── optimization_process.gif      # 优化过程动画（示例输出）
```

## 环境要求

- Python 3.9+
- numpy
- matplotlib
- pillow（保存 gif 时常用）

## 快速开始

```bash
cd odom
python filter.py
```

## 输出说明

运行后会输出：

- 终点与中间帧优化前后误差对比
- 轨迹优化过程动画（`optimization_process.gif`）
- 若干关键帧误差收敛曲线（`optimization_trend.png`）

## 适用场景

- LiDAR/视觉里程计后处理教学
- 锚点约束下轨迹平滑思路验证
- 轨迹优化可视化展示
