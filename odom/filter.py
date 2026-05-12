import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==============================================
# 场景：12帧轨迹，2个关键帧锚点，中间10帧漂移
# ==============================================

# 1. 真值（锚点，绝对不动）
pose_gt_0 = np.array([0.0, 0.0])     # 第0帧：起点
pose_gt_11 = np.array([11.0, 0.0])  # 第11帧：终点（绝对真值）

# 2. 生成抛物线轨迹的真值
# 抛物线方程：y = ax² + bx + c
# 已知两点 (0,0) 和 (11,0)，顶点在中间 x=5.5, y=5
# 计算抛物线参数
a = -20 / (11**2)  # 顶点在(5.5,5)的抛物线
b = 220 / 121
c = 0

# 生成抛物线轨迹的真值
pose_gt = []
for i in range(12):
    x = i
    y = a * x**2 + b * x + c
    pose_gt.append(np.array([x, y]))
pose_gt = np.array(pose_gt)

# 3. 模拟LiDAR Odom递推结果：带漂移的轨迹
pose_drift = []
for i in range(12):
    if i == 0:
        pose_drift.append(pose_gt_0.copy())
    elif i == 11:
        pose_drift.append(pose_gt_11.copy())
    else:
        # 抛物线真值加上漂移
        x = i
        y_true = a * x**2 + b * x + c
        # 添加漂移：y方向向上飘，x方向也有轻微漂移
        y_drift = y_true + 0.3 * i  # 向上漂移
        x_drift = x + 0.1 * i  # 向右漂移
        pose_drift.append(np.array([x_drift, y_drift]))

pose_drift = np.array(pose_drift)

# ==============================================
# 核心：全局轨迹平滑优化（两个锚点固定）
# ==============================================
def smooth_whole_trajectory(poses, fixed_idx, fixed_poses, alpha=0.5, max_iter=200):
    """
    全局轨迹优化
    原理：让每一帧 = 靠近原始值 + 靠近前后帧（平滑）
    固定关键帧不动
    """
    optimized = poses.copy()
    optimization_history = [optimized.copy()]

    for _ in range(max_iter):
        # 固定锚点（绝对真值）
        for idx, pos in zip(fixed_idx, fixed_poses):
            optimized[idx] = pos

        # 遍历中间所有帧，全局平滑更新
        for i in range(1, 11):  # 只优化 1~10 帧
            prev = optimized[i-1]  # 前一帧
            next_p = optimized[i+1] # 后一帧
            curr = poses[i]         # 原始漂移值

            # 核心公式：全局平滑
            optimized[i] = (1 - alpha) * curr + alpha * (prev + next_p) / 2
        
        # 每10次迭代记录一次
        if _ % 10 == 0:
            optimization_history.append(optimized.copy())

    # 最后再记录一次
    optimization_history.append(optimized.copy())
    return optimized, optimization_history

# 执行优化
pose_opt, optimization_history = smooth_whole_trajectory(
    poses=pose_drift,
    fixed_idx=[0, 11],        # 两个锚点索引
    fixed_poses=[pose_gt_0, pose_gt_11],  # 两个锚点真值
    alpha=0.5,
    max_iter=200  # 增加迭代次数
)

# ==============================================
# 打印结果，看漂移是否被拉回
# ==============================================
print("=== 优化前（漂移）第11帧误差 ===")
print("预测值：", pose_drift[11])
print("真值：", pose_gt_11)
print("误差：", np.linalg.norm(pose_drift[11] - pose_gt_11))

print("\n=== 优化后第11帧 ===")
print("优化值：", pose_opt[11])
print("误差：", np.linalg.norm(pose_opt[11] - pose_gt_11))

print("\n=== 中间第5帧（漂移被拉回）===")
print("优化前：", pose_drift[5])
print("优化后：", pose_opt[5])
print("真值：", pose_gt[5])

# 计算中间帧的误差变化
print("\n=== 中间帧误差变化 ===")
for i in [2, 5, 8]:
    drift_error = np.linalg.norm(pose_drift[i] - pose_gt[i])
    opt_error = np.linalg.norm(pose_opt[i] - pose_gt[i])
    improvement = (drift_error - opt_error) / drift_error * 100
    print(f"第{i}帧: 漂移误差={drift_error:.3f}, 优化误差={opt_error:.3f}, 改进={improvement:.2f}%")

# ==============================================
# 可视化优化过程
# ==============================================
def plot_optimization_process():
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制真值轨迹
    ax.plot(pose_gt[:, 0], pose_gt[:, 1], 'g-', label='Ground Truth', linewidth=2)
    ax.scatter(pose_gt[:, 0], pose_gt[:, 1], c='g', s=50, label='Truth Points')
    
    # 绘制漂移轨迹
    ax.plot(pose_drift[:, 0], pose_drift[:, 1], 'r-', label='Drifted Trajectory', linewidth=2, alpha=0.5)
    ax.scatter(pose_drift[:, 0], pose_drift[:, 1], c='r', s=50, label='Drifted Points', alpha=0.5)
    
    # 标记锚点
    ax.scatter([pose_gt_0[0], pose_gt_11[0]], [pose_gt_0[1], pose_gt_11[1]], 
                c='yellow', s=100, edgecolors='black', label='Anchor Points')
    
    # 初始化优化轨迹
    opt_line, = ax.plot([], [], 'b-', label='Optimized Trajectory', linewidth=3)
    opt_points = ax.scatter([], [], c='b', s=60, label='Optimized Points')
    
    # 添加迭代次数文本
    iter_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, 
                        bbox=dict(facecolor='white', alpha=0.8))
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Trajectory Optimization Process')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')
    
    # 设置坐标轴范围
    all_x = np.concatenate([pose_gt[:, 0], pose_drift[:, 0], pose_opt[:, 0]])
    all_y = np.concatenate([pose_gt[:, 1], pose_drift[:, 1], pose_opt[:, 1]])
    ax.set_xlim(all_x.min() - 1, all_x.max() + 1)
    ax.set_ylim(all_y.min() - 1, all_y.max() + 1)
    
    def update(frame):
        current_opt = optimization_history[frame]
        opt_line.set_data(current_opt[:, 0], current_opt[:, 1])
        opt_points.set_offsets(current_opt)
        iter_text.set_text(f'Iteration: {frame * 10}')
        return opt_line, opt_points, iter_text
    
    # 创建动画
    anim = FuncAnimation(fig, update, frames=len(optimization_history), 
                       interval=300, blit=True)
    
    # 保存动画
    anim.save('optimization_process.gif', writer='pillow', fps=5)
    print("\n=== 优化过程可视化 ===")
    print("动画已保存为: optimization_process.gif")
    
    plt.show()

# ==============================================
# 绘制中间帧的优化趋势
# ==============================================
def plot_optimization_trend():
    # 选择中间几帧进行分析
    key_frames = [2, 5, 8]
    # 生成与优化历史匹配的迭代次数
    iterations = range(0, len(optimization_history) * 10, 10)[:len(optimization_history)]
    
    fig, axes = plt.subplots(len(key_frames), 1, figsize=(12, 12))
    
    for i, frame_idx in enumerate(key_frames):
        ax = axes[i]
        
        # 计算每轮迭代的误差
        errors = []
        for opt in optimization_history:
            error = np.linalg.norm(opt[frame_idx] - pose_gt[frame_idx])
            errors.append(error)
        
        # 绘制误差趋势
        ax.plot(iterations, errors, 'b-', linewidth=2, marker='o')
        ax.axhline(y=np.linalg.norm(pose_drift[frame_idx] - pose_gt[frame_idx]), 
                   color='r', linestyle='--', label='Drifted Error')
        ax.axhline(y=0, color='g', linestyle='--', label='Ground Truth')
        
        ax.set_title(f'Frame {frame_idx} Optimization Trend')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Error')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('optimization_trend.png')
    print("趋势图已保存为: optimization_trend.png")
    plt.show()

# 执行可视化
plot_optimization_process()
plot_optimization_trend()
