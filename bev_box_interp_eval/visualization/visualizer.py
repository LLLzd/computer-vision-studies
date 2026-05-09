"""
可视化模块（Enhanced Professional Edition）

无原图依赖，纯BEV俯视视角可视化：
- 单帧对比图（png）
- 帧序列对比视频（mp4）
- 丰富指标图（多个子图，展示10+专业指标）
- 专业雷达图
- 支持速度和朝向箭头绘制
- 支持轨迹绘制（统一颜色）
- 支持方法图例
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Tuple
import matplotlib
matplotlib.use('Agg')


METHOD_COLORS = {
    "gt": (255, 0, 0),
    "linear": (0, 255, 0),
    "poly": (255, 255, 0),
    "kalman": (0, 0, 255),
    "spline": (255, 165, 0)
}

METHOD_LABELS = {
    "gt": "GT",
    "linear": "Linear",
    "poly": "Poly",
    "kalman": "Kalman",
    "spline": "Spline"
}

# 轨迹颜色：统一使用深灰色（所有轨迹同色）
TRAJECTORY_COLOR = (80, 80, 80)


class BEVVisualizer:
    """BEV可视化器（增强版）"""

    def __init__(self, config: dict):
        self.config = config
        self.canvas_width, self.canvas_height = config["visualization"]["bev_canvas_size"]
        self.output_dir = config["data"]["output_dir"]
        self.video_fps = config["visualization"]["video_fps"]
        self.dpi = config["visualization"]["dpi"]
        self.scale = self.canvas_width / 200.0  # 200米对应画布宽度（10像素/米）
        # 轨迹历史记录
        self.trajectory_history: Dict[str, List[Tuple[float, float]]] = {}

    def meters_to_pixels(self, x: float, y: float) -> Tuple[float, float]:
        """米坐标→像素坐标"""
        return x * self.scale, y * self.scale

    def draw_trajectory(self, image: np.ndarray, track_id: str, current_pos: Tuple[float, float]):
        """绘制车辆轨迹（统一颜色）"""
        if track_id not in self.trajectory_history:
            self.trajectory_history[track_id] = []
        
        # 记录当前位置
        self.trajectory_history[track_id].append(current_pos)
        
        # 绘制所有历史轨迹（统一颜色）
        if len(self.trajectory_history[track_id]) >= 2:
            pts = []
            for pos in self.trajectory_history[track_id]:
                px, py = self.meters_to_pixels(pos[0], pos[1])
                pts.append([int(px), int(self.canvas_height - py)])
            
            pts_array = np.array(pts, dtype=np.int32)
            cv2.polylines(image, [pts_array], isClosed=False, color=TRAJECTORY_COLOR, thickness=3)

    def draw_box_4points(self, image: np.ndarray, corners: List[float],
                         color: tuple, thickness: int = 2) -> np.ndarray:
        """绘制四点框

        参数:
            image: 图像数组
            corners: [x1,y1,x2,y2,x3,y3,x4,y4] 四个角点坐标（米）
            color: RGB颜色
            thickness: 线条粗细
        """
        if len(corners) < 8:
            return image

        # 米→像素转换
        px_corners = []
        for i in range(4):
            px_x, px_y = self.meters_to_pixels(corners[i*2], corners[i*2+1])
            px_corners.extend([px_x, px_y])

        pts = np.array([
            [px_corners[0], self.canvas_height - px_corners[1]],
            [px_corners[2], self.canvas_height - px_corners[3]],
            [px_corners[4], self.canvas_height - px_corners[5]],
            [px_corners[6], self.canvas_height - px_corners[7]],
        ], dtype=np.int32)

        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)
        return image

    def draw_velocity_arrow(self, image: np.ndarray, center: Tuple[float, float],
                           speed: float, yaw: float, color: tuple,
                           max_length: float = 60.0, thickness: int = 2) -> np.ndarray:
        """绘制速度箭头（长度与速度对数成正比）

        参数:
            image: 图像数组
            center: 起点 (x, y) - 米坐标
            speed: 速度 (m/s)
            yaw: 朝向角 (弧度)
            color: RGB颜色
            max_length: 最大箭头长度 (像素)
            thickness: 线条粗细
        """
        if speed < 0.1:
            return image

        log_speed = np.log1p(speed)
        log_max_speed = np.log1p(15.0)
        arrow_length = (log_speed / log_max_speed) * max_length
        arrow_length = max(8.0, min(arrow_length, max_length))

        # 米→像素转换
        px_center_x, px_center_y = self.meters_to_pixels(center[0], center[1])
        end_x = px_center_x + arrow_length * np.cos(-yaw)
        end_y = px_center_y - arrow_length * np.sin(-yaw)

        img_center = (int(px_center_x), int(self.canvas_height - px_center_y))
        img_end = (int(end_x), int(self.canvas_height - end_y))

        cv2.arrowedLine(image, img_center, img_end, color, thickness, tipLength=0.25)
        return image

    def draw_box_with_info(self, image: np.ndarray, box_info: Dict,
                          color: tuple, thickness: int = 2,
                          show_arrow: bool = True) -> np.ndarray:
        """根据box信息绘制四点框

        参数:
            image: 图像数组
            box_info: 包含bbox_bev_2d, yaw, speed, center, category, track_id等信息的字典
            color: RGB颜色
            thickness: 线条粗细
            show_arrow: 是否显示速度箭头
        """
        corners = box_info.get("bbox_bev_2d", [])
        if len(corners) < 8:
            return image

        self.draw_box_4points(image, corners, color, thickness)

        track_id = box_info.get("track_id", "")
        label = f"ID:{track_id}"

        # 米→像素转换
        px_x1, px_y1 = self.meters_to_pixels(corners[0], corners[1])
        text_x = int(px_x1)
        text_y = int(self.canvas_height - px_y1) - 5
        text_y = max(text_y, 15)

        cv2.putText(image, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if show_arrow:
            center = box_info.get("center", [corners[0], corners[1]])
            speed = box_info.get("speed", 0.0)
            yaw = box_info.get("yaw", 0.0)
            self.draw_velocity_arrow(image, center, speed, yaw, color, max_length=50.0, thickness=2)

        # 绘制轨迹（所有方法都绘制，统一颜色）
        track_id = box_info.get("track_id", "")
        if track_id:
            center = box_info.get("center", [corners[0], corners[1]])
            self.draw_trajectory(image, track_id, (center[0], center[1]))

        return image

    def draw_frame(self, frame_boxes: Dict[str, List[Dict]], frame_id: int,
                   metrics: Dict = None, show_velocity: bool = True) -> np.ndarray:
        """绘制单帧可视化图像（增强版：方法图例）"""
        image = np.ones((self.canvas_height, self.canvas_width, 3), dtype=np.uint8) * 255

        # 绘制网格
        grid_size = 100
        for x in range(0, self.canvas_width, grid_size):
            cv2.line(image, (x, 0), (x, self.canvas_height), (220, 220, 220), 1)
        for y in range(0, self.canvas_height, grid_size):
            cv2.line(image, (0, y), (self.canvas_width, y), (220, 220, 220), 1)

        # 绘制边界
        cv2.rectangle(image, (0, 0), (self.canvas_width-1, self.canvas_height-1), (100, 100, 100), 2)

        for method_name, boxes in frame_boxes.items():
            color = METHOD_COLORS.get(method_name, (128, 128, 128))

            for box in boxes:
                self.draw_box_with_info(image, box, color, thickness=2, show_arrow=show_velocity)

        cv2.putText(image, f"Frame: {frame_id}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # 绘制方法图例（不包括gt）
        legend_y = 50
        cv2.putText(image, "Methods:", (10, legend_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        for method_name, color in METHOD_COLORS.items():
            if method_name == "gt":
                continue
            cv2.rectangle(image, (10, legend_y), (30, legend_y+20), color, -1)
            cv2.putText(image, f"{METHOD_LABELS.get(method_name, method_name)}", (35, legend_y+15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            legend_y += 28

        # 绘制轨迹说明
        cv2.rectangle(image, (10, legend_y), (30, legend_y+20), TRAJECTORY_COLOR, -1)
        cv2.putText(image, "Trajectories", (35, legend_y+15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        if metrics:
            y_offset = self.canvas_height - 20
            for method, metric in metrics.items():
                iou = metric.get("iou", 0)
                center_err = metric.get("center_error", 0)
                text = f"{METHOD_LABELS.get(method, method)}: IOU={iou:.3f} CE={center_err:.1f}"
                cv2.putText(image, text, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, METHOD_COLORS.get(method, (0,0,0)), 1)
                y_offset -= 18

        return image

    def visualize_single_frame(self, frame_boxes: Dict[str, List[Dict]], frame_id: int,
                              output_path: str, show_velocity: bool = True) -> None:
        """可视化单帧并保存"""
        image = self.draw_frame(frame_boxes, frame_id, show_velocity=show_velocity)
        cv2.imwrite(output_path, image)
        print(f"Single frame saved: {output_path}")

    def visualize_sequence(self, all_boxes: Dict[str, List[Dict]],
                          frame_range: tuple, output_video_path: str,
                          show_velocity: bool = True) -> None:
        """可视化帧序列并生成视频（增强版）"""
        print(f"\nGenerating visualization video...")

        # 重置轨迹历史
        self.trajectory_history = {}

        boxes_by_frame = {}
        start_frame, end_frame = frame_range

        for method_name, boxes in all_boxes.items():
            for box in boxes:
                frame_id = box["frame_id"]
                if frame_id not in boxes_by_frame:
                    boxes_by_frame[frame_id] = {}
                if method_name not in boxes_by_frame[frame_id]:
                    boxes_by_frame[frame_id][method_name] = []
                boxes_by_frame[frame_id][method_name].append(box)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_path, fourcc, self.video_fps,
                                 (self.canvas_width, self.canvas_height))

        total_frames = end_frame - start_frame + 1
        for i, frame_id in enumerate(range(start_frame, end_frame + 1)):
            if frame_id in boxes_by_frame:
                frame_data = boxes_by_frame[frame_id]
            else:
                frame_data = {}
            image = self.draw_frame(frame_data, frame_id, show_velocity=show_velocity)
            writer.write(image)

            if (i + 1) % 20 == 0:
                print(f"  Processing frame {i+1}/{total_frames}")

        writer.release()
        print(f"Video saved: {output_video_path}")

    def plot_metrics(self, metrics_data: List[Dict], output_path: str) -> None:
        """绘制指标对比图（丰富专业版：2行3列，展示所有新增指标）"""
        if not metrics_data:
            return

        methods = [m["method_name"] for m in metrics_data]
        
        # 提取所有专业指标
        iou_means = [m.get("iou_mean", 0) for m in metrics_data]
        iou_90 = [m.get("iou_90_percentile", 0) for m in metrics_data]
        ce_means = [m.get("center_error_mean", 0) for m in metrics_data]
        ce_90 = [m.get("center_error_90_percentile", 0) for m in metrics_data]
        ade = [m.get("ade", 0) for m in metrics_data]
        fde = [m.get("fde", 0) for m in metrics_data]
        precisions = [m.get("precision", 0) for m in metrics_data]
        recalls = [m.get("recall", 0) for m in metrics_data]
        speed_vars = [m.get("speed_variance", 0) for m in metrics_data]
        jerks = [m.get("jerk", 0) for m in metrics_data]

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 配色方案
        colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
        
        # 第一行：IOU相关
        axes[0, 0].bar(methods, iou_means, color=colors[:len(methods)])
        axes[0, 0].set_title("IOU Mean", fontsize=13, fontweight='bold')
        axes[0, 0].set_ylim(0, 1.05)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        axes[0, 1].bar(methods, iou_90, color=colors[:len(methods)])
        axes[0, 1].set_title("IOU 90th Percentile", fontsize=13, fontweight='bold')
        axes[0, 1].set_ylim(0, 1.05)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 第二行：中心误差相关
        axes[0, 2].bar(methods, ce_means, color=colors[:len(methods)])
        axes[0, 2].set_title("Center Error Mean (meters)", fontsize=13, fontweight='bold')
        axes[0, 2].grid(True, alpha=0.3, axis='y')
        
        axes[1, 0].bar(methods, ce_90, color=colors[:len(methods)])
        axes[1, 0].set_title("Center Error 90th Percentile (meters)", fontsize=13, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 第三部分：轨迹指标
        axes[1, 1].bar(methods, ade, color=colors[:len(methods)])
        axes[1, 1].set_title("ADE (Average Displacement Error, meters)", fontsize=13, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # 检测指标和光滑度指标
        x = np.arange(len(methods))
        width = 0.35
        
        # 双Y轴：Precision & Recall
        ax1 = axes[1, 2]
        ax2 = ax1.twinx()
        rects1 = ax1.bar(x - width/2, precisions, width, label='Precision', color='#3498DB')
        rects2 = ax1.bar(x + width/2, recalls, width, label='Recall', color='#2ECC71')
        ax1.set_xlabel('Methods', fontsize=11)
        ax1.set_ylabel('Precision/Recall', fontsize=11, color='#2C3E50')
        ax1.set_ylim(0, 1.05)
        ax1.set_title("Detection Metrics", fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"Enhanced metrics chart saved: {output_path}")
        
        # 额外保存第二张图：速度方差等平滑度指标
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        axes[0].bar(methods, speed_vars, color=colors[:len(methods)])
        axes[0].set_title("Speed Variance (lower is smoother)", fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        axes[1].bar(methods, jerks, color=colors[:len(methods)])
        axes[1].set_title("Jerk (lower is smoother)", fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path2 = output_path.replace(".png", "_smoothness.png")
        plt.savefig(output_path2, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"Smoothness metrics chart saved: {output_path2}")

    def plot_radar_chart(self, metrics_data: List[Dict], output_path: str) -> None:
        """绘制雷达图（丰富专业版）"""
        if not metrics_data:
            return

        # 雷达图分类：7个专业指标
        categories = [
            "IOU Mean", 
            "Precision", 
            "Recall", 
            "Smoothness", 
            "Low Center Error",
            "Low ADE",
            "Low FDE"
        ]
        n_cats = len(categories)

        angles = [n / float(n_cats) * 2 * np.pi for n in range(n_cats)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']

        for idx, metrics in enumerate(metrics_data):
            # 计算各雷达值
            iou_val = metrics.get("iou_mean", 0)
            precision_val = metrics.get("precision", 0)
            recall_val = metrics.get("recall", 0)
            
            # 计算光滑度：1/(1+speed_variance)
            smoothness_val = 1.0 / (1.0 + metrics.get("speed_variance", 0.01))
            
            # 计算中心误差指标：1/(1+center_error_mean)
            ce_val = 1.0 / (1.0 + metrics.get("center_error_mean", 0.01))
            
            # 计算ADE指标：1/(1+ade)
            ade_val = 1.0 / (1.0 + metrics.get("ade", 0.01))
            
            # 计算FDE指标：1/(1+fde)
            fde_val = 1.0 / (1.0 + metrics.get("fde", 0.01))
            
            values = [iou_val, precision_val, recall_val, smoothness_val, ce_val, ade_val, fde_val]
            values += values[:1]

            ax.plot(angles, values, 'o-', linewidth=3, label=metrics["method_name"], color=colors[idx % len(colors)])
            ax.fill(angles, values, alpha=0.15, color=colors[idx % len(colors)])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=12, fontweight='medium')
        ax.set_ylim(0, 1.05)
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.05), fontsize=12)
        ax.set_title("Professional Method Comparison Radar Chart", size=16, fontweight='bold', pad=25)
        ax.grid(True, alpha=0.4)

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"Enhanced radar chart saved: {output_path}")

    def plot_frame_metrics(self, frame_metrics: Dict, output_path: str) -> None:
        """绘制逐帧指标变化图"""
        if not frame_metrics:
            return

        frame_ids = sorted(frame_metrics.keys())
        methods = list(next(iter(frame_metrics.values())).keys())

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        metrics_map = {
            "iou": ("IOU", 0, 1),
            "center_error": ("Center Error", 0, None),
            "width_error": ("Width Error", 0, None),
            "height_error": ("Height Error", 0, None),
        }

        colors = {'linear': '#2ECC71', 'poly': '#F1C40F', 'kalman': '#3498DB', 'spline': '#E74C3C'}

        for ax_idx, (metric_key, (metric_name, y_min, y_max)) in enumerate(metrics_map.items()):
            ax = axes[ax_idx // 2, ax_idx % 2]
            for method in methods:
                values = [frame_metrics[f].get(method, {}).get(metric_key, 0) for f in frame_ids]
                ax.plot(frame_ids, values, label=method, color=colors.get(method, '#333333'), linewidth=1.5)
            ax.set_xlabel("Frame ID")
            ax.set_ylabel(metric_name)
            ax.set_title(metric_name)
            ax.legend()
            ax.grid(True, alpha=0.3)
            if y_max:
                ax.set_ylim(y_min, y_max)

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"Frame metrics chart saved: {output_path}")
