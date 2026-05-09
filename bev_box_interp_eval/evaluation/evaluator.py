"""
评测指标计算模块 - 专业版

计算各递推方法的评测指标：
1. 检测精度类：mAP、Precision、Recall
2. IOU类：IOU均值、标准差、中位数、90分位数
3. 中心误差类：均值、标准差、中位数、最大值、90分位数
4. 角点误差类：均值、标准差
5. 航向角误差类：均值、标准差
6. 轨迹类：ADE、FDE、轨迹长度比
7. 平滑度类：速度方差、加速度方差、Jerk
"""

import numpy as np
from typing import List, Dict, Tuple

from utils.iou_utils import (
    calculate_iou, 
    calculate_center_distance, 
    calculate_size_error, 
    match_boxes,
    calculate_corner_error,
    calculate_yaw_error,
    calculate_trajectory_length
)
from utils.data_format import EvaluationResult


class Evaluator:
    """评测器类"""
    
    def __init__(self, config: dict):
        self.config = config
        self.iou_threshold = config["evaluation"]["iou_threshold"]
        self.iou_thresholds_ap = config["evaluation"]["iou_thresholds_ap"]
    
    def evaluate_single_method(self, pred_boxes: List[Dict], 
                              gt_boxes: List[Dict]) -> EvaluationResult:
        """
        评估单个递推方法的性能
        
        参数：
            pred_boxes: 预测Box列表
            gt_boxes: 真值Box列表
        
        返回：
            EvaluationResult
        """
        if not pred_boxes or not gt_boxes:
            return EvaluationResult(
                method_name="",
                iou_mean=0.0,
                iou_std=0.0,
                iou_median=0.0,
                iou_90_percentile=0.0,
                center_error_mean=0.0,
                center_error_std=0.0,
                center_error_median=0.0,
                center_error_max=0.0,
                center_error_90_percentile=0.0,
                width_error_mean=0.0,
                width_error_std=0.0,
                height_error_mean=0.0,
                height_error_std=0.0,
                corner_error_mean=0.0,
                corner_error_std=0.0,
                yaw_error_mean=0.0,
                yaw_error_std=0.0,
                ade=0.0,
                fde=0.0,
                trajectory_length_ratio=0.0,
                speed_variance=0.0,
                acceleration_variance=0.0,
                jerk=0.0,
                precision=0.0,
                recall=0.0,
                mAP={}
            )
        
        method_name = pred_boxes[0].get("method", "unknown")
        
        # 按帧ID分组
        pred_by_frame = {}
        for box in pred_boxes:
            frame_id = box["frame_id"]
            if frame_id not in pred_by_frame:
                pred_by_frame[frame_id] = []
            pred_by_frame[frame_id].append(box)
        
        gt_by_frame = {}
        for box in gt_boxes:
            frame_id = box["frame_id"]
            if frame_id not in gt_by_frame:
                gt_by_frame[frame_id] = []
            gt_by_frame[frame_id].append(box)
        
        # 使用所有帧，不跳过任何帧
        all_frame_ids = sorted(pred_by_frame.keys())
        
        # 逐帧计算指标
        iou_scores = []
        center_errors = []
        width_errors = []
        height_errors = []
        corner_errors = []
        yaw_errors = []
        all_matches = []
        
        for frame_id in all_frame_ids:
            if frame_id not in gt_by_frame:
                continue
            
            frame_pred = pred_by_frame[frame_id]
            frame_gt = gt_by_frame[frame_id]
            
            # 匹配Box
            matches = match_boxes(frame_pred, frame_gt, self.iou_threshold)
            all_matches.extend(matches)
            
            for pred, gt in matches:
                iou = calculate_iou(pred["bbox_bev_2d"], gt["bbox_bev_2d"])
                center_dist = calculate_center_distance(pred, gt)
                width_err, height_err = calculate_size_error(pred, gt)
                corner_err = calculate_corner_error(pred["bbox_bev_2d"], gt["bbox_bev_2d"])
                yaw_err = calculate_yaw_error(pred.get("yaw", 0), gt.get("yaw", 0))
                
                iou_scores.append(iou)
                center_errors.append(center_dist)
                width_errors.append(width_err)
                height_errors.append(height_err)
                corner_errors.append(corner_err)
                yaw_errors.append(yaw_err)
        
        # 计算精度和召回（使用所有帧）
        tp = len(all_matches)
        fp = sum(len(pred_by_frame[f]) for f in all_frame_ids) - tp
        fn = sum(len(gt_by_frame[f]) for f in all_frame_ids if f in pred_by_frame) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # 计算mAP
        mAP = self.calculate_map(pred_boxes, gt_boxes)
        
        # 计算轨迹类指标（ADE、FDE、轨迹长度比）
        ade, fde, trajectory_length_ratio = self.calculate_trajectory_metrics(pred_boxes, gt_boxes)
        
        # 计算平滑度类指标
        speed_variance, acceleration_variance, jerk = self.calculate_smoothness_metrics(pred_boxes)
        
        # 汇总所有统计指标
        iou_scores_np = np.array(iou_scores) if iou_scores else np.array([])
        center_errors_np = np.array(center_errors) if center_errors else np.array([])
        width_errors_np = np.array(width_errors) if width_errors else np.array([])
        height_errors_np = np.array(height_errors) if height_errors else np.array([])
        corner_errors_np = np.array(corner_errors) if corner_errors else np.array([])
        yaw_errors_np = np.array(yaw_errors) if yaw_errors else np.array([])
        
        return EvaluationResult(
            method_name=method_name,
            
            # IOU类指标
            iou_mean=np.mean(iou_scores_np) if len(iou_scores_np) > 0 else 0.0,
            iou_std=np.std(iou_scores_np) if len(iou_scores_np) > 1 else 0.0,
            iou_median=np.median(iou_scores_np) if len(iou_scores_np) > 0 else 0.0,
            iou_90_percentile=np.percentile(iou_scores_np, 90) if len(iou_scores_np) > 0 else 0.0,
            
            # 中心误差类指标
            center_error_mean=np.mean(center_errors_np) if len(center_errors_np) > 0 else 0.0,
            center_error_std=np.std(center_errors_np) if len(center_errors_np) > 1 else 0.0,
            center_error_median=np.median(center_errors_np) if len(center_errors_np) > 0 else 0.0,
            center_error_max=np.max(center_errors_np) if len(center_errors_np) > 0 else 0.0,
            center_error_90_percentile=np.percentile(center_errors_np, 90) if len(center_errors_np) > 0 else 0.0,
            
            # 尺寸误差类指标
            width_error_mean=np.mean(width_errors_np) if len(width_errors_np) > 0 else 0.0,
            width_error_std=np.std(width_errors_np) if len(width_errors_np) > 1 else 0.0,
            height_error_mean=np.mean(height_errors_np) if len(height_errors_np) > 0 else 0.0,
            height_error_std=np.std(height_errors_np) if len(height_errors_np) > 1 else 0.0,
            
            # 角点误差类指标
            corner_error_mean=np.mean(corner_errors_np) if len(corner_errors_np) > 0 else 0.0,
            corner_error_std=np.std(corner_errors_np) if len(corner_errors_np) > 1 else 0.0,
            
            # 航向角误差类指标
            yaw_error_mean=np.mean(yaw_errors_np) if len(yaw_errors_np) > 0 else 0.0,
            yaw_error_std=np.std(yaw_errors_np) if len(yaw_errors_np) > 1 else 0.0,
            
            # 轨迹类指标
            ade=ade,
            fde=fde,
            trajectory_length_ratio=trajectory_length_ratio,
            
            # 平滑度类指标
            speed_variance=speed_variance,
            acceleration_variance=acceleration_variance,
            jerk=jerk,
            
            # 检测类指标
            precision=precision,
            recall=recall,
            mAP=mAP
        )
    
    def calculate_trajectory_metrics(self, pred_boxes: List[Dict], gt_boxes: List[Dict]) -> Tuple[float, float, float]:
        """计算轨迹类指标：ADE、FDE、轨迹长度比"""
        # 按track_id分组
        pred_by_track = {}
        for box in pred_boxes:
            track_id = box["track_id"]
            if track_id not in pred_by_track:
                pred_by_track[track_id] = []
            pred_by_track[track_id].append(box)
        
        gt_by_track = {}
        for box in gt_boxes:
            track_id = box["track_id"]
            if track_id not in gt_by_track:
                gt_by_track[track_id] = []
            gt_by_track[track_id].append(box)
        
        ade_list = []
        fde_list = []
        length_ratios = []
        
        for track_id in pred_by_track:
            if track_id not in gt_by_track:
                continue
            
            pred_track = sorted(pred_by_track[track_id], key=lambda x: x["frame_id"])
            gt_track = sorted(gt_by_track[track_id], key=lambda x: x["frame_id"])
            
            # 建立帧的映射
            gt_frame_dict = {box["frame_id"]: box for box in gt_track}
            
            frame_errors = []
            final_error = 0.0
            
            for pred_box in pred_track:
                frame_id = pred_box["frame_id"]
                if frame_id in gt_frame_dict:
                    gt_box = gt_frame_dict[frame_id]
                    err = calculate_center_distance(pred_box, gt_box)
                    frame_errors.append(err)
                    final_error = err
            
            if frame_errors:
                ade_list.append(np.mean(frame_errors))
                fde_list.append(final_error)
            
            # 计算轨迹长度比
            pred_length = calculate_trajectory_length(pred_track)
            gt_length = calculate_trajectory_length(gt_track)
            if gt_length > 0:
                length_ratios.append(pred_length / gt_length)
        
        ade = np.mean(ade_list) if ade_list else 0.0
        fde = np.mean(fde_list) if fde_list else 0.0
        trajectory_length_ratio = np.mean(length_ratios) if length_ratios else 0.0
        
        return ade, fde, trajectory_length_ratio
    
    def calculate_smoothness_metrics(self, pred_boxes: List[Dict]) -> Tuple[float, float, float]:
        """计算平滑度类指标：速度方差、加速度方差、Jerk"""
        # 按track_id分组
        pred_by_track = {}
        for box in pred_boxes:
            track_id = box["track_id"]
            if track_id not in pred_by_track:
                pred_by_track[track_id] = []
            pred_by_track[track_id].append(box)
        
        all_speed_diffs = []
        all_accel_diffs = []
        all_jerk = []
        
        for track_id, boxes in pred_by_track.items():
            sorted_boxes = sorted(boxes, key=lambda x: x["frame_id"])
            
            if len(sorted_boxes) < 4:
                continue
            
            # 计算每个时刻的速度（基于位置变化）
            speeds = []
            for i in range(1, len(sorted_boxes)):
                frame_diff = sorted_boxes[i]["frame_id"] - sorted_boxes[i-1]["frame_id"]
                if frame_diff == 0:
                    continue
                
                center_prev = sorted_boxes[i-1]["center"]
                center_curr = sorted_boxes[i]["center"]
                distance = np.sqrt((center_curr[0] - center_prev[0])**2 + (center_curr[1] - center_prev[1])**2)
                speed = distance / frame_diff
                speeds.append(speed)
            
            # 计算速度变化和加速度
            accelerations = []
            for i in range(1, len(speeds)):
                speed_diff = speeds[i] - speeds[i-1]
                all_speed_diffs.append(abs(speed_diff))
                accel = speed_diff
                accelerations.append(accel)
            
            # 计算加速度变化和Jerk
            for i in range(1, len(accelerations)):
                accel_diff = accelerations[i] - accelerations[i-1]
                all_accel_diffs.append(abs(accel_diff))
                jerk = accel_diff
                all_jerk.append(abs(jerk))
        
        speed_variance = np.var(all_speed_diffs) if len(all_speed_diffs) > 1 else 0.0
        acceleration_variance = np.var(all_accel_diffs) if len(all_accel_diffs) > 1 else 0.0
        jerk = np.var(all_jerk) if len(all_jerk) > 1 else 0.0
        
        return speed_variance, acceleration_variance, jerk
    
    def calculate_map(self, pred_boxes: List[Dict], gt_boxes: List[Dict]) -> Dict:
        """计算mAP（mean Average Precision）"""
        mAP = {}
        
        for iou_thresh in self.iou_thresholds_ap:
            # 按track_id分组
            pred_by_track = {}
            for box in pred_boxes:
                track_id = box["track_id"]
                if track_id not in pred_by_track:
                    pred_by_track[track_id] = []
                pred_by_track[track_id].append(box)
            
            gt_by_track = {}
            for box in gt_boxes:
                track_id = box["track_id"]
                if track_id not in gt_by_track:
                    gt_by_track[track_id] = []
                gt_by_track[track_id].append(box)
            
            # 计算每个track的AP
            aps = []
            
            for track_id in pred_by_track:
                if track_id not in gt_by_track:
                    continue
                
                track_pred = sorted(pred_by_track[track_id], key=lambda x: -x.get("score", 1.0))
                track_gt = gt_by_track[track_id]
                
                tp = 0
                fp = 0
                precisions = []
                used_gt = set()
                
                for pred in track_pred:
                    matched = False
                    for idx, gt in enumerate(track_gt):
                        if idx in used_gt:
                            continue
                        iou = calculate_iou(pred["bbox_bev_2d"], gt["bbox_bev_2d"])
                        if iou >= iou_thresh:
                            tp += 1
                            used_gt.add(idx)
                            matched = True
                            break
                    
                    if not matched:
                        fp += 1
                    
                    if tp + fp > 0:
                        precisions.append(tp / (tp + fp))
                
                # 计算AP（简化版：使用所有precision的均值）
                if precisions:
                    ap = np.mean(precisions)
                    aps.append(ap)
            
            mAP[iou_thresh] = np.mean(aps) if aps else 0.0
        
        return mAP
    
    def evaluate_all_methods(self, results_by_method: Dict[str, List[Dict]], 
                            gt_boxes: List[Dict]) -> List[EvaluationResult]:
        """评估所有递推方法"""
        print("\n📊 开始评测...")
        
        results = []
        
        for method_name, pred_boxes in results_by_method.items():
            print(f"  评估方法: {method_name}")
            result = self.evaluate_single_method(pred_boxes, gt_boxes)
            result.method_name = method_name
            results.append(result)
        
        print("✅ 评测完成")
        return results
    
    def format_results(self, results: List[EvaluationResult]) -> str:
        """格式化评测结果为表格（专业版）"""
        lines = []
        lines.append("=" * 160)
        
        # 主表头
        header = (
            f"{'方法':<10} "
            f"{'IOU@mean':<10} "
            f"{'IOU@std':<10} "
            f"{'IOU@med':<10} "
            f"{'IOU@90%':<10} "
            f"{'CE@mean(m)':<10} "
            f"{'CE@90%':<10} "
            f"{'CorE@mean':<10} "
            f"{'YawE@mean':<10} "
            f"{'ADE(m)':<10} "
            f"{'FDE(m)':<10} "
            f"{'Prec':<8} "
            f"{'Recall':<8} "
            f"{'mAP@0.5':<8}"
        )
        lines.append(header)
        lines.append("-" * 160)
        
        for result in results:
            lines.append(
                f"{result.method_name:<10} "
                f"{result.iou_mean:<10.4f} "
                f"{result.iou_std:<10.4f} "
                f"{result.iou_median:<10.4f} "
                f"{result.iou_90_percentile:<10.4f} "
                f"{result.center_error_mean:<10.4f} "
                f"{result.center_error_90_percentile:<10.4f} "
                f"{result.corner_error_mean:<10.4f} "
                f"{result.yaw_error_mean:<10.4f} "
                f"{result.ade:<10.4f} "
                f"{result.fde:<10.4f} "
                f"{result.precision:<8.4f} "
                f"{result.recall:<8.4f} "
                f"{result.mAP.get(0.5, 0):<8.4f}"
            )
        
        lines.append("=" * 160)
        
        return "\n".join(lines)
