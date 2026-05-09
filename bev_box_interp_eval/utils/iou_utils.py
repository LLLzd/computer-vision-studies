"""
IOU计算工具函数
"""

import numpy as np
from typing import List, Tuple


def polygon_area(polygon: np.ndarray) -> float:
    """计算多边形面积（使用Shoelace公式）"""
    if len(polygon) < 3:
        return 0.0
    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def polygon_intersection(poly1: np.ndarray, poly2: np.ndarray) -> list:
    """
    计算两个凸多边形的交集（使用Sutherland-Hodgman算法）
    """
    def inside(point, edge_start, edge_end):
        return (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1]) >= \
               (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0])

    def intersect(s, e, edge_start, edge_end):
        dc = [edge_start[0] - edge_end[0], edge_start[1] - edge_end[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = edge_start[0] * edge_end[1] - edge_start[1] * edge_end[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    output = poly1.tolist()
    for i in range(len(poly2)):
        edge_start = poly2[i]
        edge_end = poly2[(i + 1) % len(poly2)]
        if not output:
            break
        input_list = output
        output = []
        s = input_list[-1]
        for e in input_list:
            if inside(e, edge_start, edge_end):
                if not inside(s, edge_start, edge_end):
                    output.append(intersect(s, e, edge_start, edge_end))
                output.append(e)
            elif inside(s, edge_start, edge_end):
                output.append(intersect(s, e, edge_start, edge_end))
            s = e
    return output


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """计算两个BEV 2D Box的IOU（支持四点旋转框）"""

    if len(box1) >= 8 and len(box2) >= 8:
        poly1 = np.array([
            [box1[0], box1[1]],
            [box1[2], box1[3]],
            [box1[4], box1[5]],
            [box1[6], box1[7]]
        ])
        poly2 = np.array([
            [box2[0], box2[1]],
            [box2[2], box2[3]],
            [box2[4], box2[5]],
            [box2[6], box2[7]]
        ])

        area1 = polygon_area(poly1)
        area2 = polygon_area(poly2)

        intersection_poly = polygon_intersection(poly1, poly2)
        if len(intersection_poly) >= 3:
            intersection_area = polygon_area(np.array(intersection_poly))
        else:
            intersection_area = 0.0

        union_area = area1 + area2 - intersection_area

        if union_area == 0:
            return 0.0

        return intersection_area / union_area

    x1_1, y1_1, x2_1, y2_1 = box1[:4]
    x1_2, y1_2, x2_2, y2_2 = box2[:4]

    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    area1 = abs(x2_1 - x1_1) * abs(y2_1 - y1_1)
    area2 = abs(x2_2 - x1_2) * abs(y2_2 - y1_2)
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def calculate_center_distance(box1: List[float], box2: List[float]) -> float:
    """计算两个Box中心点之间的L2距离"""

    if len(box1) >= 2 and len(box2) >= 2:
        if isinstance(box1, dict) and "center" in box1:
            cx1, cy1 = box1["center"]
        elif len(box1) >= 8:
            xs = [box1[0], box1[2], box1[4], box1[6]]
            ys = [box1[1], box1[3], box1[5], box1[7]]
            cx1, cy1 = np.mean(xs), np.mean(ys)
        else:
            cx1 = (box1[0] + box1[2]) / 2
            cy1 = (box1[1] + box1[3]) / 2

        if isinstance(box2, dict) and "center" in box2:
            cx2, cy2 = box2["center"]
        elif len(box2) >= 8:
            xs = [box2[0], box2[2], box2[4], box2[6]]
            ys = [box2[1], box2[3], box2[5], box2[7]]
            cx2, cy2 = np.mean(xs), np.mean(ys)
        else:
            cx2 = (box2[0] + box2[2]) / 2
            cy2 = (box2[1] + box2[3]) / 2

        return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

    cx1 = (box1[0] + box1[2]) / 2
    cy1 = (box1[1] + box1[3]) / 2
    cx2 = (box2[0] + box2[2]) / 2
    cy2 = (box2[1] + box2[3]) / 2

    return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)


def calculate_size_error(box_pred, box_gt) -> Tuple[float, float]:
    """计算Box尺寸误差"""

    if isinstance(box_pred, dict) and "dimensions" in box_pred:
        w_pred = box_pred["dimensions"][0]
        l_pred = box_pred["dimensions"][1]
    elif len(box_pred) >= 8:
        xs = [box_pred[0], box_pred[2], box_pred[4], box_pred[6]]
        ys = [box_pred[1], box_pred[3], box_pred[5], box_pred[7]]
        w_pred = max(xs) - min(xs)
        l_pred = max(ys) - min(ys)
    else:
        w_pred = abs(box_pred[2] - box_pred[0])
        l_pred = abs(box_pred[3] - box_pred[1])

    if isinstance(box_gt, dict) and "dimensions" in box_gt:
        w_gt = box_gt["dimensions"][0]
        l_gt = box_gt["dimensions"][1]
    elif len(box_gt) >= 8:
        xs = [box_gt[0], box_gt[2], box_gt[4], box_gt[6]]
        ys = [box_gt[1], box_gt[3], box_gt[5], box_gt[7]]
        w_gt = max(xs) - min(xs)
        l_gt = max(ys) - min(ys)
    else:
        w_gt = abs(box_gt[2] - box_gt[0])
        l_gt = abs(box_gt[3] - box_gt[1])

    return abs(w_pred - w_gt), abs(l_pred - l_gt)


def match_boxes(pred_boxes: List[dict], gt_boxes: List[dict],
                iou_threshold: float = 0.5) -> List[Tuple[dict, dict]]:
    """基于IOU匹配预测Box和真值Box"""
    matches = []
    used_gt_indices = set()

    for pred_box in pred_boxes:
        best_iou = 0.0
        best_gt_idx = -1

        for idx, gt_box in enumerate(gt_boxes):
            if idx in used_gt_indices:
                continue

            iou = calculate_iou(pred_box["bbox_bev_2d"], gt_box["bbox_bev_2d"])
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_gt_idx = idx

        if best_gt_idx != -1:
            matches.append((pred_box, gt_boxes[best_gt_idx]))
            used_gt_indices.add(best_gt_idx)

    return matches


def calculate_corner_error(box_pred: List[float], box_gt: List[float]) -> float:
    """计算四个角点的平均距离误差（单位：米）"""
    if len(box_pred) < 8 or len(box_gt) < 8:
        return 0.0
    
    # 获取预测框和真值框的角点
    corners_pred = [
        (box_pred[0], box_pred[1]),
        (box_pred[2], box_pred[3]),
        (box_pred[4], box_pred[5]),
        (box_pred[6], box_pred[7]),
    ]
    corners_gt = [
        (box_gt[0], box_gt[1]),
        (box_gt[2], box_gt[3]),
        (box_gt[4], box_gt[5]),
        (box_gt[6], box_gt[7]),
    ]
    
    # 对预测角点进行排序以匹配真值角点顺序（基于最近邻）
    total_dist = 0.0
    used_indices = set()
    
    for gt_corner in corners_gt:
        min_dist = float('inf')
        best_idx = -1
        for i, pred_corner in enumerate(corners_pred):
            if i in used_indices:
                continue
            dist = np.sqrt((gt_corner[0] - pred_corner[0])**2 + (gt_corner[1] - pred_corner[1])**2)
            if dist < min_dist:
                min_dist = dist
                best_idx = i
        if best_idx != -1:
            total_dist += min_dist
            used_indices.add(best_idx)
    
    return total_dist / 4.0


def calculate_yaw_error(yaw_pred: float, yaw_gt: float) -> float:
    """计算航向角误差（归一化到[-pi, pi]范围）"""
    error = yaw_pred - yaw_gt
    error = np.arctan2(np.sin(error), np.cos(error))  # 归一化
    return abs(error)


def calculate_trajectory_length(boxes: List[dict]) -> float:
    """计算轨迹的总长度（单位：米）"""
    if len(boxes) < 2:
        return 0.0
    
    sorted_boxes = sorted(boxes, key=lambda x: x["frame_id"])
    total_length = 0.0
    
    for i in range(1, len(sorted_boxes)):
        prev_center = sorted_boxes[i-1]["center"]
        curr_center = sorted_boxes[i]["center"]
        dist = np.sqrt((curr_center[0] - prev_center[0])**2 + (curr_center[1] - prev_center[1])**2)
        total_length += dist
    
    return total_length
