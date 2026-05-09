"""
线性插值递推方法

原理：对同一track_id相邻两个关键帧的姿态(x,y,yaw)做帧间线性插值

适用：车辆匀速直线运动场景
缺点：转弯、变速误差大
"""

from typing import List, Dict
import numpy as np

from utils.data_format import BEVBox2D, compute_corners


def linear_interpolate_boxes(key_frame_boxes: List[BEVBox2D],
                             frame_ids: List[int]) -> List[Dict]:
    """
    线性插值生成中间帧Box

    参数：
        key_frame_boxes: 关键帧Box列表（至少2个）
        frame_ids: 待生成的帧ID列表

    返回：
        各帧递推Box结果列表
    """
    if len(key_frame_boxes) < 2:
        return []

    sorted_key_frames = sorted(key_frame_boxes, key=lambda x: x.frame_id)
    start_frame = sorted_key_frames[0].frame_id
    end_frame = sorted_key_frames[-1].frame_id

    results = []

    for frame_id in frame_ids:
        if frame_id <= start_frame:
            box = sorted_key_frames[0]
        elif frame_id >= end_frame:
            box = sorted_key_frames[-1]
        else:
            prev_key_frame = None
            next_key_frame = None

            for i in range(len(sorted_key_frames) - 1):
                if sorted_key_frames[i].frame_id <= frame_id < sorted_key_frames[i + 1].frame_id:
                    prev_key_frame = sorted_key_frames[i]
                    next_key_frame = sorted_key_frames[i + 1]
                    break

            if prev_key_frame is None or next_key_frame is None:
                continue

            total_frames = next_key_frame.frame_id - prev_key_frame.frame_id
            current_offset = frame_id - prev_key_frame.frame_id
            alpha = current_offset / total_frames

            center_x = prev_key_frame.center_x + (next_key_frame.center_x - prev_key_frame.center_x) * alpha
            center_y = prev_key_frame.center_y + (next_key_frame.center_y - prev_key_frame.center_y) * alpha

            yaw = prev_key_frame.yaw + (next_key_frame.yaw - prev_key_frame.yaw) * alpha

            speed = prev_key_frame.speed + (next_key_frame.speed - prev_key_frame.speed) * alpha

            w = prev_key_frame.w + (next_key_frame.w - prev_key_frame.w) * alpha
            l = prev_key_frame.l + (next_key_frame.l - prev_key_frame.l) * alpha

            corners = compute_corners([center_x, center_y], [w, l], yaw)

            box = BEVBox2D(
                frame_id=frame_id,
                is_key_frame=False,
                category=prev_key_frame.category,
                track_id=prev_key_frame.track_id,
                center=[center_x, center_y],
                velocity=[np.cos(yaw) * speed, np.sin(yaw) * speed],
                speed=speed,
                yaw=yaw,
                dimensions=[w, l],
                bbox_bev_2d=corners,
                score=prev_key_frame.score,
                category_name=prev_key_frame.category_name,
            )

        results.append({
            "frame_id": box.frame_id,
            "track_id": box.track_id,
            "category": box.category,
            "category_name": box.category_name,
            "center": box.center,
            "velocity": box.velocity,
            "speed": box.speed,
            "yaw": box.yaw,
            "dimensions": box.dimensions,
            "bbox_bev_2d": box.bbox_bev_2d,
            "score": box.score,
            "method": "linear"
        })

    return results


def run_linear_interp(track_sequences: dict, frame_range: tuple) -> List[Dict]:
    """对所有跟踪序列执行线性插值"""
    print("🔹 执行线性插值...")

    all_results = []
    start_frame, end_frame = frame_range

    for track_id, seq in track_sequences.items():
        key_frames = seq.get_key_frames()

        if len(key_frames) < 2:
            continue

        frame_ids = []
        for f in range(start_frame, end_frame + 1):
            if not any(kf.frame_id == f for kf in key_frames):
                frame_ids.append(f)

        results = linear_interpolate_boxes(key_frames, frame_ids)
        all_results.extend(results)

    print(f"✅ 线性插值完成，生成 {len(all_results)} 个递推Box")
    return all_results
