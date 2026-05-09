"""
滑动窗口样条插值递推方法

原理：B样条/三次样条拟合关键帧轨迹，连续生成中间帧平滑Box

优势：非线性运动、转弯场景拟合效果最优
"""

from typing import List, Dict
import numpy as np
from scipy.interpolate import CubicSpline

from utils.data_format import BEVBox2D, compute_corners


def spline_interpolate_boxes(key_frame_boxes: List[BEVBox2D],
                            frame_ids: List[int]) -> List[Dict]:
    """三次样条插值生成中间帧Box"""
    if len(key_frame_boxes) < 4:
        return []

    sorted_key_frames = sorted(key_frame_boxes, key=lambda x: x.frame_id)
    frame_nums = np.array([kf.frame_id for kf in sorted_key_frames], dtype=np.float64)

    center_x_vals = np.array([kf.center_x for kf in sorted_key_frames], dtype=np.float64)
    center_y_vals = np.array([kf.center_y for kf in sorted_key_frames], dtype=np.float64)
    yaw_vals = np.array([kf.yaw for kf in sorted_key_frames], dtype=np.float64)
    speed_vals = np.array([kf.speed for kf in sorted_key_frames], dtype=np.float64)
    w_vals = np.array([kf.w for kf in sorted_key_frames], dtype=np.float64)
    l_vals = np.array([kf.l for kf in sorted_key_frames], dtype=np.float64)

    spline_cx = CubicSpline(frame_nums, center_x_vals)
    spline_cy = CubicSpline(frame_nums, center_y_vals)
    spline_yaw = CubicSpline(frame_nums, yaw_vals)
    spline_speed = CubicSpline(frame_nums, speed_vals)
    spline_w = CubicSpline(frame_nums, w_vals)
    spline_l = CubicSpline(frame_nums, l_vals)

    results = []

    for frame_id in frame_ids:
        if frame_id < frame_nums[0] or frame_id > frame_nums[-1]:
            continue

        center_x = float(spline_cx(frame_id))
        center_y = float(spline_cy(frame_id))
        yaw = float(spline_yaw(frame_id))
        speed = max(0.0, float(spline_speed(frame_id)))
        w = max(0.5, float(spline_w(frame_id)))
        l = max(0.5, float(spline_l(frame_id)))

        corners = compute_corners([center_x, center_y], [w, l], yaw)

        results.append({
            "frame_id": frame_id,
            "track_id": sorted_key_frames[0].track_id,
            "category": sorted_key_frames[0].category,
            "category_name": sorted_key_frames[0].category_name,
            "center": [center_x, center_y],
            "velocity": [np.cos(yaw) * speed, np.sin(yaw) * speed],
            "speed": speed,
            "yaw": yaw,
            "dimensions": [w, l],
            "bbox_bev_2d": corners,
            "score": sorted_key_frames[0].score,
            "method": "spline"
        })

    return results


def run_spline_interp(track_sequences: dict, frame_range: tuple) -> List[Dict]:
    """对所有跟踪序列执行样条插值"""
    print("🔹 执行样条插值...")

    all_results = []
    start_frame, end_frame = frame_range

    for track_id, seq in track_sequences.items():
        key_frames = seq.get_key_frames()

        if len(key_frames) < 4:
            from interp_method.poly_interp import run_poly_interp
            results = run_poly_interp({track_id: seq}, frame_range)
            for r in results:
                r["method"] = "spline"
            all_results.extend(results)
            continue

        frame_ids = []
        for f in range(start_frame, end_frame + 1):
            if not any(kf.frame_id == f for kf in key_frames):
                frame_ids.append(f)

        results = spline_interpolate_boxes(key_frames, frame_ids)
        all_results.extend(results)

    print(f"✅ 样条插值完成，生成 {len(all_results)} 个递推Box")
    return all_results
