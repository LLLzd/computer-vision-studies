"""
二次多项式插值递推方法

原理：用前后多关键帧拟合二次曲线，预测中间帧Box姿态(x,y,yaw)

适用：车辆匀变速、平缓转向
"""

from typing import List, Dict
import numpy as np

from utils.data_format import BEVBox2D, compute_corners


def poly_interpolate_boxes(key_frame_boxes: List[BEVBox2D],
                          frame_ids: List[int]) -> List[Dict]:
    """二次多项式插值生成中间帧Box"""
    if len(key_frame_boxes) < 3:
        return []

    sorted_key_frames = sorted(key_frame_boxes, key=lambda x: x.frame_id)
    frame_nums = np.array([kf.frame_id for kf in sorted_key_frames], dtype=np.float64)

    center_x_vals = np.array([kf.center_x for kf in sorted_key_frames], dtype=np.float64)
    center_y_vals = np.array([kf.center_y for kf in sorted_key_frames], dtype=np.float64)
    yaw_vals = np.array([kf.yaw for kf in sorted_key_frames], dtype=np.float64)
    speed_vals = np.array([kf.speed for kf in sorted_key_frames], dtype=np.float64)
    w_vals = np.array([kf.w for kf in sorted_key_frames], dtype=np.float64)
    l_vals = np.array([kf.l for kf in sorted_key_frames], dtype=np.float64)

    coef_cx = np.polyfit(frame_nums, center_x_vals, 2)
    coef_cy = np.polyfit(frame_nums, center_y_vals, 2)
    coef_yaw = np.polyfit(frame_nums, yaw_vals, 2)
    coef_speed = np.polyfit(frame_nums, speed_vals, 2)
    coef_w = np.polyfit(frame_nums, w_vals, 2)
    coef_l = np.polyfit(frame_nums, l_vals, 2)

    poly_cx = np.poly1d(coef_cx)
    poly_cy = np.poly1d(coef_cy)
    poly_yaw = np.poly1d(coef_yaw)
    poly_speed = np.poly1d(coef_speed)
    poly_w = np.poly1d(coef_w)
    poly_l = np.poly1d(coef_l)

    results = []

    for frame_id in frame_ids:
        center_x = float(poly_cx(frame_id))
        center_y = float(poly_cy(frame_id))
        yaw = float(poly_yaw(frame_id))
        speed = max(0.0, float(poly_speed(frame_id)))
        w = float(poly_w(frame_id))
        l = float(poly_l(frame_id))

        w = max(0.5, w)
        l = max(0.5, l)

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
            "method": "poly"
        })

    return results


def run_poly_interp(track_sequences: dict, frame_range: tuple) -> List[Dict]:
    """对所有跟踪序列执行二次多项式插值"""
    print("🔹 执行二次多项式插值...")

    all_results = []
    start_frame, end_frame = frame_range

    for track_id, seq in track_sequences.items():
        key_frames = seq.get_key_frames()

        if len(key_frames) < 3:
            from interp_method.linear_interp import run_linear_interp
            results = run_linear_interp({track_id: seq}, frame_range)
            for r in results:
                r["method"] = "poly"
            all_results.extend(results)
            continue

        frame_ids = []
        for f in range(start_frame, end_frame + 1):
            if not any(kf.frame_id == f for kf in key_frames):
                frame_ids.append(f)

        results = poly_interpolate_boxes(key_frames, frame_ids)
        all_results.extend(results)

    print(f"✅ 二次多项式插值完成，生成 {len(all_results)} 个递推Box")
    return all_results
