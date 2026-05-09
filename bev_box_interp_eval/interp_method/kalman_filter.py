
"""
卡尔曼滤波轨迹平滑递推方法

状态量：center_x, center_y, vx, vy, yaw, yaw_rate, width, length
观测值：关键帧的(x, y, yaw, w, l)
流程：关键帧观测更新，中间帧状态预测递推

优势：抗标注抖动、运动轨迹更平滑
"""

from typing import List, Dict
import numpy as np

from utils.data_format import BEVBox2D, compute_corners


class BoxKalmanFilter:
    """BEV Box卡尔曼滤波器（姿态版本）"""

    def __init__(self, initial_box: BEVBox2D, vx: float, vy: float):
        """状态量：[center_x, center_y, vx, vy, yaw, yaw_rate, w, l]"""
        self.x = np.array([
            initial_box.center_x,
            initial_box.center_y,
            vx,
            vy,
            initial_box.yaw,
            0.0,
            initial_box.w,
            initial_box.l
        ], dtype=np.float64)

        self.P = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01])

        self.F = np.eye(8)
        self.F[0, 2] = 1.0
        self.F[1, 3] = 1.0
        self.F[4, 5] = 1.0

        self.H = np.zeros((5, 8))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 4] = 1.0
        self.H[3, 6] = 1.0
        self.H[4, 7] = 1.0

        self.Q = np.diag([0.001, 0.001, 0.01, 0.01, 0.001, 0.0001, 0.0001, 0.0001])

        self.R = np.diag([0.1, 0.1, 0.01, 0.01, 0.01])

    def predict(self):
        """状态预测"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, obs_center: List[float], obs_yaw: float, obs_w: float, obs_l: float):
        """观测更新"""
        z = np.array([obs_center[0], obs_center[1], obs_yaw, obs_w, obs_l], dtype=np.float64)
        y = z - self.H @ self.x

        y[2] = np.arctan2(np.sin(y[2]), np.cos(y[2]))

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

    def get_box(self, frame_id: int, track_id: str, category: str, category_name: str) -> Dict:
        """获取当前状态对应的Box"""
        center_x, center_y, vx, vy, yaw, yaw_rate, w, l = self.x

        speed = np.sqrt(vx**2 + vy**2)
        corners = compute_corners([center_x, center_y], [w, l], yaw)

        return {
            "frame_id": frame_id,
            "track_id": track_id,
            "category": category,
            "category_name": category_name,
            "center": [float(center_x), float(center_y)],
            "velocity": [float(vx), float(vy)],
            "speed": float(speed),
            "yaw": float(yaw),
            "dimensions": [float(w), float(l)],
            "bbox_bev_2d": [float(c) for c in corners],
            "score": 1.0,
            "method": "kalman"
        }


def run_kalman_filter(track_sequences: dict, frame_range: tuple) -> List[Dict]:
    """对所有跟踪序列执行卡尔曼滤波递推"""
    print("🔹 执行卡尔曼滤波...")

    all_results = []
    start_frame, end_frame = frame_range

    for track_id, seq in track_sequences.items():
        key_frames = seq.get_key_frames()

        if len(key_frames) < 2:
            continue

        sorted_key_frames = sorted(key_frames, key=lambda x: x.frame_id)

        # 根据前两个关键帧计算初始速度
        f0 = sorted_key_frames[0]
        f1 = sorted_key_frames[1]
        frame_delta = f1.frame_id - f0.frame_id
        vx = (f1.center_x - f0.center_x) / frame_delta
        vy = (f1.center_y - f0.center_y) / frame_delta

        kf = BoxKalmanFilter(f0, vx=vx, vy=vy)

        results = []
        key_frame_idx = 0

        for frame_id in range(start_frame, end_frame + 1):
            is_key_frame = False
            obs_box = None

            if key_frame_idx < len(sorted_key_frames) and sorted_key_frames[key_frame_idx].frame_id == frame_id:
                is_key_frame = True
                obs_box = sorted_key_frames[key_frame_idx]
                key_frame_idx += 1

            if is_key_frame:
                kf.predict()
                kf.update(obs_box.center, obs_box.yaw, obs_box.w, obs_box.l)
            else:
                kf.predict()

            if not is_key_frame:
                box = kf.get_box(frame_id, track_id, sorted_key_frames[0].category, sorted_key_frames[0].category_name)
                results.append(box)

        all_results.extend(results)

    print(f"✅ 卡尔曼滤波完成，生成 {len(all_results)} 个递推Box")
    return all_results
