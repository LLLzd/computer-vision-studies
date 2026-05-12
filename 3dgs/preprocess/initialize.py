#!/usr/bin/env python3
"""
使用OpenCV进行简化的Structure from Motion初始化
用于3DGS的高斯初始化
"""

import os
import numpy as np
import cv2
import pickle
from pathlib import Path


class SimpleSfM:
    """简化版Structure from Motion，用于初始化3DGS"""

    def __init__(self, intrinsic_matrix=None):
        """
        初始化SfM

        参数：
            intrinsic_matrix: 3x3相机内参矩阵，如果为None则使用默认参数
        """
        if intrinsic_matrix is None:
            self.K = np.array([
                [1000, 0, 640],
                [0, 1000, 360],
                [0, 0, 1]
            ], dtype=np.float32)
        else:
            self.K = intrinsic_matrix

        self.frames = []
        self.points_3d = []
        self.colors = []

    def extract_features(self, image, num_features=1000):
        """提取ORB特征"""
        orb = cv2.ORB_create(nfeatures=num_features)
        keypoints, descriptors = orb.detectAndCompute(image, None)
        return keypoints, descriptors

    def match_features(self, desc1, desc2):
        """匹配两组特征"""
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches

    def findEssentialMatrix(self, pts1, pts2):
        """使用5点算法计算本质矩阵"""
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        return E, mask

    def recoverPose(self, E, pts1, pts2, mask):
        """从本质矩阵恢复相机位姿"""
        retval, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)
        return R, t

    def triangulate(self, R, t, pts1, pts2):
        """三角化获得3D点"""
        proj1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        proj2 = self.K @ np.hstack((R, t))

        pts1 = pts1.T  # OpenCV expects shape (2, N)
        pts2 = pts2.T

        points_4d = cv2.triangulatePoints(proj1, proj2, pts1, pts2)
        points_3d = points_4d[:3] / points_4d[3]
        return points_3d.T

    def process_frames(self, frames_dir, output_file=None, max_frames=30):
        """
        处理帧目录，进行增量式SfM

        参数：
            frames_dir: str - 帧图像目录
            output_file: str - 输出文件路径
            max_frames: int - 最大处理帧数

        返回：
            points_3d: np.ndarray (N, 3) - 3D点坐标
            colors: np.ndarray (N, 3) - 颜色
            poses: list - 相机位姿列表
        """
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.png', '.jpg'))])
        frame_files = frame_files[:max_frames]

        print(f"Processing {len(frame_files)} frames...")

        prev_image = None
        prev_keypoints = None
        prev_descriptors = None
        prev_R = np.eye(3)
        prev_t = np.zeros((3, 1))
        all_points_3d = []
        all_colors = []

        for i, frame_file in enumerate(frame_files):
            image = cv2.imread(os.path.join(frames_dir, frame_file))
            if image is None:
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.extract_features(gray)

            print(f"Frame {i}: {len(keypoints)} features detected")

            if prev_descriptors is not None and len(keypoints) > 50:
                matches = self.match_features(prev_descriptors, descriptors)

                if len(matches) > 20:
                    pts1 = np.float32([prev_keypoints[m.queryIdx].pt for m in matches])
                    pts2 = np.float32([keypoints[m.trainIdx].pt for m in matches])

                    E, mask = self.findEssentialMatrix(pts1, pts2)

                    if E is not None and mask is not None and np.sum(mask) > 10:
                        R, t = self.recoverPose(E, pts1, pts2, mask)

                        relative_R = R.T @ prev_R
                        relative_t = -R.T @ t

                        prev_R = R
                        prev_t = relative_t

                        points_3d = self.triangulate(prev_R, prev_t, pts1, pts2)

                        for j, m in enumerate(matches):
                            if mask[j] and points_3d[j, 2] > 0 and points_3d[j, 2] < 20:
                                pt = points_3d[j]
                                x, y = int(pts2[j, 0]), int(pts2[j, 1])
                                if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                                    color = image[y, x] / 255.0
                                    all_points_3d.append(pt)
                                    all_colors.append(color)

            prev_image = image
            prev_keypoints = keypoints
            prev_descriptors = descriptors

        self.points_3d = np.array(all_points_3d)
        self.colors = np.array(all_colors)

        print(f"\nReconstructed {len(self.points_3d)} 3D points")

        return self.points_3d, self.colors


def create_gaussians_from_points(points_3d, colors, num_gaussians=50000):
    """
    从3D点云创建高斯分布

    参数：
        points_3d: np.ndarray (N, 3) - 3D点坐标
        colors: np.ndarray (N, 3) - 颜色 (0-1)
        num_gaussians: int - 生成的高斯数量

    返回：
        gaussians: list of dict - 高斯参数列表
    """
    n_points = len(points_3d)
    print(f"Creating {num_gaussians} Gaussians from {n_points} points...")

    indices = np.random.choice(n_points, min(num_gaussians, n_points), replace=n_points < num_gaussians)

    gaussians = []
    for idx in indices:
        point = points_3d[idx]
        color = colors[idx]

        gaussian = {
            'position': point + np.random.randn(3) * 0.01,
            'scale': np.array([0.02, 0.02, 0.02]) + np.random.rand(3) * 0.01,
            'rotation': np.random.rand(3) * 0.1,
            'color': color,
            'opacity': 0.8 + np.random.rand() * 0.2
        }
        gaussians.append(gaussian)

    return gaussians


def initialize_from_frames(frames_dir, output_dir, num_gaussians=50000, max_frames=30):
    """
    从帧序列初始化高斯

    参数：
        frames_dir: str - 帧目录
        output_dir: str - 输出目录
        num_gaussians: int - 高斯数量
        max_frames: int - 最大处理帧数

    返回：
        gaussians: list - 初始化的高斯列表
    """
    os.makedirs(output_dir, exist_ok=True)

    sfm = SimpleSfM()
    points_3d, colors = sfm.process_frames(frames_dir, output_file=os.path.join(output_dir, "pointcloud.npy"), max_frames=max_frames)

    if len(points_3d) < 100:
        print("Warning: Very few points reconstructed. Using fallback initialization.")
        gaussians = create_fallback_gaussians(frames_dir, num_gaussians)
    else:
        gaussians = create_gaussians_from_points(points_3d, colors, num_gaussians)

    output_file = os.path.join(output_dir, "gaussians_init.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(gaussians, f)
    print(f"Saved {len(gaussians)} Gaussians to {output_file}")

    return gaussians


def create_fallback_gaussians(frames_dir, num_gaussians=50000):
    """
    回退方案：从单帧图像创建高斯（基于颜色和边缘）
    """
    print("Using fallback initialization from single frame...")

    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.png', '.jpg'))])
    if not frame_files:
        print("No frames found!")
        return []

    image = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    if image is None:
        print("Cannot read frame!")
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=num_gaussians // 10)
    keypoints = orb.detect(gray, None)

    gaussians = []
    center_x, center_y = image.shape[1] // 2, image.shape[0] // 2

    for kp in keypoints:
        x, y = kp.pt
        color = image[int(y), int(x)] / 255.0

        gaussian = {
            'position': np.array([(x - center_x) * 0.001, (y - center_y) * 0.001, np.random.rand() * 0.5]),
            'scale': np.array([0.02, 0.02, 0.02]),
            'rotation': np.random.rand(3) * 0.1,
            'color': color,
            'opacity': 0.7 + np.random.rand() * 0.3
        }
        gaussians.append(gaussian)

    while len(gaussians) < num_gaussians:
        x = np.random.randint(0, image.shape[1])
        y = np.random.randint(0, image.shape[0])
        color = image[y, x] / 255.0

        gaussian = {
            'position': np.array([(x - center_x) * 0.001, (y - center_y) * 0.001, np.random.rand() * 0.5]),
            'scale': np.array([0.03, 0.03, 0.03]),
            'rotation': np.random.rand(3) * 0.1,
            'color': color,
            'opacity': 0.5 + np.random.rand() * 0.5
        }
        gaussians.append(gaussian)

    return gaussians


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Initialize Gaussians from video frames")
    parser.add_argument("--frames", "-f", type=str, default="frames",
                        help="Directory containing frames")
    parser.add_argument("--output", "-o", type=str, default="output",
                        help="Output directory")
    parser.add_argument("--num_gaussians", "-n", type=int, default=50000,
                        help="Number of Gaussians to create")
    parser.add_argument("--max_frames", "-m", type=int, default=30,
                        help="Maximum frames to process")

    args = parser.parse_args()

    frames_dir = args.frames
    if not os.path.exists(frames_dir):
        print(f"Frames directory not found: {frames_dir}")
        print("Please run extract_frames.py first")
        return

    gaussians = initialize_from_frames(
        frames_dir,
        args.output,
        num_gaussians=args.num_gaussians,
        max_frames=args.max_frames
    )

    print(f"\nInitialization complete!")
    print(f"Created {len(gaussians)} Gaussians")


if __name__ == "__main__":
    main()