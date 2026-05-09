"""
视频动作识别 - 快速版

直接使用帧差法检测动作并截取片段，不使用大模型验证
适合快速测试和预览
"""

import os
import cv2
import numpy as np
from datetime import timedelta


def detect_and_extract(video_path, output_dir="outputs", threshold=30, min_duration=0.5):
    """
    检测动作并截取片段
    
    参数：
        video_path: 视频路径
        output_dir: 输出目录
        threshold: 帧差阈值
        min_duration: 最小动作时长（秒）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("🔍 分析视频...")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"视频信息:")
    print(f"  时长: {duration:.2f}秒")
    print(f"  帧率: {fps:.0f}fps")
    print(f"  分辨率: {width}x{height}")
    
    # 读取第一帧
    ret, prev_frame = cap.read()
    if not ret:
        print("❌ 无法读取视频")
        return
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)
    
    motion_frames = []
    frame_num = 0
    
    print("\n📊 检测动作...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        frame_diff = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        
        change_ratio = np.sum(thresh) / (thresh.shape[0] * thresh.shape[1] * 255)
        
        if change_ratio > 0.03:
            motion_frames.append(frame_num)
        
        prev_gray = gray
        frame_num += 1
    
    cap.release()
    
    # 合并连续帧为时间段
    if not motion_frames:
        print("⚠️  未检测到明显动作")
        return
    
    segments = []
    start_frame = motion_frames[0]
    
    for i in range(1, len(motion_frames)):
        if motion_frames[i] - motion_frames[i-1] > fps:
            end_frame = motion_frames[i-1]
            seg_duration = (end_frame - start_frame) / fps
            if seg_duration >= min_duration:
                segments.append((start_frame / fps, end_frame / fps))
            start_frame = motion_frames[i]
    
    end_frame = motion_frames[-1]
    seg_duration = (end_frame - start_frame) / fps
    if seg_duration >= min_duration:
        segments.append((start_frame / fps, end_frame / fps))
    
    print(f"\n📍 检测到 {len(segments)} 个动作片段")
    
    # 截取片段
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    for i, (start_time, end_time) in enumerate(segments, 1):
        # 扩展时间范围
        actual_start = max(0, start_time - 0.5)
        actual_end = min(duration, end_time + 0.5)
        
        output_path = os.path.join(output_dir, f"{video_name}_action_{i}.mp4")
        
        cap = cv2.VideoCapture(video_path)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        cap.set(cv2.CAP_PROP_POS_MSEC, actual_start * 1000)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if current_time > actual_end:
                break
            
            out.write(frame)
        
        cap.release()
        out.release()
        
        print(f"✂️ 片段{i}: {timedelta(seconds=start_time)} - {timedelta(seconds=end_time)} -> {output_path}")
    
    print(f"\n🎉 完成！共生成 {len(segments)} 个动作片段")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="视频动作检测与截取")
    parser.add_argument("-i", "--input", required=True, help="视频路径")
    parser.add_argument("-o", "--output", default="outputs", help="输出目录")
    args = parser.parse_args()
    
    detect_and_extract(args.input, args.output)
