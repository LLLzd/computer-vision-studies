"""
视频动作识别 - 混合方案

流程：
1. 使用OpenCV检测画面变化，定位可能包含动作的时间段
2. 只对关键帧使用Qwen-VL进行验证
3. 截取动作片段

优点：大大减少模型推理次数，提高速度
"""

import os
import cv2
import numpy as np
from datetime import timedelta


def detect_motion_segments(video_path, threshold=30, min_duration=1.0):
    """
    使用帧差法检测动作时间段
    
    参数：
        video_path: 视频路径
        threshold: 帧差阈值
        min_duration: 最小动作时长（秒）
    
    返回：
        segments: 动作时间段列表 [(start_time, end_time), ...]
    """
    print("🔍 使用帧差法检测动作...")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 读取第一帧
    ret, prev_frame = cap.read()
    if not ret:
        return []
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)
    
    motion_frames = []
    frame_num = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 转为灰度并模糊
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # 计算帧差
        frame_diff = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        
        # 计算变化比例
        change_ratio = np.sum(thresh) / (thresh.shape[0] * thresh.shape[1] * 255)
        
        # 如果变化超过阈值，标记为运动帧
        if change_ratio > 0.05:  # 超过5%的像素变化
            motion_frames.append(frame_num)
        
        prev_gray = gray
        frame_num += 1
    
    cap.release()
    
    # 合并连续的运动帧为时间段
    if not motion_frames:
        print("⚠️  未检测到明显动作")
        return []
    
    segments = []
    start_frame = motion_frames[0]
    
    for i in range(1, len(motion_frames)):
        if motion_frames[i] - motion_frames[i-1] > fps:  # 超过1秒间隔
            end_frame = motion_frames[i-1]
            duration = (end_frame - start_frame) / fps
            if duration >= min_duration:
                segments.append((start_frame / fps, end_frame / fps))
            start_frame = motion_frames[i]
    
    # 添加最后一个片段
    end_frame = motion_frames[-1]
    duration = (end_frame - start_frame) / fps
    if duration >= min_duration:
        segments.append((start_frame / fps, end_frame / fps))
    
    print(f"📍 检测到 {len(segments)} 个可能包含动作的时间段")
    for i, (start, end) in enumerate(segments, 1):
        print(f"  [{i}] {timedelta(seconds=start)} - {timedelta(seconds=end)}")
    
    return segments


def extract_keyframes_for_segments(video_path, segments, frames_per_segment=2):
    """
    从动作时间段中提取关键帧
    
    参数：
        video_path: 视频路径
        segments: 时间段列表
        frames_per_segment: 每个时间段提取的帧数
    
    返回：
        keyframes: 关键帧列表 [(timestamp, image), ...]
    """
    from PIL import Image
    
    print(f"\n📸 从动作时间段提取关键帧...")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    keyframes = []
    
    for start_time, end_time in segments:
        # 在时间段内均匀采样
        interval = (end_time - start_time) / (frames_per_segment + 1)
        
        for i in range(1, frames_per_segment + 1):
            timestamp = start_time + interval * i
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ret, frame = cap.read()
            
            if ret:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                keyframes.append((timestamp, image))
                print(f"    提取帧: {timestamp:.2f}秒")
    
    cap.release()
    print(f"✅ 共提取 {len(keyframes)} 个关键帧")
    return keyframes


def analyze_with_qwen(keyframes, model, processor, keywords):
    """
    使用Qwen-VL分析关键帧
    
    参数：
        keyframes: 关键帧列表
        model, processor: Qwen-VL模型和处理器
        keywords: 动作关键词
    
    返回：
        verified_segments: 验证后的动作时间段
    """
    import torch
    
    print(f"\n🧠 使用Qwen-VL验证动作...")
    
    action_timestamps = []
    
    for timestamp, image in keyframes:
        print(f"  分析帧: {timestamp:.2f}秒...", end=" ")
        
        # 简洁提示词
        prompt = f"图片中是否有人在进行以下动作：{', '.join(keywords)}？回答是或否。"
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }]
        
        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = processor(
            text=[text_prompt],
            images=[image],
            return_tensors="pt"
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=151643,
                eos_token_id=151643
            )
        
        response = processor.decode(outputs[0], skip_special_tokens=True)
        
        if "是" in response or "Yes" in response:
            action_timestamps.append(timestamp)
            print("✅ 确认动作")
        else:
            print("❌ 排除")
    
    return action_timestamps


def extract_video_segment(video_path, start_time, end_time, output_path):
    """截取视频片段"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 扩展时间范围
    start_time = max(0, start_time - 0.5)
    end_time = end_time + 0.5
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if current_time > end_time:
            break
        
        out.write(frame)
    
    cap.release()
    out.release()


def main():
    import argparse
    import torch
    from PIL import Image
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    
    parser = argparse.ArgumentParser(description="视频动作识别")
    parser.add_argument("-i", "--input", required=True, help="视频路径")
    parser.add_argument("-o", "--output", default="outputs", help="输出目录")
    parser.add_argument("-m", "--model", default="./models/Qwen/Qwen2___5-VL-3B-Instruct", help="模型路径")
    parser.add_argument("-k", "--keywords", nargs="+", default=["倒水", "拿水杯"], help="动作关键词")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # 步骤1: 使用OpenCV检测动作时间段
    segments = detect_motion_segments(args.input)
    
    if not segments:
        print("❌ 未检测到任何动作")
        return
    
    # 步骤2: 提取关键帧
    keyframes = extract_keyframes_for_segments(args.input, segments)
    
    # 步骤3: 使用Qwen-VL验证（如果有关键帧）
    if keyframes:
        print("\n📦 加载Qwen-VL模型...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval()
        
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        
        action_timestamps = analyze_with_qwen(keyframes, model, processor, args.keywords)
        
        if action_timestamps:
            # 合并时间戳为时间段
            action_timestamps.sort()
            start_time = action_timestamps[0]
            end_time = action_timestamps[-1]
            
            # 截取片段
            video_name = os.path.splitext(os.path.basename(args.input))[0]
            output_path = os.path.join(args.output, f"{video_name}_action.mp4")
            extract_video_segment(args.input, start_time, end_time, output_path)
            
            print(f"\n🎉 完成！动作片段已保存至: {output_path}")
            print(f"   时间段: {timedelta(seconds=start_time)} - {timedelta(seconds=end_time)}")
        else:
            print("\n⚠️  Qwen-VL未验证到目标动作")
    
    else:
        # 如果没有关键帧，直接输出检测到的时间段
        print("\n📋 输出检测到的动作片段")
        video_name = os.path.splitext(os.path.basename(args.input))[0]
        
        for i, (start, end) in enumerate(segments, 1):
            output_path = os.path.join(args.output, f"{video_name}_motion_{i}.mp4")
            extract_video_segment(args.input, start, end, output_path)
            print(f"   片段{i}: {timedelta(seconds=start)} - {timedelta(seconds=end)} -> {output_path}")


if __name__ == "__main__":
    main()
