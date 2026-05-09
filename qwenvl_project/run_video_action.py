"""
Qwen-VL 视频动作识别 - 简化版

功能：
1. 从视频提取关键帧并保存
2. 使用Qwen-VL分析关键帧
3. 自动识别目标动作并截取片段

简化策略：
- 只分析关键帧（每2秒1帧）
- 使用更简洁的提示词
- 并行处理
"""

import os
import cv2
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from datetime import timedelta


def load_model(model_path):
    """加载模型（使用MPS加速）"""
    print("📦 加载Qwen-VL模型...")
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="mps",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).eval()
    
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    print("✅ 模型加载完成")
    return model, processor


def extract_keyframes(video_path, interval=2.0):
    """提取关键帧"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"\n🎬 视频信息: {duration:.2f}秒, {fps:.0f}fps")
    
    frames = []
    frame_interval = int(fps * interval)
    target_frame = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if pos >= target_frame:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            frames.append((timestamp, image))
            target_frame = pos + frame_interval
    
    cap.release()
    print(f"📸 提取了 {len(frames)} 个关键帧")
    return frames, fps


def analyze_keyframes(frames, model, processor, keywords):
    """分析关键帧"""
    print("\n🧠 分析关键帧...")
    
    action_timestamps = []
    
    for i, (timestamp, image) in enumerate(frames):
        print(f"  帧{i}: {timestamp:.2f}秒...", end=" ")
        
        # 简洁提示词
        prompt = f"图中是否有人在{', '.join(keywords)}？回答是或否。"
        
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
        
        # 判断是否包含动作
        if "是" in response or "Yes" in response:
            action_timestamps.append(timestamp)
            print("✅ 检测到动作")
        else:
            print("❌ 未检测到")
    
    return action_timestamps


def merge_segments(timestamps, interval=2.0):
    """合并连续的动作时间段"""
    if not timestamps:
        return []
    
    segments = []
    start = timestamps[0]
    end = timestamps[0]
    
    for ts in timestamps[1:]:
        if ts - end <= interval:
            end = ts
        else:
            segments.append((start, end))
            start = ts
            end = ts
    
    segments.append((start, end))
    return segments


def extract_segment(video_path, start_time, end_time, output_path):
    """截取视频片段"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 扩展时间范围
    start_time = max(0, start_time - 1.0)
    end_time = end_time + 1.0
    
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
    
    parser = argparse.ArgumentParser(description="视频动作识别")
    parser.add_argument("-i", "--input", required=True, help="视频路径")
    parser.add_argument("-o", "--output", default="outputs", help="输出目录")
    parser.add_argument("-m", "--model", default="./models/Qwen/Qwen2___5-VL-3B-Instruct", help="模型路径")
    parser.add_argument("-k", "--keywords", nargs="+", default=["倒水", "拿水杯"], help="动作关键词")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # 1. 加载模型
    model, processor = load_model(args.model)
    
    # 2. 提取关键帧
    frames, fps = extract_keyframes(args.input)
    
    # 3. 分析关键帧
    action_timestamps = analyze_keyframes(frames, model, processor, args.keywords)
    
    if not action_timestamps:
        print("\n⚠️  未检测到目标动作")
        return
    
    # 4. 合并时间段
    segments = merge_segments(action_timestamps)
    print(f"\n📊 检测到 {len(segments)} 个动作片段")
    
    # 5. 截取片段
    video_name = os.path.splitext(os.path.basename(args.input))[0]
    for i, (start, end) in enumerate(segments, 1):
        output_path = os.path.join(args.output, f"{video_name}_action_{i}.mp4")
        extract_segment(args.input, start, end, output_path)
        print(f"✂️ 截取片段{i}: {timedelta(seconds=start)} - {timedelta(seconds=end)} -> {output_path}")
    
    print("\n🎉 完成！")


if __name__ == "__main__":
    main()
