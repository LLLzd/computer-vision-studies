"""
Qwen-VL 视频理解与动作识别系统

功能：
1. 从视频中提取关键帧
2. 使用Qwen-VL进行帧级语义理解
3. 识别并定位目标动作片段（如水杯倒水）
4. 自动截取动作片段

技术方案：
- 帧采样：每秒提取1帧
- 帧分析：多模态大模型理解帧内容
- 动作检测：基于帧描述判断动作发生
- 时间戳融合：合并连续相似帧的时间戳
"""

import os
import json
import cv2
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import numpy as np
from tqdm import tqdm
from datetime import timedelta


class VideoActionAnalyzer:
    """视频动作分析器"""
    
    def __init__(self, model_path=None):
        self.model = None
        self.processor = None
        self.device = "cpu"  # 使用CPU避免MPS内存不足
        self.torch_dtype = torch.float32  # 使用float32提高兼容性
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """加载Qwen-VL模型"""
        print("=" * 60)
        print("📦 正在加载Qwen-VL模型...")
        print("=" * 60)
        
        # 加载模型
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval()
        
        # 加载处理器
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        print("✅ 模型加载完成！")
    
    def extract_frames(self, video_path, frame_interval=1.0, max_frames=500):
        """
        从视频中提取关键帧
        
        参数：
            video_path: 视频文件路径
            frame_interval: 帧采样间隔（秒）
            max_frames: 最大采样帧数
        
        返回：
            frames: 帧列表 [(timestamp, image), ...]
        """
        print(f"\n🎬 正在提取视频帧: {os.path.basename(video_path)}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"  视频信息:")
        print(f"    帧率: {fps:.2f} fps")
        print(f"    总帧数: {total_frames}")
        print(f"    时长: {timedelta(seconds=duration)}")
        print(f"    采样间隔: {frame_interval}秒")
        
        frames = []
        frame_count = 0
        target_frame = 0
        frame_interval_frames = int(fps * frame_interval)
        
        with tqdm(total=min(max_frames, int(total_frames / frame_interval_frames)), desc="提取帧") as pbar:
            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 按间隔采样
                if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) >= target_frame:
                    # 转换为PIL图像
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(image)
                    
                    # 获取时间戳
                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    
                    frames.append((timestamp, image))
                    frame_count += 1
                    target_frame = frame_count * frame_interval_frames
                    pbar.update(1)
        
        cap.release()
        print(f"✅ 提取完成，共 {len(frames)} 帧")
        return frames
    
    def analyze_frame(self, image, prompt=None):
        """
        使用Qwen-VL分析单帧图像
        
        参数：
            image: PIL图像
            prompt: 分析提示词
        
        返回：
            result: 分析结果（JSON格式）
        """
        if prompt is None:
            # 优化：更简洁的提示词，提高速度
            prompt = "描述图片中的动作，如：倒水、拿杯、行走等。只用中文4-8个字描述动作。"
        
        # 构造对话
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # 应用聊天模板
        text_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 准备输入
        inputs = self.processor(
            text=[text_prompt],
            images=[image],
            return_tensors="pt"
        ).to(self.model.device)
        
        # 推理（优化参数）
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,  # 大幅减少生成长度
                do_sample=False,
                pad_token_id=151643,
                eos_token_id=151643
            )
        
        # 解析结果
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # 清理输出
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()
        
        return response.strip()
    
    def detect_action(self, frame_descriptions, action_keywords):
        """
        检测动作发生的时间段
        
        参数：
            frame_descriptions: 帧描述列表 [(timestamp, description), ...]
            action_keywords: 动作关键词列表
        
        返回：
            action_segments: 动作片段列表 [(start_time, end_time, description), ...]
        """
        print(f"\n🔍 正在检测动作: {', '.join(action_keywords)}")
        
        action_frames = []
        
        for timestamp, description in frame_descriptions:
            # 检查是否包含动作关键词
            for keyword in action_keywords:
                if keyword in description:
                    action_frames.append((timestamp, description))
                    break
        
        if not action_frames:
            print("⚠️  未检测到目标动作")
            return []
        
        # 合并连续的动作帧
        action_segments = []
        current_segment = [action_frames[0]]
        
        for frame in action_frames[1:]:
            # 检查是否连续（间隔小于1.5倍采样间隔）
            if frame[0] - current_segment[-1][0] < 2.0:
                current_segment.append(frame)
            else:
                # 结束当前片段
                start_time = current_segment[0][0]
                end_time = current_segment[-1][0]
                descriptions = [f[1] for f in current_segment]
                action_segments.append((start_time, end_time, descriptions))
                current_segment = [frame]
        
        # 添加最后一个片段
        if current_segment:
            start_time = current_segment[0][0]
            end_time = current_segment[-1][0]
            descriptions = [f[1] for f in current_segment]
            action_segments.append((start_time, end_time, descriptions))
        
        # 输出结果
        print(f"✅ 检测到 {len(action_segments)} 个动作片段:")
        for i, (start, end, desc) in enumerate(action_segments, 1):
            print(f"  [{i}] {timedelta(seconds=start)} - {timedelta(seconds=end)}")
        
        return action_segments
    
    def extract_video_segment(self, video_path, start_time, end_time, output_path):
        """
        截取视频片段
        
        参数：
            video_path: 原始视频路径
            start_time: 开始时间（秒）
            end_time: 结束时间（秒）
            output_path: 输出路径
        """
        print(f"\n✂️ 正在截取片段: {timedelta(seconds=start_time)} - {timedelta(seconds=end_time)}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 设置视频编码器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 定位到开始时间
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        
        current_time = start_time
        frame_count = 0
        
        while current_time < end_time:
            ret, frame = cap.read()
            if not ret:
                break
            
            out.write(frame)
            frame_count += 1
            current_time = start_time + frame_count / fps
        
        cap.release()
        out.release()
        print(f"✅ 片段已保存: {output_path}")
    
    def analyze_video(self, video_path, action_keywords, output_dir="outputs"):
        """
        完整的视频分析流程
        
        参数：
            video_path: 视频路径
            action_keywords: 动作关键词
            output_dir: 输出目录
        
        返回：
            results: 分析结果
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 提取帧
        frames = self.extract_frames(video_path)
        
        # 2. 分析每帧
        frame_descriptions = []
        print("\n🧠 正在分析帧内容...")
        for timestamp, image in tqdm(frames, desc="分析帧"):
            description = self.analyze_frame(image)
            frame_descriptions.append((timestamp, description))
            # 打印进度
            if (len(frame_descriptions) % 10 == 0):
                print(f"   帧 {len(frame_descriptions)}: {description[:50]}...")
        
        # 保存帧分析结果
        analysis_path = os.path.join(output_dir, "frame_analysis.json")
        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump(frame_descriptions, f, indent=2, ensure_ascii=False)
        
        # 3. 检测动作
        action_segments = self.detect_action(frame_descriptions, action_keywords)
        
        # 4. 截取动作片段
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        results = []
        
        for i, (start_time, end_time, descriptions) in enumerate(action_segments, 1):
            output_path = os.path.join(output_dir, f"{video_name}_action_{i}.mp4")
            self.extract_video_segment(video_path, start_time, end_time, output_path)
            
            results.append({
                "segment_id": i,
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
                "output_path": output_path,
                "frame_descriptions": descriptions
            })
        
        # 保存结果
        results_path = os.path.join(output_dir, "action_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 分析完成！结果已保存至 {output_dir}")
        return results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Qwen-VL视频动作识别")
    parser.add_argument("-i", "--input", required=True, help="输入视频路径")
    parser.add_argument("-o", "--output", default="outputs", help="输出目录")
    parser.add_argument("-m", "--model", default="./models/Qwen/Qwen2___5-VL-3B-Instruct", help="模型路径")
    parser.add_argument("-k", "--keywords", nargs="+", default=["倒水", "拿水杯", "倒水动作"], 
                        help="动作关键词")
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = VideoActionAnalyzer()
    
    # 加载模型
    analyzer.load_model(args.model)
    
    # 分析视频
    results = analyzer.analyze_video(
        video_path=args.input,
        action_keywords=args.keywords,
        output_dir=args.output
    )
    
    # 打印结果摘要
    print("\n" + "=" * 60)
    print("📋 分析结果摘要")
    print("=" * 60)
    for result in results:
        print(f"\n动作片段 {result['segment_id']}:")
        print(f"  时间: {timedelta(seconds=result['start_time'])} - {timedelta(seconds=result['end_time'])}")
        print(f"  时长: {result['duration']:.2f}秒")
        print(f"  文件: {result['output_path']}")


if __name__ == "__main__":
    main()
