"""
Qwen-VL 视频动作识别 - 优化加速版

优化策略：
1. ✅ 使用轻量模型 (3B)
2. ✅ AWQ 4bit 量化（直接使用预量化模型）
3. ✅ 分辨率下采样 (1/8)
4. ✅ 增大抽帧间隔 (5秒)
5. ✅ 批处理
6. ✅ 阿里云模型下载（国内速度快）

可用模型：
   - 3b: Qwen2.5-VL-3B-Instruct（原始模型）
   - 3b-awq: Qwen2.5-VL-3B-Instruct-AWQ（4bit量化模型，推荐）

使用方法：
    # 推荐：使用AWQ量化模型（更快更小）
    python run_video_analysis_fast.py -i inputs/IMG_7817.MOV -o outputs --model-size 3b-awq

    # 使用原始模型
    python run_video_analysis_fast.py -i inputs/IMG_7817.MOV -o outputs --model-size 3b

    # 指定本地模型路径
    python run_video_analysis_fast.py -i inputs/IMG_7817.MOV -o outputs -m "./models/Qwen/Qwen2___5-VL-3B-Instruct-AWQ"
"""

import os
import json
import cv2
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
from datetime import timedelta
import argparse


class OptimizedVideoAnalyzer:
    """优化后的视频分析器"""
    
    # 可用模型列表
    MODELS = {
        "3b": "Qwen/Qwen2.5-VL-3B-Instruct",
        "3b-awq": "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
    }
    
    def __init__(self, model_size="3b", use_quantization=False, device="auto"):
        """
        初始化分析器
        
        参数：
            model_size: 模型大小 ("3b", "3b-awq")
            use_quantization: 是否使用AWQ量化
            device: 设备 ("auto", "cpu", "mps")
        """
        self.model_size = model_size
        self.use_quantization = use_quantization
        self.device = device
        self.model = None
        self.processor = None
        
        # 计算下采样后的图像大小
        # 原始1080x1920（竖屏视频）- 8倍下采样后 135x240
        self.downsample_ratio = 8  # 可以设置为4或8
        self.target_size = (135, 240)  # 8倍下采样：1920/8=240宽, 1080/8=135高
    
    def download_model(self, model_name):
        """下载模型（使用阿里云ModelScope）"""
        from modelscope import snapshot_download

        print(f"📥 正在下载模型: {model_name}")
        local_dir = f"./models/{model_name.replace('/', '/')}"

        if os.path.exists(local_dir):
            print(f"✅ 模型已存在: {local_dir}")
            return local_dir

        # 使用阿里云ModelScope下载（国内速度快，不需要登录）
        local_dir = snapshot_download(
            model_name,
            cache_dir="./models"
        )
        print(f"✅ 模型下载完成: {local_dir}")
        return local_dir
    
    def load_model(self, model_path=None):
        """加载模型（优化版）"""
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        
        if model_path is None:
            model_name = self.MODELS[self.model_size]
            model_path = f"./models/{model_name}"
            
            if not os.path.exists(model_path):
                model_path = self.download_model(model_name)
        
        print("=" * 60)
        print(f"📦 正在加载模型: Qwen2.5-VL-{self.model_size.upper()}")
        print(f"⚙️  配置:")
        print(f"   - 量化: {self.use_quantization}")
        print(f"   - 设备: {self.device}")
        print(f"   - 下采样: 1/{self.downsample_ratio}")
        print("=" * 60)
        
        # 根据设备选择dtype
        if self.device == "cpu":
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.bfloat16
        
        # 加载模型
        # 检查是否是AWQ模型
        is_awq_model = "awq" in model_path.lower() or "AWQ" in model_path
        
        if is_awq_model:
            # 尝试加载预量化的AWQ模型
            print("🔄 检测到AWQ预量化模型，尝试加载...")
            try:
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    device_map=self.device,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    use_safetensors=True
                ).eval()
                print("✅ AWQ预量化模型加载成功")
            except Exception as e:
                print(f"❌ AWQ模型加载失败: {str(e)[:150]}")
                print("\n� 解决方案:")
                print("  1. 使用原始3B模型: --model-size 3b")
                print("  2. 使用bitsandbytes动态量化: --model-size 3b --quantize")
                print("  3. 该AWQ模型与Qwen2.5-VL存在兼容性问题（视觉模块特征数不兼容）")
                raise RuntimeError("AWQ模型加载失败，请尝试上述解决方案")
        elif self.use_quantization:
            # 使用bitsandbytes进行4bit量化
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path,
                    quantization_config=quantization_config,
                    device_map=self.device,
                    trust_remote_code=True
                )
                print("✅ 4bit量化加载成功")
            except ImportError:
                print("⚠️  bitsandbytes未安装，使用普通加载")
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    device_map=self.device,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                ).eval()
        else:
            # 尝试使用Flash Attention，如果不支持则回退
            try:
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    device_map=self.device,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2"
                ).eval()
            except (ImportError, ValueError):
                print("⚠️  FlashAttention2不可用，使用默认注意力机制")
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
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
    
    def preprocess_image(self, image, target_size=None):
        """
        图像下采样预处理
        
        参数：
            image: PIL图像
            target_size: 目标尺寸 (height, width)
        
        返回：
            resized_image: 调整后的图像
        """
        if target_size is None:
            h, w = image.height, image.width
            new_h, new_w = h // self.downsample_ratio, w // self.downsample_ratio
            # 确保尺寸是2的倍数（模型要求）
            new_h = (new_h // 2) * 2
            new_w = (new_w // 2) * 2
            target_size = (new_h, new_w)
        
        return image.resize(target_size, Image.LANCZOS)
    
    def extract_frames(self, video_path, frame_interval=5.0, max_frames=100):
        """
        从视频中提取关键帧（带下采样）
        
        参数：
            video_path: 视频文件路径
            frame_interval: 帧采样间隔（秒）
            max_frames: 最大采样帧数
        
        返回：
            frames: 帧列表 [(timestamp, PIL_image), ...]
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
        
        # 获取原始尺寸
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 计算下采样后的尺寸
        new_height = (height // self.downsample_ratio // 2) * 2
        new_width = (width // self.downsample_ratio // 2) * 2
        self.target_size = (new_height, new_width)
        
        print(f"  分辨率: {width}x{height} -> {new_width}x{new_height} (1/{self.downsample_ratio})")
        
        frames = []
        frame_interval_frames = int(fps * frame_interval)
        target_frame = 0
        
        with tqdm(total=min(max_frames, int(total_frames / frame_interval_frames)), desc="提取帧") as pbar:
            while cap.isOpened() and len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if pos >= target_frame:
                    # BGR -> RGB -> PIL
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame_rgb)
                    
                    # 下采样
                    image = image.resize(self.target_size, Image.LANCZOS)
                    
                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    frames.append((timestamp, image))
                    
                    target_frame = pos + frame_interval_frames
                    pbar.update(1)
        
        cap.release()
        print(f"✅ 提取完成，共 {len(frames)} 帧")
        return frames
    
    def analyze_frames_batch(self, frames, prompt=None):
        """
        批量分析多帧（加速版）
        
        参数：
            frames: 帧列表
            prompt: 分析提示词
        
        返回：
            results: 分析结果列表 [(timestamp, description), ...]
        """
        if prompt is None:
            prompt = "描述图片中的动作，如：倒水、拿杯、行走等。简短回答。"
        
        print(f"\n🧠 正在批量分析 {len(frames)} 帧...")
        
        # 构造对话
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }]
        
        text_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        results = []
        
        # 分批处理（每批4帧，避免内存溢出）
        batch_size = 4
        
        for i in tqdm(range(0, len(frames), batch_size), desc="分析帧"):
            batch = frames[i:i+batch_size]
            timestamps = [f[0] for f in batch]
            images = [f[1] for f in batch]
            
            # 批量编码
            inputs = self.processor(
                text=[text_prompt] * len(images),
                images=images,
                return_tensors="pt",
                padding=True
            ).to(self.model.device)
            
            # 批量推理
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=32,  # 减少生成长度
                    do_sample=False,
                    pad_token_id=151643,
                    eos_token_id=151643
                )
            
            # 解析结果
            for j, output in enumerate(outputs):
                response = self.processor.decode(output, skip_special_tokens=True)
                if "assistant" in response:
                    response = response.split("assistant")[-1].strip()
                results.append((timestamps[j], response.strip()))
        
        return results
    
    def analyze_frame_single(self, image, prompt=None):
        """分析单帧"""
        if prompt is None:
            prompt = "描述图片中的动作。简短回答。"
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }]
        
        text_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text_prompt],
            images=[image],
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=151643,
                eos_token_id=151643
            )
        
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()
        
        return response.strip()
    
    def detect_action(self, frame_descriptions, action_keywords, frame_interval=5.0):
        """检测动作时间段"""
        print(f"\n🔍 正在检测动作: {', '.join(action_keywords)}")
        
        action_frames = []
        
        for timestamp, description in frame_descriptions:
            desc_lower = description.lower()
            for keyword in action_keywords:
                if keyword.lower() in desc_lower or keyword in description:
                    action_frames.append((timestamp, description, keyword))
                    break
        
        if not action_frames:
            print("⚠️  未检测到目标动作")
            return []
        
        # 合并连续动作帧（使用2倍帧间隔作为阈值）
        merge_threshold = frame_interval * 2.5
        action_segments = []
        current_segment = [action_frames[0]]
        
        for frame in action_frames[1:]:
            if frame[0] - current_segment[-1][0] <= merge_threshold:
                current_segment.append(frame)
            else:
                start_time = current_segment[0][0]
                end_time = current_segment[-1][0]
                # 扩展时间范围以包含动作的完整过程
                expanded_start = max(0, start_time - frame_interval)
                expanded_end = expanded_start + frame_interval * (len(current_segment) + 1)
                descriptions = [f[1] for f in current_segment]
                keywords = [f[2] for f in current_segment]
                action_segments.append((expanded_start, expanded_end, descriptions, keywords))
                current_segment = [frame]
        
        # 添加最后一个片段
        if current_segment:
            start_time = current_segment[0][0]
            end_time = current_segment[-1][0]
            expanded_start = max(0, start_time - frame_interval)
            expanded_end = expanded_start + frame_interval * (len(current_segment) + 1)
            descriptions = [f[1] for f in current_segment]
            keywords = [f[2] for f in current_segment]
            action_segments.append((expanded_start, expanded_end, descriptions, keywords))
        
        print(f"✅ 检测到 {len(action_segments)} 个动作片段:")
        for i, (start, end, desc, kw) in enumerate(action_segments, 1):
            print(f"  [{i}] {timedelta(seconds=start)} - {timedelta(seconds=end)} (关键词: {kw[0] if kw else 'unknown'})")
        
        return action_segments
    
    def extract_segment(self, video_path, start_time, end_time, output_path, padding=1.0):
        """截取视频片段（使用原始分辨率）"""
        print(f"\n✂️ 正在截取片段: {timedelta(seconds=start_time)} - {timedelta(seconds=end_time)}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
        
        # 添加前后padding
        actual_start = max(0, start_time - padding)
        actual_end = min(total_duration, end_time + padding)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
        print(f"✅ 片段已保存: {output_path}")
    
    def analyze_video(self, video_path, action_keywords, output_dir="outputs", frame_interval=5.0):
        """完整视频分析流程"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 提取帧（下采样）
        frames = self.extract_frames(video_path, frame_interval=frame_interval)
        
        # 2. 批量分析帧
        frame_descriptions = self.analyze_frames_batch(frames)
        
        # 打印帧分析结果
        print("\n📋 帧分析结果:")
        for timestamp, desc in frame_descriptions:
            print(f"  {timestamp:.1f}s: {desc}")
        
        # 保存帧分析结果
        analysis_path = os.path.join(output_dir, "frame_analysis.json")
        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump(frame_descriptions, f, indent=2, ensure_ascii=False)
        
        # 3. 检测动作
        action_segments = self.detect_action(frame_descriptions, action_keywords, frame_interval)
        
        # 4. 截取动作片段（原始分辨率）
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        results = []
        
        for i, (start_time, end_time, descriptions, keywords) in enumerate(action_segments, 1):
            output_path = os.path.join(output_dir, f"{video_name}_action_{i}.mp4")
            self.extract_segment(video_path, start_time, end_time, output_path)
            
            results.append({
                "segment_id": i,
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
                "output_path": output_path,
                "frame_descriptions": descriptions,
                "matched_keywords": keywords
            })
        
        # 保存结果
        results_path = os.path.join(output_dir, "action_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 分析完成！结果已保存至 {output_dir}")
        return results


def main():
    parser = argparse.ArgumentParser(description="Qwen-VL视频动作识别（优化加速版）")
    
    parser.add_argument("-i", "--input", required=True, help="输入视频路径")
    parser.add_argument("-o", "--output", default="outputs", help="输出目录")
    parser.add_argument("-m", "--model-path", default=None, help="模型本地路径")
    parser.add_argument("-k", "--keywords", nargs="+", default=["倒水", "拿水杯"], 
                        help="动作关键词")
    
    # 模型配置
    parser.add_argument("--model-size", default="3b", choices=["3b", "3b-awq"],
                        help="模型大小 (3b: 3B原始模型, 3b-awq: 3B AWQ量化模型)")
    parser.add_argument("--quantize", action="store_true", help="启用4bit量化")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"],
                        help="运行设备")
    
    # 性能配置
    parser.add_argument("--frame-interval", type=float, default=5.0,
                        help="抽帧间隔（秒），默认5秒")
    parser.add_argument("--downsample", type=int, default=8,
                        help="分辨率下采样倍数 (4或8)，默认8")
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = OptimizedVideoAnalyzer(
        model_size=args.model_size,
        use_quantization=args.quantize,
        device=args.device
    )
    analyzer.downsample_ratio = args.downsample
    
    # 加载模型
    analyzer.load_model(args.model_path)
    
    # 分析视频
    results = analyzer.analyze_video(
        video_path=args.input,
        action_keywords=args.keywords,
        output_dir=args.output,
        frame_interval=args.frame_interval
    )
    
    # 打印摘要
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
