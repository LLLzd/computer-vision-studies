# -*- coding: utf-8 -*-
"""
Qwen2.5-VL-3B-Instruct 图片结构化描述脚本（Apple M1 16G 专用优化版）
功能：本地加载多模态大模型，对图片进行强语义、结构化、JSON 格式描述
项目路径：~/Download/workspace/study/qwenvl_project
模型：Qwen/Qwen2.5-VL-3B-Instruct
"""

# ===================== 1. 导入依赖库 =====================
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
import argparse
import glob
import json
import os

# ===================== 2. 全局配置（可直接修改） =====================
# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 模型路径
MODEL_PATH = os.path.join(PROJECT_ROOT, "models/Qwen/Qwen2___5-VL-3B-Instruct")

# 图片目录
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# 输出结果保存目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 指令：强语义 + 严格 JSON 输出（优化版提示词，准确率更高）
PROMPT = """
你是专业的图像分析助手，擅长对图片进行精细、全面的语义理解和结构化描述。

请对这张图片进行详细分析，并输出标准 JSON 格式的结构化描述。

要求：
1. 只输出标准 JSON，不要任何多余文字、解释、备注
2. JSON 必须包含以下字段：
   - scene: 图片场景（如：室内、户外、办公室、厨房、街道等）
   - main_subject: 图片核心主体
   - objects: 图片中所有物体列表（数组格式，每个物体包含名称和描述）
   - text_content: 图片中的所有文字内容（OCR，无则为空数组）
   - colors: 图片主色调（数组格式）
   - layout: 物品布局描述
   - key_details: 关键细节（清晰度、状态、特征等）
   - description: 整体一句话总结
   - sentiment: 图片传达的情感或氛围
   - quality: 图片质量评估
3. 描述要详细、准确，体现专业的图像分析能力
4. 确保 JSON 格式正确，可直接被解析
"""

# ===================== 3. M1 芯片优化配置（核心优化） =====================
# M1 专用精度，大幅节省内存，提升速度
TORCH_DTYPE = torch.bfloat16

# 自动使用 Apple GPU (MPS) 加速，无需手动配置
DEVICE = "auto"

# 输入图片统一降采样到此固定尺寸 (宽, 高)
TARGET_SIZE = (1280, 960)

# 图片最大像素数（Qwen2.5-VL 官方默认 1280×28×28）
# 不限制时，高分辨率截图会被原尺寸编码，vision token 爆炸，M1 上会报 Invalid buffer size
MAX_PIXELS = 1280 * 28 * 28
MIN_PIXELS = 256 * 28 * 28

# 生成参数（保证输出稳定、结构化、不胡编）
GENERATE_CONFIG = {
    "max_new_tokens": 1024,    # 最大生成长度
    "do_sample": False,        # 关闭随机采样，保证结果一致
    "use_cache": True,         # 加速推理
    "pad_token_id": 151643,    # Qwen 模型专用结束符
    "eos_token_id": 151643
}

# ===================== 4. 加载模型与处理器（只加载一次，最耗时） =====================
def load_model():
    """加载模型和处理器"""
    print("=" * 60)
    print(f"📦 正在加载模型...")
    print(f"⚙️  运行设备：Apple M1 + GPU 加速")
    print("=" * 60)
    
    # 加载模型（低内存模式 + M1 优化）
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=TORCH_DTYPE,
        device_map=DEVICE,
        low_cpu_mem_usage=True,      # 低内存模式，M1 16G 必备
        trust_remote_code=True       # 允许加载自定义模型代码
    )
    
    # 加载处理器（负责图片编码 + 文本编码）
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        max_pixels=MAX_PIXELS,
        min_pixels=MIN_PIXELS,
    )
    
    print("✅ 模型加载完成！")
    return model, processor


def preprocess_image(image):
    """降采样到固定尺寸：保持宽高比缩放，居中贴到 TARGET_SIZE 画布"""
    target_w, target_h = TARGET_SIZE
    src_w, src_h = image.size

    scale = min(target_w / src_w, target_h / src_h, 1.0)  # 只降采样，不放大小图
    new_w = max(1, round(src_w * scale))
    new_h = max(1, round(src_h * scale))
    image = image.resize((new_w, new_h), Image.LANCZOS)

    if (new_w, new_h) == (target_w, target_h):
        return image

    canvas = Image.new("RGB", TARGET_SIZE, (0, 0, 0))
    canvas.paste(image, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return canvas

# ===================== 5. 处理单张图片 =====================
def process_image(image_path, model, processor):
    """处理单张图片并生成结构化描述"""
    print(f"\n🖼️  正在处理图片：{os.path.basename(image_path)}")
    
    try:
        # 打开图片并转为 RGB 格式（避免透明通道报错）
        image = Image.open(image_path).convert("RGB")
        print(f"   原始尺寸：{image.size[0]}×{image.size[1]}")

        image = preprocess_image(image)
        print(f"   预处理后：{image.size[0]}×{image.size[1]}")
        
        # 构造对话指令（符合官方模板）
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},          # 传入图片
                    {"type": "text", "text": PROMPT},  # 传入指令
                ]
            }
        ]
        
        # 应用官方聊天模板（必须这样写，否则模型输出异常）
        text_prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 模型推理（图片 + 文本输入）
        print("🚀 正在进行图像语义分析...")
        
        # 把图片和文本转为模型可识别的张量
        inputs = processor(
            text=[text_prompt],
            images=[image],
            return_tensors="pt"
        ).to(model.device)  # 自动放到 GPU/CPU

        vision_tokens = inputs["pixel_values"].shape[0]
        print(f"   编码后 vision tokens：{vision_tokens}")
        
        # 关闭梯度计算，大幅节省内存 + 加速
        with torch.no_grad():
            outputs = model.generate(**inputs, **GENERATE_CONFIG)
        
        # 解析输出结果
        response = processor.decode(outputs[0], skip_special_tokens=True)
        
        print("\n" + "=" * 60)
        print("📝 模型原始输出：")
        print("-" * 60)
        print(response)
        print("=" * 60)
        
        # 自动解析 JSON 并保存
        try:
            # 清理输出：移除 Markdown 代码块标记和对话历史
            cleaned_response = response
            
            # 移除 assistant 前缀
            if "assistant" in cleaned_response:
                cleaned_response = cleaned_response.split("assistant")[-1].strip()
            
            # 移除 Markdown 代码块标记
            cleaned_response = cleaned_response.replace("```json", "").replace("```", "").strip()
            
            # 移除 JSON 中的注释（JSON 不支持注释）
            import re
            cleaned_response = re.sub(r'//.*', '', cleaned_response)
            
            # 移除多余的空格和换行
            cleaned_response = ' '.join(cleaned_response.split())
            
            # 尝试把输出转为 JSON 对象
            result_json = json.loads(cleaned_response)
            
            # 美化打印
            print("\n✅ 结构化 JSON 解析成功：")
            print(json.dumps(result_json, indent=2, ensure_ascii=False))
            
            # 生成输出文件名
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(OUTPUT_DIR, f"{base_name}_analysis.json")
            
            # 自动保存到项目目录
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result_json, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 结果已保存至：{output_path}")
            return True
            
        except json.JSONDecodeError as e:
            print(f"\n⚠️  输出不是纯 JSON，已打印原始内容: {str(e)}")
            return False
            
    except Exception as e:
        print(f"\n❌ 处理图片时出错：{str(e)}")
        return False

# ===================== 6. 主函数 =====================
IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]


def resolve_image_path(path):
    """解析图片路径，支持相对项目根目录的路径"""
    if os.path.isabs(path):
        return os.path.abspath(path)
    from_cwd = os.path.abspath(path)
    if os.path.isfile(from_cwd):
        return from_cwd
    from_project = os.path.join(PROJECT_ROOT, path)
    if os.path.isfile(from_project):
        return os.path.abspath(from_project)
    return from_cwd


def collect_images_from_dir(directory):
    """收集目录下的所有图片"""
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(image_files)


def main():
    """主函数：指定 --image 时只处理该图片，否则批量处理 data 目录"""
    parser = argparse.ArgumentParser(description="Qwen2.5-VL 图片结构化分析")
    parser.add_argument(
        "--image", "-i",
        nargs="+",
        help="要分析的图片路径（可指定多张，如 inputs/desktop.jpg）",
    )
    args = parser.parse_args()

    if args.image:
        image_files = [resolve_image_path(p) for p in args.image]
        missing = [p for p in image_files if not os.path.isfile(p)]
        if missing:
            print("\n❌ 找不到以下图片文件：")
            for p in missing:
                print(f"   - {p}")
            return
    else:
        image_files = collect_images_from_dir(DATA_DIR)
        if not image_files:
            print(f"\n❌ 未指定 --image，且在 {DATA_DIR} 目录下未找到图片文件")
            print("   示例：python run_image_analysis.py --image inputs/desktop.jpg")
            return

    # 加载模型
    model, processor = load_model()

    print(f"\n📁 找到 {len(image_files)} 张图片待处理：")
    for img_file in image_files:
        print(f"   - {os.path.basename(img_file)}")
    
    # 处理每张图片
    print("\n" + "=" * 60)
    print("🔄 开始批量处理图片...")
    print("=" * 60)
    
    success_count = 0
    for img_file in image_files:
        if process_image(img_file, model, processor):
            success_count += 1
    
    # 统计结果
    print("\n" + "=" * 60)
    print(f"📊 处理完成！")
    print(f"✅ 成功处理：{success_count} 张")
    print(f"❌ 失败处理：{len(image_files) - success_count} 张")
    print(f"📁 结果保存在：{OUTPUT_DIR}")
    print("=" * 60)

# ===================== 7. 执行主函数 =====================
if __name__ == "__main__":
    main()