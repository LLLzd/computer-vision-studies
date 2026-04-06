import os
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 导入配置和模型
from config import DEVICE, UPSCALE_FACTOR, NUM_CHANNELS, MODEL_PATH, OUTPUT_DIR, MODEL_NAME
from models import ESPCN, EDSR

# ===================== 1. 辅助函数：单次超分 =====================
def single_sr(model, img):
    """执行单次超分辨率
    Args:
        model: ESPCN模型
        img: PIL图像
    Returns:
        超分辨率图像
    """
    # 转换为tensor
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)  # 添加batch维度
    
    # 前向传播：生成超分辨率图像
    sr_tensor = model(img_tensor)
    
    # 转换回numpy数组
    sr_np = sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    sr_np = np.clip(sr_np, 0, 1)
    sr_np = (sr_np * 255).astype(np.uint8)
    
    # 转换为PIL图像
    return Image.fromarray(sr_np)

# ===================== 2. 推理函数 =====================
def inference(input_path, output_dir=None, iterations=1):
    """对输入图像进行超分辨率推理
    Args:
        input_path: 输入图像路径或目录
        output_dir: 输出目录
        iterations: 迭代次数，默认1次（3x），2次为9x
    """
    # 设置默认输出目录
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, "inference")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算总超分倍数
    total_scale = UPSCALE_FACTOR ** iterations
    print(f"📦 准备进行 {total_scale}x 超分（{iterations}次 {UPSCALE_FACTOR}x 迭代）")
    
    # 加载模型
    print(f"📦 加载模型: {MODEL_PATH}")
    if MODEL_NAME == 'espcn':
        model = ESPCN(UPSCALE_FACTOR, NUM_CHANNELS).to(DEVICE)
    elif MODEL_NAME == 'edsr':
        model = EDSR(UPSCALE_FACTOR, NUM_CHANNELS).to(DEVICE)
    else:
        raise ValueError(f"不支持的模型名称: {MODEL_NAME}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # 获取输入图像列表
    if os.path.isfile(input_path):
        image_paths = [input_path]
    elif os.path.isdir(input_path):
        image_paths = [os.path.join(input_path, f) for f in os.listdir(input_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    else:
        print(f"❌ 输入路径不存在: {input_path}")
        return
    
    print(f"📊 找到 {len(image_paths)} 张图像")
    
    # 处理每张图像
    with torch.no_grad():
        for idx, img_path in enumerate(image_paths, 1):
            print(f"\n处理 [{idx}/{len(image_paths)}]: {os.path.basename(img_path)}")
            
            # 读取图像
            img = Image.open(img_path).convert("RGB")
            original_size = img.size  # (width, height)
            
            # 迭代超分
            current_img = img
            for i in range(iterations):
                print(f"   第 {i+1} 次 {UPSCALE_FACTOR}x 超分...")
                current_img = single_sr(model, current_img)
            
            sr_img = current_img
            
            # 保存结果
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # 保存原始图片（input）
            input_path = os.path.join(output_dir, f"{img_name}_input.png")
            # 在图片上添加尺寸信息
            plt.figure(figsize=(10, 7))
            plt.imshow(img)
            plt.title(f"输入图像 (尺寸: {original_size[0]}x{original_size[1]})")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(input_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ 原始图片已保存: {input_path}")
            
            # 保存超分结果
            sr_path = os.path.join(output_dir, f"{img_name}_{MODEL_NAME}_sr_{total_scale}x.png")
            # 在图片上添加尺寸信息
            plt.figure(figsize=(10, 7))
            plt.imshow(sr_img)
            plt.title(f"{MODEL_NAME.upper()} 超分结果 ({total_scale}x) (尺寸: {sr_img.size[0]}x{sr_img.size[1]})")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(sr_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ 超分结果已保存: {sr_path}")
            print(f"   原始尺寸: {original_size[0]}x{original_size[1]}")
            print(f"   超分尺寸: {sr_img.size[0]}x{sr_img.size[1]}")
    
    print(f"\n✅ 所有图像处理完成！结果保存在: {output_dir}")

# ===================== 2. 主函数 =====================
def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='超分辨率推理')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='输入图像路径或目录')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='输出目录 (默认: outputs/inference)')
    parser.add_argument('--iterations', '-n', type=int, default=1,
                       help=f'迭代次数，默认1次（{UPSCALE_FACTOR}x），2次为{UPSCALE_FACTOR**2}x')
    
    args = parser.parse_args()
    
    print("🚀 开始超分辨率推理...")
    inference(args.input, args.output, args.iterations)

if __name__ == "__main__":
    main()
