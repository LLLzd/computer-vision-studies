import torch 
import argparse 
from PIL import Image 
from torchvision import transforms 
from torchvision.utils import save_image 
import os 

# 导入模块
from config import DEVICE, DEFAULT_CONTENT_IMAGE, DEFAULT_STYLE_MODEL, DEFAULT_OUTPUT_IMAGE, TRANSFER_SIZE
from net import TransformerNetwork

# 图像加载 + 预处理 
def load_image(image_path, size=None): 
    transform = transforms.Compose([ 
        transforms.Resize(size), 
        transforms.CenterCrop(size), 
        transforms.ToTensor(), 
        transforms.Lambda(lambda x: x.mul(255)) 
    ]) 
    image = Image.open(image_path).convert("RGB") 
    image = transform(image).unsqueeze(0) 
    return image.to(DEVICE, torch.float) 

# 风格化主函数 
def stylize(content_path, style_model_path, output_path, image_size=512): 
    # 加载图片 
    content = load_image(content_path, image_size) 
    
    # 加载风格模型 
    style_model = TransformerNetwork().to(DEVICE)
    style_model.load_state_dict(torch.load(style_model_path, map_location=DEVICE))
    style_model.to(DEVICE) 
    style_model.eval() 

    # 推理（生成风格图） 
    with torch.no_grad(): 
        output = style_model(content) 

    # 保存结果 
    output = output.cpu().clamp(0, 255) 
    save_image(output / 255, output_path) 
    print(f"✅ 生成完成！保存到：{output_path}") 

# ====================== 你只需要改这里 ====================== 
if __name__ == "__main__": 
    # 1. 你的照片路径 
    CONTENT_IMAGE = DEFAULT_CONTENT_IMAGE
    
    # 2. 风格模型（选一个） 
    # STYLE_MODEL = "models/mosaic.pth"      # 马赛克 
    # STYLE_MODEL = "models/rain_princess.pth" # 油画公主 
    # STYLE_MODEL = "models/udnie.pth"       # 毕加索 
    STYLE_MODEL = DEFAULT_STYLE_MODEL        # 梵高星空 
    
    # 3. 输出文件名 
    OUTPUT_IMAGE = DEFAULT_OUTPUT_IMAGE 
    
    # 运行 
    stylize(CONTENT_IMAGE, STYLE_MODEL, OUTPUT_IMAGE, image_size=TRANSFER_SIZE) 
