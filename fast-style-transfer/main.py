#!/usr/bin/env python3
"""
Fast Style Transfer Implementation

基于PyTorch实现的快速风格迁移模型，使用VGG19作为特征提取网络，
可以将任意内容图像转换为指定风格的图像。

Usage:
    # 训练模型
    python main.py train --style images/style/starry_night.jpg
    
    # 风格迁移
    python main.py transfer --content images/content/cat.jpg
"""

import os
import argparse
import torch
from torch.optim import Adam
from tqdm import tqdm

# 导入模块
from config import (
    DEVICE, CONTENT_DIR, STYLE_DIR, OUTPUT_DIR, MODEL_DIR,
    IMAGE_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    CONTENT_WEIGHT, STYLE_WEIGHT, TV_WEIGHT
)
from net import TransformerNetwork, VGG19Features, ContentLoss, StyleLoss, TotalVariationLoss
from dataset import get_dataloader
from data_visual import visualize_images

# 图像预处理
def get_transforms(image_size=256):
    """获取图像预处理变换"""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# 反预处理（用于显示和保存图像）
def denormalize(tensor):
    """反标准化图像张量"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(DEVICE)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(DEVICE)
    return tensor * std + mean

# 保存图像
def save_image(tensor, path):
    """保存图像张量到文件"""
    from torchvision import transforms
    tensor = denormalize(tensor)
    tensor = torch.clamp(tensor, 0, 1)
    image = transforms.ToPILImage()(tensor)
    image.save(path)

# 训练函数
def train(args):
    """训练风格迁移模型"""
    print(f"Using device: {DEVICE}")
    
    # 加载风格图像
    from PIL import Image
    style_transform = get_transforms()
    style_image = Image.open(args.style).convert('RGB')
    style_tensor = style_transform(style_image).unsqueeze(0).to(DEVICE)
    
    # 创建数据加载器
    content_loader = get_dataloader(CONTENT_DIR, BATCH_SIZE, IMAGE_SIZE)
    
    # 初始化模型
    transformer = TransformerNetwork().to(DEVICE)
    vgg = VGG19Features().to(DEVICE)
    
    # 计算风格图像的特征
    style_features = vgg(style_tensor)
    
    # 定义损失函数
    content_criterion = ContentLoss()
    style_criterion = StyleLoss()
    tv_criterion = TotalVariationLoss()
    
    # 定义优化器
    optimizer = Adam(transformer.parameters(), lr=LEARNING_RATE)
    
    # 训练循环
    for epoch in range(EPOCHS):
        total_loss = 0
        
        with tqdm(total=len(content_loader), desc=f"Epoch [{epoch+1}/{EPOCHS}]") as pbar:
            for batch_idx, content_images in enumerate(content_loader):
                content_images = content_images.to(DEVICE)
                
                # 生成风格化图像
                styled_images = transformer(content_images)
                
                # 计算特征
                content_features = vgg(content_images)
                styled_features = vgg(styled_images)
                
                # 计算损失
                content_loss = content_criterion(content_features[3], styled_features[3])  # relu4_3
                style_loss = style_criterion(style_features, styled_features)
                tv_loss = tv_criterion(styled_images)
                
                # 总损失
                loss = (CONTENT_WEIGHT * content_loss + 
                        STYLE_WEIGHT * style_loss + 
                        TV_WEIGHT * tv_loss)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # 更新进度条
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                pbar.update(1)
        
        # 保存模型
        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            style_name = os.path.splitext(os.path.basename(args.style))[0]
            model_path = os.path.join(MODEL_DIR, f'{style_name}_model_epoch_{epoch+1}.pth')
            torch.save(transformer.state_dict(), model_path)
            print(f"Model saved to {model_path}")
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] Average Loss: {total_loss / len(content_loader):.4f}")

# 风格迁移函数
def transfer(args):
    """执行风格迁移"""
    from config import DEFAULT_CONTENT_IMAGE, DEFAULT_STYLE_MODEL, DEFAULT_OUTPUT_IMAGE, TRANSFER_SIZE
    
    # 使用默认值或命令行参数
    content_path = args.content or DEFAULT_CONTENT_IMAGE
    style_model_path = args.model or DEFAULT_STYLE_MODEL
    output_path = args.output or DEFAULT_OUTPUT_IMAGE
    
    print(f"Using device: {DEVICE}")
    print(f"Content image: {content_path}")
    print(f"Style model: {style_model_path}")
    print(f"Output path: {output_path}")
    
    # 加载模型
    transformer = TransformerNetwork().to(DEVICE)
    transformer.load_state_dict(torch.load(style_model_path, map_location=DEVICE))
    transformer.eval()
    
    # 加载内容图像
    from torchvision import transforms
    from PIL import Image
    transform = transforms.Compose([
        transforms.Resize((TRANSFER_SIZE, TRANSFER_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    content_image = Image.open(content_path).convert('RGB')
    content_tensor = transform(content_image).unsqueeze(0).to(DEVICE)
    
    # 生成风格化图像
    with torch.no_grad():
        styled_tensor = transformer(content_tensor)
    
    # 保存结果
    save_image(styled_tensor.squeeze(0), output_path)
    print(f"Style transfer completed! Result saved to {output_path}")

# 主函数
def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Fast Style Transfer')
    
    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='Train a style transfer model')
    train_parser.add_argument('--style', type=str, required=True, help='Path to style image')
    
    # 风格迁移命令
    transfer_parser = subparsers.add_parser('transfer', help='Apply style transfer to an image')
    transfer_parser.add_argument('--content', type=str, help='Path to content image')
    transfer_parser.add_argument('--model', type=str, help='Path to trained model')
    transfer_parser.add_argument('--output', type=str, help='Path to save output image')
    
    # 可视化命令
    visualize_parser = subparsers.add_parser('visualize', help='Visualize images')
    
    args = parser.parse_args()
    
    # 执行命令
    if args.command == 'train':
        train(args)
    elif args.command == 'transfer':
        transfer(args)
    elif args.command == 'visualize':
        visualize_images(CONTENT_DIR, OUTPUT_DIR)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
