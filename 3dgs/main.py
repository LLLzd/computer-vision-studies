#!/usr/bin/env python3
"""
3D Gaussian Splatting - 主入口脚本
整合预处理、训练和渲染功能
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """运行命令并显示进度"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    print()
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\nError: {description} failed!")
        sys.exit(1)
    print(f"\n{description} completed!")


def extract_frames(video_path, output_dir, sample_rate=5, max_frames=100, resize=0.5):
    """从视频中提取帧"""
    cmd = f"python preprocess/extract_frames.py --video {video_path} --output {output_dir} --sample_rate {sample_rate} --max_frames {max_frames} --resize {resize}"
    run_command(cmd, "Step 1: Extracting frames from video")


def initialize_gaussians(frames_dir, output_dir, num_gaussians=50000, max_frames=30):
    """初始化高斯"""
    cmd = f"python preprocess/initialize.py --frames {frames_dir} --output {output_dir} --num_gaussians {num_gaussians} --max_frames {max_frames}"
    run_command(cmd, "Step 2: Initializing Gaussians")


def train_gaussians(frames_dir, init_file, output_dir, mode='simple', iterations=100, lr=0.01):
    """训练高斯"""
    if mode == 'simple':
        cmd = f"python train/train_simple.py --frames {frames_dir} --init {init_file} --output {output_dir} --iterations {iterations}"
    elif mode == 'torch':
        cmd = f"python train/train_torch.py --frames {frames_dir} --init {init_file} --output {output_dir} --iterations {iterations} --lr {lr}"
    elif mode == '3dgs':
        cmd = f"python train/train_3dgs.py --frames {frames_dir} --init {init_file} --output {output_dir} --iterations {iterations} --lr {lr}"
    elif mode == 'full':
        cmd = f"python train/train.py --frames {frames_dir} --init {init_file} --output {output_dir} --iterations {iterations}"
    else:
        print(f"Unknown training mode: {mode}")
        sys.exit(1)
    
    run_command(cmd, f"Step 3: Training Gaussians (mode: {mode})")


def render_gaussians(gaussians_file, frames_dir, output_dir, mode='simple', render_mode='360'):
    """渲染高斯"""
    if mode == 'simple':
        cmd = f"python render/render_simple.py --gaussians {gaussians_file} --frames {frames_dir} --output {output_dir} --mode {render_mode}"
    elif mode == 'full':
        cmd = f"python render/render.py --gaussians {gaussians_file} --frames {frames_dir} --output {output_dir} --mode {render_mode}"
    else:
        print(f"Unknown render mode: {mode}")
        sys.exit(1)
    
    run_command(cmd, f"Step 4: Rendering (mode: {render_mode})")


def full_pipeline(video_path, output_dir, train_mode='simple', render_mode='360', 
                  sample_rate=5, max_frames=100, num_gaussians=50000, 
                  iterations=100, lr=0.01, resize=0.5):
    """完整流程：提取帧 -> 初始化 -> 训练 -> 渲染"""
    frames_dir = os.path.join(output_dir, "input/frames")
    gaussians_dir = os.path.join(output_dir, "output")
    
    print("\n" + "="*60)
    print("3D Gaussian Splatting - Full Pipeline")
    print("="*60)
    
    extract_frames(video_path, frames_dir, sample_rate, max_frames, resize)
    initialize_gaussians(frames_dir, gaussians_dir, num_gaussians, max_frames)
    
    init_file = os.path.join(gaussians_dir, "gaussians_init.pkl")
    train_gaussians(frames_dir, init_file, gaussians_dir, train_mode, iterations, lr)
    
    trained_file = os.path.join(gaussians_dir, "gaussians_trained.pkl")
    render_output = os.path.join(gaussians_dir, "renders")
    render_gaussians(trained_file, frames_dir, render_output, train_mode, render_mode)
    
    print("\n" + "="*60)
    print("Full Pipeline Completed!")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Frames: {frames_dir}")
    print(f"  - Gaussians: {gaussians_dir}")
    print(f"  - Renders: {render_output}")


def main():
    parser = argparse.ArgumentParser(
        description="3D Gaussian Splatting - Main Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py extract --video input/videos/IMG_7834.MOV
  python main.py initialize --frames input/frames
  python main.py train --mode simple --iterations 100
  python main.py render --mode simple --render-mode 360
  python main.py full --video input/videos/IMG_7834.MOV --train-mode simple
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    extract_parser = subparsers.add_parser('extract', help='Extract frames from video')
    extract_parser.add_argument('--video', '-v', type=str, required=True, help='Video file path')
    extract_parser.add_argument('--output', '-o', type=str, default='input/frames', help='Output directory')
    extract_parser.add_argument('--sample-rate', '-r', type=int, default=5, help='Sample every N frames')
    extract_parser.add_argument('--max-frames', '-m', type=int, default=100, help='Maximum frames')
    extract_parser.add_argument('--resize', '-s', type=float, default=0.5, help='Resize factor')
    
    init_parser = subparsers.add_parser('initialize', help='Initialize Gaussians from frames')
    init_parser.add_argument('--frames', '-f', type=str, required=True, help='Frames directory')
    init_parser.add_argument('--output', '-o', type=str, default='output', help='Output directory')
    init_parser.add_argument('--num-gaussians', '-n', type=int, default=50000, help='Number of Gaussians')
    init_parser.add_argument('--max-frames', '-m', type=int, default=30, help='Max frames to process')
    
    train_parser = subparsers.add_parser('train', help='Train Gaussians')
    train_parser.add_argument('--frames', '-f', type=str, default='input/frames', help='Frames directory')
    train_parser.add_argument('--init', '-i', type=str, default='output/gaussians_init.pkl', help='Init file')
    train_parser.add_argument('--output', '-o', type=str, default='output', help='Output directory')
    train_parser.add_argument('--mode', '-m', type=str, default='simple', 
                              choices=['simple', 'torch', '3dgs', 'full'],
                              help='Training mode: simple (fast), torch (GPU), 3dgs (full), full (demo)')
    train_parser.add_argument('--iterations', '-n', type=int, default=100, help='Training iterations')
    train_parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    
    render_parser = subparsers.add_parser('render', help='Render Gaussians')
    render_parser.add_argument('--gaussians', '-g', type=str, default='output/gaussians_trained.pkl', help='Gaussians file')
    render_parser.add_argument('--frames', '-f', type=str, default='input/frames', help='Frames directory')
    render_parser.add_argument('--output', '-o', type=str, default='output/renders', help='Output directory')
    render_parser.add_argument('--mode', '-m', type=str, default='simple', 
                              choices=['simple', 'full'],
                              help='Render mode: simple (fast) or full (GPU)')
    render_parser.add_argument('--render-mode', '-r', type=str, default='360',
                              choices=['360', 'compare', '3d', 'interactive'],
                              help='Rendering type: 360 video, comparison, 3D visualization, or interactive')
    
    full_parser = subparsers.add_parser('full', help='Run full pipeline')
    full_parser.add_argument('--video', '-v', type=str, required=True, help='Video file path')
    full_parser.add_argument('--output', '-o', type=str, default='.', help='Output directory')
    full_parser.add_argument('--train-mode', '-t', type=str, default='simple',
                            choices=['simple', 'torch', '3dgs', 'full'],
                            help='Training mode')
    full_parser.add_argument('--render-mode', '-r', type=str, default='360',
                            choices=['360', 'compare', '3d'],
                            help='Rendering mode')
    full_parser.add_argument('--sample-rate', type=int, default=5, help='Sample every N frames')
    full_parser.add_argument('--max-frames', type=int, default=100, help='Maximum frames')
    full_parser.add_argument('--num-gaussians', type=int, default=50000, help='Number of Gaussians')
    full_parser.add_argument('--iterations', type=int, default=100, help='Training iterations')
    full_parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    full_parser.add_argument('--resize', type=float, default=0.5, help='Resize factor')
    
    args = parser.parse_args()
    
    if args.command == 'extract':
        extract_frames(args.video, args.output, args.sample_rate, args.max_frames, args.resize)
    elif args.command == 'initialize':
        initialize_gaussians(args.frames, args.output, args.num_gaussians, args.max_frames)
    elif args.command == 'train':
        train_gaussians(args.frames, args.init, args.output, args.mode, args.iterations, args.lr)
    elif args.command == 'render':
        render_gaussians(args.gaussians, args.frames, args.output, args.mode, args.render_mode)
    elif args.command == 'full':
        full_pipeline(args.video, args.output, args.train_mode, args.render_mode,
                     args.sample_rate, args.max_frames, args.num_gaussians,
                     args.iterations, args.lr, args.resize)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
