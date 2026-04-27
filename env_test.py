#!/usr/bin/env python3
"""
环境测试脚本，用于验证常用包是否正确导入
"""

import sys
import importlib

# 要测试的包列表
packages_to_test = [
    'torch',           # PyTorch
    'torchvision',     # PyTorch 视觉库
    'numpy',           # 数值计算库
    'matplotlib',      # 绘图库
    'PIL',             # 图像处理库
    'tqdm',            # 进度条库
    'yaml',            # YAML配置文件解析
    'argparse',        # 命令行参数解析
    'os',              # 操作系统接口
    'time',            # 时间处理
    'random',          # 随机数生成
    'collections',     # 集合类
    'json',            # JSON处理
    'glob',            # 文件路径匹配
    'shutil',          # 高级文件操作
]

# 测试函数
def test_import(package_name):
    """测试包是否能正确导入"""
    try:
        importlib.import_module(package_name)
        print(f"✓ 成功导入: {package_name}")
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {package_name} - {e}")
        return False

if __name__ == "__main__":
    print("开始测试环境包导入...")
    print("=" * 50)
    
    # 测试所有包
    success_count = 0
    for pkg in packages_to_test:
        if test_import(pkg):
            success_count += 1
    
    print("=" * 50)
    print(f"测试完成: 成功导入 {success_count}/{len(packages_to_test)} 个包")
    
    # 检查Python版本
    print(f"Python版本: {sys.version}")
    
    # 检查PyTorch版本（如果已导入）
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA设备数: {torch.cuda.device_count()}")
            print(f"当前CUDA设备: {torch.cuda.current_device()}")
    except ImportError:
        pass
    
    print("环境测试完成！")
