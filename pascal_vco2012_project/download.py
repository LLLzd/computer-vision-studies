"""
Pascal VOC 2012 数据集下载脚本
用于下载完整的 Pascal VOC 2012 数据集，包括分割标注

数据集信息：
- 包含 JPEGImages、Annotations、SegmentationClass、SegmentationObject 等目录
- 约 1.9GB（解压后 ~2GB）
- 使用国内镜像加速下载
"""

# 导入必要的库
import os  # 用于文件路径操作
import zipfile  # 用于解压 zip 文件
import requests  # 用于下载文件
from tqdm import tqdm  # 用于显示下载进度

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据目录
DATA_DIR = os.path.join(ROOT_DIR, 'data')

# 确保数据目录存在
os.makedirs(DATA_DIR, exist_ok=True)


def check_dataset_exists():
    """
    检查数据集是否已存在
    
    Returns:
        bool: 如果数据集已存在返回 True，否则返回 False
    """
    # 检查 VOC2012 目录是否存在
    voc_dir = os.path.join(DATA_DIR, "VOC2012")
    
    # 检查必要的子目录
    required_dirs = [
        "JPEGImages",
        "Annotations",
        "SegmentationClass",
        "SegmentationObject",
        "ImageSets"
    ]
    
    # 检查所有必要目录是否存在
    for dir_name in required_dirs:
        dir_path = os.path.join(voc_dir, dir_name)
        if not os.path.exists(dir_path):
            return False
    
    return True


def download_file(url, save_path):
    """
    下载文件并显示进度
    
    Args:
        url: 下载链接
        save_path: 保存路径
    """
    print(f"开始下载: {url}")
    print(f"保存到: {save_path}")
    
    # 发送请求
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    # 显示进度条
    with open(save_path, 'wb') as file, tqdm(
        desc=os.path.basename(save_path),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
    
    print(f"下载完成: {save_path}")


def extract_zip(zip_path, extract_path):
    """
    解压 zip 文件
    
    Args:
        zip_path: zip 文件路径
        extract_path: 解压路径
    """
    print(f"开始解压: {zip_path}")
    print(f"解压到: {extract_path}")
    
    # 解压文件
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    print(f"解压完成: {extract_path}")


def verify_dataset():
    """
    验证数据集结构是否完整
    
    Returns:
        bool: 如果数据集结构完整返回 True，否则返回 False
    """
    # 检查 VOC2012 目录
    voc_dir = os.path.join(DATA_DIR, "VOC2012")
    
    # 检查必要的子目录
    required_dirs = [
        "JPEGImages",
        "Annotations",
        "SegmentationClass",
        "SegmentationObject",
        "ImageSets"
    ]
    
    print("\n验证数据集结构:")
    all_exist = True
    
    for dir_name in required_dirs:
        dir_path = os.path.join(voc_dir, dir_name)
        if os.path.exists(dir_path):
            print(f"✓ {dir_name} 存在")
        else:
            print(f"✗ {dir_name} 不存在")
            all_exist = False
    
    return all_exist


def main():
    """
    主下载函数
    检查数据集是否已存在，不存在则下载并解压
    """
    print("开始处理 Pascal VOC 2012 数据集...")
    
    # 检查数据集是否已存在
    if check_dataset_exists():
        print("✅ 数据集已存在，跳过下载")
        # 验证数据集结构
        verify_dataset()
        print("\n🎉 任务完成！")
        return
    
    print("数据集不存在，开始下载...")
    
    # 下载链接（国内镜像）
    download_url = "https://data.deepai.org/PascalVOC2012.zip"
    
    # 下载文件保存路径
    zip_path = os.path.join(DATA_DIR, "PascalVOC2012.zip")
    
    try:
        # 下载文件
        download_file(download_url, zip_path)
        
        # 解压文件
        extract_zip(zip_path, DATA_DIR)
        
        # 验证数据集结构
        is_valid = verify_dataset()
        
        if is_valid:
            print("\n🎉 数据集下载完成！")
        else:
            print("\n❌ 数据集下载不完整，请重新运行脚本")
    
    finally:
        # 清理 zip 文件
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print(f"\n清理临时文件: {zip_path}")


if __name__ == "__main__":
    # 调用主函数
    main()
