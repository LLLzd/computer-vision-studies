import os
import requests
from tqdm import tqdm
import zipfile
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# 🔥 阿里云国内镜像（DIV2K 数据集）
ALIBABA_BASE = "https://isv-data.oss-cn-beijing.aliyuncs.com/ML/DIV2K"

# 数据集配置
datasets = [
    # 高分辨率数据
    ("DIV2K_train_HR", f"{ALIBABA_BASE}/DIV2K_train_HR.zip"),
    ("DIV2K_valid_HR", f"{ALIBABA_BASE}/DIV2K_valid_HR.zip"),
    ("DIV2K_test_HR", f"{ALIBABA_BASE}/DIV2K_test_HR.zip"),
    
    # 低分辨率数据 (X2)
    ("DIV2K_train_LR_bicubic/X2", f"{ALIBABA_BASE}/DIV2K_train_LR_bicubic_X2.zip"),
    ("DIV2K_valid_LR_bicubic/X2", f"{ALIBABA_BASE}/DIV2K_valid_LR_bicubic_X2.zip"),
    ("DIV2K_test_LR_bicubic/X2", f"{ALIBABA_BASE}/DIV2K_test_LR_bicubic_X2.zip"),
    
    # 低分辨率数据 (X3)
    ("DIV2K_train_LR_bicubic/X3", f"{ALIBABA_BASE}/DIV2K_train_LR_bicubic_X3.zip"),
    ("DIV2K_valid_LR_bicubic/X3", f"{ALIBABA_BASE}/DIV2K_valid_LR_bicubic_X3.zip"),
    ("DIV2K_test_LR_bicubic/X3", f"{ALIBABA_BASE}/DIV2K_test_LR_bicubic_X3.zip"),
    
    # 低分辨率数据 (X4)
    ("DIV2K_train_LR_bicubic/X4", f"{ALIBABA_BASE}/DIV2K_train_LR_bicubic_X4.zip"),
    ("DIV2K_valid_LR_bicubic/X4", f"{ALIBABA_BASE}/DIV2K_valid_LR_bicubic_X4.zip"),
    ("DIV2K_test_LR_bicubic/X4", f"{ALIBABA_BASE}/DIV2K_test_LR_bicubic_X4.zip"),
]

# 下载函数
def download_file(url, save_path):
    """下载文件并显示进度"""
    session = requests.Session()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive'
    }
    
    max_retries = 1
    for attempt in range(max_retries):
        try:
            response = session.get(url, stream=True, headers=headers, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            chunk_size = 16384
            
            with open(save_path, 'wb') as f, tqdm(
                desc=os.path.basename(save_path), total=total_size, unit='B', unit_scale=True
            ) as pbar:
                for data in response.iter_content(chunk_size):
                    f.write(data)
                    pbar.update(len(data))
            return True
        except requests.exceptions.RequestException as e:
            print(f"❌ 下载失败 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(2)
    return False

# 解压函数
def extract_zip(zip_path, extract_dir):
    """解压zip文件"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        return True
    except Exception as e:
        print(f"❌ 解压失败: {e}")
        return False

# 主函数
def main():
    """主函数"""
    # 创建数据目录
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # 收集需要下载的任务
    download_tasks = []
    for target_dir, url in datasets:
        full_target_dir = os.path.join(data_dir, target_dir)
        if not os.path.exists(full_target_dir):
            zip_filename = os.path.basename(url)
            zip_path = os.path.join(data_dir, zip_filename)
            download_tasks.append((url, zip_path, full_target_dir))
        else:
            print(f"✅ {target_dir} 目录已存在，跳过下载")
    
    # 并行下载
    if download_tasks:
        max_workers = min(4, multiprocessing.cpu_count())
        print(f"🚀 开始下载数据集（{max_workers}线程并行）...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            def download_task(task):
                url, zip_path, target_dir = task
                print(f"📥 开始下载: {os.path.basename(url)}")
                success = download_file(url, zip_path)
                if success:
                    print(f"📦 开始解压: {os.path.basename(zip_path)}")
                    extract_success = extract_zip(zip_path, data_dir)
                    if extract_success:
                        os.remove(zip_path)
                        print(f"✅ {os.path.basename(url)} 处理完成")
                    else:
                        print(f"❌ {os.path.basename(url)} 解压失败")
                else:
                    print(f"❌ {os.path.basename(url)} 下载失败")
            
            futures = [executor.submit(download_task, task) for task in download_tasks]
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"❌ 任务失败: {e}")
    
    print("\n🎉 数据集下载完成！")
    print("📁 数据已保存到 data/ 目录")

if __name__ == "__main__":
    main()
