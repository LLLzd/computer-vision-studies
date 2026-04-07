"""内存监控脚本

实时监控Mac内存使用情况，帮助调整batch size。
"""

import os
import subprocess
import time
import re

def get_memory_info():
    """获取内存信息"""
    # 使用vm_stat获取详细内存信息
    result = subprocess.run(['vm_stat'], capture_output=True, text=True)
    
    # 解析vm_stat输出
    lines = result.stdout.split('\n')
    page_size = 4096  # macOS默认页面大小
    
    memory_info = {}
    for line in lines:
        if 'page size of' in line:
            match = re.search(r'page size of (\d+)', line)
            if match:
                page_size = int(match.group(1))
        
        match = re.match(r'Pages\s+([^:]+):\s+(\d+)', line)
        if match:
            key = match.group(1).strip()
            value = int(match.group(2)) * page_size / (1024**3)  # 转换为GB
            memory_info[key] = value
    
    # 使用top获取总内存信息
    result = subprocess.run(['top', '-l', '1', '-s', '0'], capture_output=True, text=True)
    for line in result.stdout.split('\n'):
        if 'PhysMem' in line:
            # 解析: PhysMem: 15G used (1952M wired, 5525M compressor), 451M unused.
            match = re.search(r'PhysMem:\s+(\d+)G\s+used.*?(\d+)M\s+unused', line)
            if match:
                used_gb = int(match.group(1))
                unused_mb = int(match.group(2))
                memory_info['total_used_gb'] = used_gb
                memory_info['unused_mb'] = unused_mb
    
    return memory_info

def calculate_batch_size_recommendation(unused_mb):
    """根据可用内存计算推荐的batch size
    
    Args:
        unused_mb: 可用内存（MB）
        
    Returns:
        推荐的batch size
    """
    # 每个样本大约需要的内存（MB）
    # 图像: 512x512x3x4bytes = 3MB
    # Heatmap: 128x128x20x4bytes = 1.3MB
    # Offset: 128x128x2x4bytes = 0.13MB
    # WH: 128x128x2x4bytes = 0.13MB
    # 模型梯度等: 约10MB
    # 总计每样本约15MB
    
    memory_per_sample = 15  # MB
    
    # 保留500MB作为缓冲
    available_memory = max(0, unused_mb - 500)
    
    # 计算推荐batch size
    recommended_batch_size = int(available_memory / memory_per_sample)
    
    # 限制在合理范围内
    return max(1, min(recommended_batch_size, 32))

def main():
    """主函数"""
    print("=" * 80)
    print("📊 Mac Memory Monitor for Deep Learning")
    print("=" * 80)
    
    while True:
        # 获取内存信息
        mem_info = get_memory_info()
        
        # 清屏（可选）
        # os.system('clear')
        
        # 打印内存信息
        print(f"\n{'='*80}")
        print(f"⏰ Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
        print(f"\n💾 Memory Usage:")
        print(f"   Total Used: {mem_info.get('total_used_gb', 0):.1f} GB")
        print(f"   Unused: {mem_info.get('unused_mb', 0):.0f} MB")
        print(f"   Free: {mem_info.get('free', 0):.2f} GB")
        print(f"   Active: {mem_info.get('active', 0):.2f} GB")
        print(f"   Inactive: {mem_info.get('inactive', 0):.2f} GB")
        print(f"   Wired: {mem_info.get('wired down', 0):.2f} GB")
        print(f"   Compressed: {mem_info.get('occupied by compressor', 0):.2f} GB")
        
        # 计算推荐batch size
        unused_mb = mem_info.get('unused_mb', 0)
        recommended_bs = calculate_batch_size_recommendation(unused_mb)
        
        print(f"\n🎯 Batch Size Recommendation:")
        print(f"   Current unused memory: {unused_mb:.0f} MB")
        print(f"   Recommended batch size: {recommended_bs}")
        
        if unused_mb < 500:
            print(f"\n⚠️  WARNING: Low memory! ({unused_mb:.0f} MB unused)")
            print(f"   Consider reducing batch size or closing other applications")
        elif unused_mb < 1000:
            print(f"\n⚠️  CAUTION: Memory is getting low ({unused_mb:.0f} MB unused)")
        else:
            print(f"\n✅ Memory status: Good ({unused_mb:.0f} MB unused)")
        
        # 每5秒更新一次
        time.sleep(5)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped by user")
