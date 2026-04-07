import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']  # 用于显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用于显示负号

# 数据集路径
DATA_DIR = 'data'
JPEGIMAGES_DIR = os.path.join(DATA_DIR, 'VOC2012', 'JPEGImages')
ANNOTATIONS_DIR = os.path.join(DATA_DIR, 'VOC2012', 'Annotations')
OUTPUT_DIR = 'outputs'

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

def analyze_data():
    """分析数据集"""
    # 初始化统计变量
    image_count = 0
    image_sizes = {}
    total_boxes = 0
    no_label_images = 0
    boxes_per_image = {}
    categories = set()
    category_count = {}
    
    # 图像尺寸范围
    min_width = float('inf')
    max_width = 0
    min_height = float('inf')
    max_height = 0
    
    # 获取所有图像路径
    image_paths = [f for f in os.listdir(JPEGIMAGES_DIR) if f.endswith('.jpg')]
    image_count = len(image_paths)
    
    # 遍历所有图像
    for image_name in image_paths:
        # 构建图像和标注路径
        image_path = os.path.join(JPEGIMAGES_DIR, image_name)
        annotation_path = os.path.join(ANNOTATIONS_DIR, image_name.replace('.jpg', '.xml'))
        
        # 检查标注文件是否存在
        if not os.path.exists(annotation_path):
            no_label_images += 1
            continue
        
        # 解析标注
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        # 获取图像尺寸
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        size_key = f"{width}x{height}"
        image_sizes[size_key] = image_sizes.get(size_key, 0) + 1
        
        # 更新尺寸范围
        min_width = min(min_width, width)
        max_width = max(max_width, width)
        min_height = min(min_height, height)
        max_height = max(max_height, height)
        
        # 获取边界框和类别
        boxes = []
        for obj in root.findall('object'):
            cls = obj.find('name').text
            categories.add(cls)
            category_count[cls] = category_count.get(cls, 0) + 1
            boxes.append(cls)
        
        # 统计边界框数量
        box_count = len(boxes)
        total_boxes += box_count
        boxes_per_image[box_count] = boxes_per_image.get(box_count, 0) + 1
    
    return {
        'image_count': image_count,
        'image_sizes': image_sizes,
        'total_boxes': total_boxes,
        'no_label_images': no_label_images,
        'boxes_per_image': boxes_per_image,
        'categories': sorted(list(categories)),
        'category_count': category_count,
        'min_width': min_width,
        'max_width': max_width,
        'min_height': min_height,
        'max_height': max_height
    }

def generate_statistics_table(stats):
    """生成统计表格"""
    # 创建表格数据
    table_data = []
    
    # 基本统计信息
    table_data.append(['项目', '数值'])
    table_data.append(['总图像数', stats['image_count']])
    table_data.append(['有标签图像数', stats['image_count'] - stats['no_label_images']])
    table_data.append(['无标签图像数', stats['no_label_images']])
    table_data.append(['总边界框数', stats['total_boxes']])
    table_data.append(['类别数', len(stats['categories'])])
    table_data.append(['平均边界框数/图像', f"{stats['total_boxes'] / (stats['image_count'] - stats['no_label_images']):.2f}"])
    table_data.append(['', ''])
    
    # 图像尺寸范围
    table_data.append(['图像尺寸范围', '数值'])
    table_data.append(['最小宽度', stats['min_width']])
    table_data.append(['最大宽度', stats['max_width']])
    table_data.append(['最小高度', stats['min_height']])
    table_data.append(['最大高度', stats['max_height']])
    table_data.append(['', ''])
    
    # 图像尺寸分布
    table_data.append(['图像尺寸', '数量'])
    sorted_sizes = sorted(stats['image_sizes'].items(), key=lambda x: x[1], reverse=True)
    for size, count in sorted_sizes[:10]:  # 只显示前10个最常见的尺寸
        table_data.append([size, count])
    if len(sorted_sizes) > 10:
        table_data.append(['其他尺寸', sum([count for _, count in sorted_sizes[10:]])])
    table_data.append(['', ''])
    
    # 边界框数量分布
    table_data.append(['边界框数量/图像', '图像数'])
    sorted_box_counts = sorted(stats['boxes_per_image'].items())
    for box_count, img_count in sorted_box_counts[:10]:  # 只显示前10个
        table_data.append([box_count, img_count])
    if len(sorted_box_counts) > 10:
        table_data.append(['10+', sum([count for _, count in sorted_box_counts[10:]])])
    table_data.append(['', ''])
    
    # 类别分布
    table_data.append(['类别', '数量'])
    sorted_categories = sorted(stats['category_count'].items(), key=lambda x: x[1], reverse=True)
    for category, count in sorted_categories[:15]:  # 只显示前15个最常见的类别
        table_data.append([category, count])
    if len(sorted_categories) > 15:
        table_data.append(['其他类别', sum([count for _, count in sorted_categories[15:]])])
    
    return table_data

def save_statistics_to_txt(stats, table_data):
    """将统计结果保存为txt文件"""
    # 构建输出路径
    output_path = os.path.join(OUTPUT_DIR, 'data_analysis_result.txt')
    
    # 打开文件并写入
    with open(output_path, 'w', encoding='utf-8') as f:
        # 写入标题
        f.write('Pascal VOC 2012 数据集统计\n')
        f.write('=' * 80 + '\n\n')
        
        # 写入表格数据
        for row in table_data:
            if len(row) == 2:
                if row[0]:  # 只要第一列不为空就写入，允许第二列为0
                    f.write(f"{row[0]:<30} {row[1]:<10}\n")
                else:
                    f.write('\n')
        
        # 写入类别列表
        f.write('\n' + '=' * 80 + '\n')
        f.write('类别列表:\n')
        f.write('=' * 80 + '\n')
        categories_str = ', '.join(stats['categories'])
        # 分段写入，避免一行太长
        lines = []
        current_line = ''
        for category in stats['categories']:
            if len(current_line) + len(category) + 2 <= 80:
                current_line += category + ', '
            else:
                lines.append(current_line.rstrip(', '))
                current_line = category + ', '
        if current_line:
            lines.append(current_line.rstrip(', '))
        
        for line in lines:
            f.write(line + '\n')
    
    print(f"Analysis result saved to: {output_path}")

def main():
    """主函数"""
    # 检查数据集是否存在
    if not os.path.exists(JPEGIMAGES_DIR):
        print(f"Error: JPEGImages directory not found at {JPEGIMAGES_DIR}")
        print("Please make sure the Pascal VOC 2012 dataset is downloaded")
        return
    
    if not os.path.exists(ANNOTATIONS_DIR):
        print(f"Error: Annotations directory not found at {ANNOTATIONS_DIR}")
        print("Please make sure the Pascal VOC 2012 dataset is downloaded")
        return
    
    # 分析数据
    print("Analyzing dataset...")
    stats = analyze_data()
    
    # 生成统计表格
    table_data = generate_statistics_table(stats)
    
    # 保存统计结果到txt文件
    save_statistics_to_txt(stats, table_data)
    
    # 打印基本统计信息
    print("\n=== 基本统计信息 ===")
    print(f"总图像数: {stats['image_count']}")
    print(f"有标签图像数: {stats['image_count'] - stats['no_label_images']}")
    print(f"无标签图像数: {stats['no_label_images']}")
    print(f"总边界框数: {stats['total_boxes']}")
    print(f"类别数: {len(stats['categories'])}")
    print(f"平均边界框数/图像: {stats['total_boxes'] / (stats['image_count'] - stats['no_label_images']):.2f}")
    print(f"\n=== 图像尺寸范围 ===")
    print(f"最小宽度: {stats['min_width']}")
    print(f"最大宽度: {stats['max_width']}")
    print(f"最小高度: {stats['min_height']}")
    print(f"最大高度: {stats['max_height']}")
    print(f"\n类别列表: {', '.join(stats['categories'])}")

if __name__ == '__main__':
    main()
