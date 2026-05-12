"""
RGB图像EDT完整流程

流程：
    1. 读取RGB图像
    2. 边缘检测：使用Canny等算法提取边缘线
    3. EDT计算：计算到边缘线的距离
    4. 可视化结果
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

import edge_line  # 边缘检测
import edt       # EDT计算


def process_rgb_to_edt(input_path, output_dir=None, edge_method='canny', edt_method='bfs'):
    """
    完整的RGB图像到EDT的处理流程

    参数：
        input_path: str - 输入RGB图像路径
        output_dir: str - 输出目录
        edge_method: str - 边缘检测方法
        edt_method: str - EDT计算方法

    返回：
        result: dict - 包含边缘图和EDT结果
    """
    print("=" * 60)
    print("RGB图像 -> EDT 完整流程")
    print("=" * 60)

    # 1. 读取RGB图像
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"无法读取图像: {input_path}")
    print(f"\n步骤1: 读取图像")
    print(f"  尺寸: {image.shape}")

    # 2. 边缘检测
    print(f"\n步骤2: 边缘检测 (method={edge_method})")
    edge_image = edge_line.edge_detection_canny(image)
    print(f"  边缘像素数量: {np.sum(edge_image > 0)}")

    # 3. EDT计算
    print(f"\n步骤3: EDT计算 (method={edt_method})")

    # 正确逻辑：
    # Canny输出：边缘=255(白), 背景=0(黑)
    # EDT计算：0像素的距离=0，非0像素计算到最近0的距离
    #
    # 我们想要：边缘处距离=0，非边缘区域计算到边缘的距离
    # 做法：将Canny边缘(255)变为非0，将Canny背景(0)变为0
    # 这样Canny边缘就变成了EDT的"内部/前景"，Canny背景变成了EDT的"边缘"
    binary = (edge_image == 0).astype(np.uint8)
    # 结果：边缘(255) -> binary=1, 背景(0) -> binary=0
    # 或者更简洁: binary = (edges > 0).astype(np.uint8)

    if edt_method == 'scipy':
        distance_map = edt.edt_scipy(binary)
    elif edt_method == 'bfs':
        distance_map = edt.edt_bfs(binary)
    elif edt_method == 'bfs_4':
        distance_map = edt.edt_bfs_4connected(binary)
    elif edt_method == 'two_pass':
        distance_map = edt.edt_two_pass(binary)
    else:
        distance_map = edt.edt_bfs(binary)

    print(f"  距离范围: [{distance_map.min():.2f}, {distance_map.max():.2f}]")

    # 4. 保存结果
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 保存边缘图
        cv2.imwrite(str(output_path / 'edges.png'), edge_image)

        # 保存EDT灰度图
        normalized = (distance_map / distance_map.max() * 255).astype(np.uint8)
        cv2.imwrite(str(output_path / 'edt_output.png'), normalized)

        # 保存EDT彩色图
        colored = edt.colormap_edt(distance_map)
        cv2.imwrite(str(output_path / 'edt_colored.png'), colored)

        print(f"\n结果已保存到: {output_dir}")

    # 5. 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # 原图
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # 边缘图
    axes[0, 1].imshow(edge_image, cmap='gray')
    axes[0, 1].set_title('Edge Detection')
    axes[0, 1].axis('off')

    # EDT灰度图
    axes[1, 0].imshow(distance_map, cmap='gray')
    axes[1, 0].set_title('EDT (Grayscale)')
    axes[1, 0].axis('off')

    # EDT彩色图
    axes[1, 1].imshow(edt.colormap_edt(distance_map))
    axes[1, 1].set_title('EDT (Colormap)')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

    return {
        'image': image,
        'edges': edge_image,
        'edt': distance_map
    }


def batch_rgb_to_edt(input_dir, output_dir, edge_method='canny', edt_method='bfs'):
    """
    批量处理RGB图像到EDT
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_files = [f for f in input_path.iterdir()
                  if f.suffix.lower() in image_extensions]

    print(f"找到 {len(image_files)} 张图像")

    for img_file in image_files:
        print(f"\n处理: {img_file.name}")
        try:
            # 为每张图像创建子目录
            img_output = output_path / img_file.stem
            img_output.mkdir(exist_ok=True)

            process_rgb_to_edt(
                str(img_file),
                str(img_output),
                edge_method=edge_method,
                edt_method=edt_method
            )

        except Exception as e:
            print(f"  错误: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='RGB Image to EDT')
    parser.add_argument('--input', '-i', type=str, help='输入图像路径')
    parser.add_argument('--output', '-o', type=str, default='output', help='输出目录')
    parser.add_argument('--edge', '-e', type=str, default='canny',
                       choices=['canny', 'sobel', 'laplacian', 'log', 'manual'],
                       help='边缘检测方法')
    parser.add_argument('--edt', '-d', type=str, default='bfs',
                       choices=['scipy', 'bfs', 'bfs_4', 'two_pass'],
                       help='EDT方法')
    parser.add_argument('--batch', '-b', action='store_true', help='批量处理模式')

    args = parser.parse_args()

    if args.batch:
        batch_rgb_to_edt('input', args.output, args.edge, args.edt)
    elif args.input:
        process_rgb_to_edt(args.input, args.output, args.edge, args.edt)
    else:
        print("RGB图像 -> EDT")
        print("=" * 50)
        print("使用方法:")
        print("  python rgb_to_edt.py -i input/image.png -o output/")
        print("  python rgb_to_edt.py -b  # 批量处理")
