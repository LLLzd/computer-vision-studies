"""
边缘检测模块

功能：
    对RGB图像进行边缘检测，生成二值边缘线图像
    用于EDT（欧几里得距离变换）的预处理

边缘检测算法：
    1. Canny：多尺度边缘检测，输出细边缘（推荐）
    2. Sobel：基于梯度的边缘检测
    3. Laplacian：基于二阶导数的边缘检测
    4. LoG：Laplacian of Gaussian，先平滑再求导
    5. Manual：手动阈值分割

边缘检测结果：
    - 白色线条：检测到的边缘
    - 黑色背景：非边缘区域
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path


def edge_detection_canny(image, low_threshold=50, high_threshold=150):
    """
    Canny边缘检测

    原理：
        1. 高斯平滑：减少噪声影响
        2. 梯度计算：Sobel算子计算梯度幅值和方向
        3. 非极大值抑制：细化边缘，只保留局部最大值
        4. 双阈值处理：区分强边缘和弱边缘
        5. 边缘连接：通过滞后机制连接断开的边缘

    参数：
        image: np.array (H, W, 3) 或 (H, W) - RGB或灰度图像
        low_threshold: int - 低阈值（弱边缘起始）
        high_threshold: int - 高阈值（强边缘起始）

    返回：
        edges: np.array (H, W) - 二值边缘图像，白色为边缘
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Canny边缘检测
    edges = cv2.Canny(gray, low_threshold, high_threshold)

    return edges


def edge_detection_sobel(image, aperture_size=3):
    """
    Sobel边缘检测

    原理：
        使用Sobel算子计算图像梯度：
        - Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]（水平梯度）
        - Gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]（垂直梯度）
        梯度幅值 = sqrt(Gx² + Gy²)

    参数：
        image: np.array - 输入图像
        aperture_size: int - Sobel算子核大小

    返回：
        edges: np.array - 二值边缘图像
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 计算Sobel梯度
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=aperture_size)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=aperture_size)

    # 梯度幅值
    magnitude = np.sqrt(sobelx**2 + sobely**2)

    # 归一化并二值化
    magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
    _, edges = cv2.threshold(magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return edges


def edge_detection_laplacian(image, ksize=3):
    """
    Laplacian边缘检测

    原理：
        使用Laplacian算子（二阶导数）检测边缘：
        - 边缘处二阶导数为零（过零点）
        - ∇²f = ∂²f/∂x² + ∂²f/∂y²

    参数：
        image: np.array - 输入图像
        ksize: int - Laplacian算子核大小

    返回：
        edges: np.array - 二值边缘图像
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 高斯模糊减少噪声
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)

    # 取绝对值并归一化
    laplacian = np.abs(laplacian)
    laplacian = (laplacian / laplacian.max() * 255).astype(np.uint8)

    # 二值化
    _, edges = cv2.threshold(laplacian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return edges


def edge_detection_log(image, sigma=2.0):
    """
    LoG (Laplacian of Gaussian) 边缘检测

    原理：
        1. 先用高斯滤波器平滑图像
        2. 再用Laplacian检测边缘
        等价于直接使用LoG滤波器

    参数：
        image: np.array - 输入图像
        sigma: float - 高斯核标准差

    返回：
        edges: np.array - 二值边缘图像
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 高斯模糊
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)

    # Laplacian
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

    # 归一化
    laplacian = np.abs(laplacian)
    laplacian = (laplacian / (laplacian.max() + 1e-10) * 255).astype(np.uint8)

    # 二值化
    _, edges = cv2.threshold(laplacian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return edges


def edge_detection_manual(image, threshold_low=100, threshold_high=200):
    """
    手动阈值边缘检测

    原理：
        直接使用边缘的灰度值进行阈值分割
        适合边缘有明显灰度差异的图像

    参数：
        image: np.array - 输入图像
        threshold_low: int - 低阈值
        threshold_high: int - 高阈值

    返回：
        edges: np.array - 二值边缘图像
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 高斯模糊
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Canny本身就是双阈值，所以这个函数主要用于手动控制阈值
    edges = cv2.Canny(blurred, threshold_low, threshold_high)

    return edges


def extract_edge_lines(image, method='canny', output_path=None, visualize=True):
    """
    提取图像边缘线

    参数：
        image: np.array - 输入RGB图像
        method: str - 边缘检测方法：'canny', 'sobel', 'laplacian', 'log', 'manual'
        output_path: str - 保存路径
        visualize: bool - 是否可视化

    返回：
        edge_image: np.array - 二值边缘图像
    """
    print(f"边缘检测方法: {method}")
    print(f"输入图像尺寸: {image.shape}")

    # 执行边缘检测
    if method == 'canny':
        edge_image = edge_detection_canny(image)
    elif method == 'sobel':
        edge_image = edge_detection_sobel(image)
    elif method == 'laplacian':
        edge_image = edge_detection_laplacian(image)
    elif method == 'log':
        edge_image = edge_detection_log(image)
    elif method == 'manual':
        edge_image = edge_detection_manual(image)
    else:
        raise ValueError(f"未知边缘检测方法: {method}")

    print(f"边缘像素数量: {np.sum(edge_image > 0)}")

    # 保存结果
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), edge_image)
        print(f"边缘图像已保存到: {output_path}")

    # 可视化
    if visualize:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 原图
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # 灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        axes[1].imshow(gray, cmap='gray')
        axes[1].set_title('Grayscale')
        axes[1].axis('off')

        # 边缘检测结果
        axes[2].imshow(edge_image, cmap='gray')
        axes[2].set_title(f'Edge Detection ({method})')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

    return edge_image


def batch_edge_detection(input_dir, output_dir, method='canny'):
    """
    批量处理图像边缘检测

    参数：
        input_dir: str - 输入目录
        output_dir: str - 输出目录
        method: str - 边缘检测方法
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 获取所有图像文件
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_files = [f for f in input_path.iterdir()
                  if f.suffix.lower() in image_extensions]

    print(f"找到 {len(image_files)} 张图像")

    for img_file in image_files:
        print(f"\n处理: {img_file.name}")
        try:
            image = cv2.imread(str(img_file))
            if image is None:
                print(f"  无法读取图像")
                continue

            # 提取边缘
            edge_image = extract_edge_lines(
                image, method=method,
                output_path=output_path / img_file.name,
                visualize=False
            )

        except Exception as e:
            print(f"  错误: {e}")

    print(f"\n批量处理完成！结果保存在: {output_dir}")


def compare_edge_methods(image):
    """
    对比所有边缘检测方法
    """
    print("=" * 60)
    print("边缘检测算法对比")
    print("=" * 60)

    methods = {
        'Canny': edge_detection_canny,
        'Sobel': edge_detection_sobel,
        'Laplacian': edge_detection_laplacian,
        'LoG': edge_detection_log,
    }

    results = {}
    for name, func in methods.items():
        print(f"\n计算 {name}...")
        result = func(image)
        results[name] = result

    # 可视化对比
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 原图
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    idx = 1
    for name, result in results.items():
        i, j = idx // 3, idx % 3
        axes[i, j].imshow(result, cmap='gray')
        axes[i, j].set_title(name)
        axes[i, j].axis('off')
        idx += 1

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Edge Detection')
    parser.add_argument('--input', '-i', type=str, help='输入图像路径')
    parser.add_argument('--output', '-o', type=str, default='output', help='输出目录')
    parser.add_argument('--method', '-m', type=str, default='canny',
                        choices=['canny', 'sobel', 'laplacian', 'log', 'manual'],
                        help='边缘检测方法')
    parser.add_argument('--batch', '-b', action='store_true', help='批量处理模式')
    parser.add_argument('--compare', '-c', action='store_true', help='对比所有方法')
    parser.add_argument('--low', type=int, default=50, help='低阈值 (Canny/manual)')
    parser.add_argument('--high', type=int, default=150, help='高阈值 (Canny/manual)')

    args = parser.parse_args()

    if args.compare:
        # 创建测试图像并对比
        H, W = 300, 300
        test_image = np.zeros((H, W, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (50, 50), (250, 250), (255, 255, 255), 2)
        cv2.circle(test_image, (150, 150), 80, (255, 255, 255), 2)
        cv2.line(test_image, (50, 150), (250, 150), (255, 255, 255), 2)
        compare_edge_methods(test_image)

    elif args.batch:
        batch_edge_detection('input', args.output, args.method)

    elif args.input:
        image = cv2.imread(args.input)
        if image is None:
            print(f"无法读取图像: {args.input}")
            exit(1)

        if args.method == 'manual':
            edge_image = extract_edge_lines(
                image, method=args.method,
                output_path=f"{args.output}/edge_output.png",
                visualize=True
            )
        else:
            edge_image = extract_edge_lines(
                image, method=args.method,
                output_path=f"{args.output}/edge_output.png",
                visualize=True
            )

    else:
        print("边缘检测 项目")
        print("=" * 50)
        print("使用方法:")
        print("  python edge_line.py -i input/image.png -o output/")
        print("  python edge_line.py -b  # 批量处理")
        print("  python edge_line.py -c  # 对比所有方法")
        print()
        print("可用方法:")
        print("  canny     - Canny边缘检测（推荐）")
        print("  sobel     - Sobel边缘检测")
        print("  laplacian - Laplacian边缘检测")
        print("  log       - LoG边缘检测")
        print("  manual    - 手动阈值")
