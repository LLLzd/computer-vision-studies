"""
EDT (Euclidean Distance Transform) 项目

EDT（欧几里得距离变换）简介：
    计算图像中每个像素到最近零像素（背景/边缘）的距离

原理：
    对于二值图像，EDT将每个前景像素的值替换为该像素到最近背景像素的欧几里得距离

应用场景：
    - 计算机图形学：字体渲染、抗锯齿
    - 图像处理：骨架提取、形态学操作
    - 目标检测：锚框生成、中心点检测
    - 路径规划：距离场构建
    - 碰撞检测：空间距离计算

算法方法：
    1. 穷举法：每个像素遍历所有零像素，O(n²m²) 时间复杂度
    2. BFS扩散法：使用队列扩散，每个像素只访问一次，O(nm) 时间复杂度
    3. 扫描线算法（两遍DP）：利用动态规划，O(nm) 时间复杂度
    4. 两步法2D EDT：先水平扫描，再垂直扫描
    5. 基于Voronoi图的算法：更高效但实现复杂
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque


def edt_exhaustive(binary_image):
    """
    穷举法计算EDT（教学用，不推荐实际使用）

    原理：
        对于图像中每个前景像素，计算到所有背景像素的距离，取最小值

    时间复杂度：O(n² × m²)，其中n×m是图像尺寸
    空间复杂度：O(1)
    """
    H, W = binary_image.shape
    distance_map = np.full((H, W), np.inf)

    bg_points = np.argwhere(binary_image == 0)

    for i in range(H):
        for j in range(W):
            if binary_image[i, j] > 0:
                min_dist = np.inf
                for bg in bg_points:
                    dist = np.sqrt((i - bg[0])**2 + (j - bg[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                distance_map[i, j] = min_dist

    return distance_map


def edt_bfs(binary_image):
    """
    BFS扩散法计算EDT（推荐的教学算法）

    原理：
        1. 将所有边缘像素（值为0）加入队列，距离设为0
        2. 从队列中取出像素，向四周扩散
        3. 每个被访问的像素，其距离 = 父像素距离 + 步长
        4. 由于使用BFS（队列），先访问到的路径一定是最短的

    重要：binary中0表示"边缘"，非0表示"内部/前景"
          EDT计算每个非0像素到最近0像素的距离

    特点：
        - 每个像素只访问一次
        - 时间复杂度 O(n × m)
        - 空间复杂度 O(n × m)

    参数：
        binary_image: np.array (H, W) - 二值图像，0为边缘，非0为前景

    返回：
        distance_map: np.array (H, W) - 距离变换结果
    """
    H, W = binary_image.shape
    distance_map = np.full((H, W), np.inf)

    # 初始化队列和距离图
    q = deque()

    # 将所有边缘像素（值为0）加入队列
    # 这些像素的距离为0
    for i in range(H):
        for j in range(W):
            if binary_image[i, j] == 0:
                distance_map[i, j] = 0
                q.append((i, j))

    # 4邻域方向：上、下、左、右
    # 也可以使用8邻域，包含对角方向
    directions_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # 8邻域方向
    directions_8 = [(-1, 0), (1, 0), (0, -1), (0, 1),
                    (-1, -1), (-1, 1), (1, -1), (1, 1)]

    # 使用8邻域可以获得更准确的欧几里得距离
    directions = directions_8

    # BFS扩散
    while q:
        i, j = q.popleft()

        # 当前像素的距离
        current_dist = distance_map[i, j]

        # 向四周扩散
        for di, dj in directions:
            ni, nj = i + di, j + dj

            # 检查边界
            if 0 <= ni < H and 0 <= nj < W:
                # 计算新距离（使用欧几里得度量）
                # 8邻域：对角线距离为sqrt(2)，其他为1
                if abs(di) + abs(dj) == 2:  # 对角线
                    step_dist = np.sqrt(2)
                else:  # 水平或垂直
                    step_dist = 1.0

                new_dist = current_dist + step_dist

                # 只更新距离更小的路径（首次访问时一定是最短的）
                if new_dist < distance_map[ni, nj]:
                    distance_map[ni, nj] = new_dist
                    q.append((ni, nj))

    return distance_map


def edt_bfs_4connected(binary_image):
    """
    BFS扩散法（4邻域版本）

    4邻域BFS的距离是曼哈顿距离，不是真正的欧几里得距离
    用于对比学习

    参数：
        binary_image: np.array (H, W) - 二值图像

    返回：
        distance_map: np.array (H, W) - 曼哈顿距离变换结果
    """
    H, W = binary_image.shape
    distance_map = np.full((H, W), np.inf)

    q = deque()

    # 初始化边缘像素
    for i in range(H):
        for j in range(W):
            if binary_image[i, j] == 0:
                distance_map[i, j] = 0
                q.append((i, j))

    # 4邻域方向
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while q:
        i, j = q.popleft()
        current_dist = distance_map[i, j]

        for di, dj in directions:
            ni, nj = i + di, j + dj

            if 0 <= ni < H and 0 <= nj < W:
                new_dist = current_dist + 1

                if new_dist < distance_map[ni, nj]:
                    distance_map[ni, nj] = new_dist
                    q.append((ni, nj))

    return distance_map


def edt_two_pass(binary_image):
    """
    两遍扫描算法计算EDT（经典算法）

    原理：
        第一遍：从左上到右下扫描，计算到左、上、左上、右上像素的最小距离
        第二遍：从右下到左上扫描，结合右、下、左下、右下像素更新距离

    时间复杂度：O(n × m)
    """
    H, W = binary_image.shape
    INF = 1e10

    distance_map = np.where(binary_image == 0, 0, INF).astype(np.float64)

    # 第一遍：从左上到右下
    for i in range(1, H):
        for j in range(1, W):
            if distance_map[i, j] > 0:
                min_neighbor = min(
                    distance_map[i-1, j],
                    distance_map[i, j-1],
                    distance_map[i-1, j-1],
                    distance_map[i-1, j+1] if j+1 < W else INF
                )
                distance_map[i, j] = min_neighbor + 1

    # 第二遍：从右下到左上
    for i in range(H-2, -1, -1):
        for j in range(W-2, -1, -1):
            if distance_map[i, j] > 0:
                min_neighbor = min(
                    distance_map[i+1, j],
                    distance_map[i, j+1],
                    distance_map[i+1, j+1],
                    distance_map[i+1, j-1] if j-1 >= 0 else INF,
                    distance_map[i, j]
                )
                distance_map[i, j] = min(min_neighbor + 1, distance_map[i, j])

    return distance_map


def edt_scipy(binary_image):
    """
    使用scipy计算EDT（参考实现）

    这是最快最准确的实现
    """
    from scipy.ndimage import distance_transform_edt

    distance_map = distance_transform_edt(binary_image)

    return distance_map


def compute_edt(input_path, output_dir=None, method='bfs', visualize=True):
    """
    计算图像的EDT

    参数：
        input_path: str - 输入图像路径
        output_dir: str - 输出目录，None则不保存
        method: str - 算法选择：'scipy', 'bfs', 'bfs_4', 'two_pass', 'exhaustive'
        visualize: bool - 是否可视化结果

    返回：
        distance_map: np.array - EDT结果
    """
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"无法读取图像: {input_path}")

    print(f"读取图像: {input_path}")
    print(f"图像尺寸: {image.shape}")
    print(f"使用算法: {method}")

    # 转换为二值图像
    # 白色(255)为前景，非白色为背景(0)
    _, binary = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY)

    # 计算EDT
    if method == 'scipy':
        distance_map = edt_scipy(binary)
    elif method == 'bfs':
        distance_map = edt_bfs(binary)
    elif method == 'bfs_4':
        distance_map = edt_bfs_4connected(binary)
    elif method == 'two_pass':
        distance_map = edt_two_pass(binary)
    elif method == 'exhaustive':
        distance_map = edt_exhaustive(binary)
    else:
        raise ValueError(f"未知算法: {method}")

    print(f"距离范围: [{distance_map.min():.2f}, {distance_map.max():.2f}]")

    # 保存结果
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        normalized = (distance_map / distance_map.max() * 255).astype(np.uint8)
        cv2.imwrite(str(output_path / 'edt_output.png'), normalized)

        colored = colormap_edt(distance_map)
        cv2.imwrite(str(output_path / 'edt_colored.png'), colored)

        print(f"结果已保存到: {output_dir}")

    # 可视化
    if visualize:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Input Image')
        axes[0].axis('off')

        axes[1].imshow(distance_map, cmap='gray')
        axes[1].set_title('EDT (Grayscale)')
        axes[1].axis('off')

        axes[2].imshow(colormap_edt(distance_map))
        axes[2].set_title('EDT (Colormap)')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

    return distance_map


def colormap_edt(distance_map, mode='jet'):
    """
    将EDT结果转换为伪彩色图像
    """
    normalized = distance_map / (distance_map.max() + 1e-10)
    cmap = plt.get_cmap(mode)
    colored = cmap(normalized)
    colored = (colored[:, :, :3] * 255).astype(np.uint8)
    return colored


def batch_process(input_dir, output_dir, method='bfs'):
    """
    批量处理目录中的所有图像
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
            compute_edt(str(img_file), str(output_path), method=method, visualize=False)
            import shutil
            shutil.copy(str(img_file), str(output_path / img_file.name))
        except Exception as e:
            print(f"  错误: {e}")

    print(f"\n批量处理完成！结果保存在: {output_dir}")


def compare_algorithms(binary_image):
    """
    对比所有EDT算法的结果
    """
    print("=" * 60)
    print("EDT算法对比")
    print("=" * 60)

    algorithms = {
        'scipy (ground truth)': edt_scipy,
        'BFS 8-neighbor': edt_bfs,
        'BFS 4-neighbor': edt_bfs_4connected,
        'two_pass': edt_two_pass,
    }

    results = {}
    for name, func in algorithms.items():
        print(f"\n计算 {name}...")
        result = func(binary_image)
        results[name] = result

    # 计算与scipy的误差
    ground_truth = results['scipy (ground truth)']
    print("\n与ground truth的误差:")
    for name, result in results.items():
        if name != 'scipy (ground truth)':
            error = np.sqrt(np.mean((ground_truth - result)**2))
            print(f"  {name}: RMSE = {error:.4f}")

    # 可视化对比
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(binary_image, cmap='gray')
    axes[0, 0].set_title('Input Binary Image')
    axes[0, 0].axis('off')

    for idx, (name, result) in enumerate(results.items()):
        i, j = (idx + 1) // 3, (idx + 1) % 3
        axes[i, j].imshow(result, cmap='jet')
        axes[i, j].set_title(name)
        axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='EDT (Euclidean Distance Transform)')
    parser.add_argument('--input', '-i', type=str, help='输入图像路径')
    parser.add_argument('--output', '-o', type=str, default='output', help='输出目录')
    parser.add_argument('--method', '-m', type=str, default='bfs',
                        choices=['scipy', 'bfs', 'bfs_4', 'two_pass', 'exhaustive'],
                        help='EDT算法')
    parser.add_argument('--batch', '-b', action='store_true', help='批量处理模式')
    parser.add_argument('--compare', '-c', action='store_true', help='对比所有算法')

    args = parser.parse_args()

    if args.compare:
        # 创建测试图像并对比算法
        H, W = 200, 200
        test_image = np.zeros((H, W), dtype=np.uint8)
        cv2.circle(test_image, (100, 100), 50, 255, -1)
        cv2.rectangle(test_image, (30, 30), (80, 80), 255, -1)
        _, binary = cv2.threshold(test_image, 127, 1, cv2.THRESH_BINARY)
        compare_algorithms(binary)
    elif args.batch:
        batch_process('input', args.output, args.method)
    elif args.input:
        compute_edt(args.input, args.output, args.method)
    else:
        print("EDT 项目")
        print("=" * 50)
        print("使用方法:")
        print("  python edt.py -i input/image.png -o output/")
        print("  python edt.py -b  # 批量处理")
        print("  python edt.py -c  # 对比所有算法")
        print()
        print("可用算法:")
        print("  scipy     - 使用scipy库（最快最准确）")
        print("  bfs       - BFS扩散法8邻域（推荐）")
        print("  bfs_4     - BFS扩散法4邻域")
        print("  two_pass  - 两遍扫描算法")
        print("  exhaustive - 穷举法（教学用）")
