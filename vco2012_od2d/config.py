"""配置文件

集中管理所有配置参数，包括数据集路径、模型参数、训练参数等。
"""

import os
import yaml
import argparse

# 全局配置对象
config = {}

# 加载配置文件
def load_config(config_path):
    """加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
    """
    global config
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    
    # 更新全局配置
    config.update(config_data)
    
    # 确保路径存在
    os.makedirs(config.get('OUTPUT_DIR', 'outputs'), exist_ok=True)
    os.makedirs(config.get('WEIGHTS_DIR', 'outputs/weights'), exist_ok=True)
    
    # 计算派生值
    config['CLS_TO_IDX'] = {cls: i for i, cls in enumerate(config['VOC_CLASSES'])}
    config['NUM_CLASSES'] = len(config['VOC_CLASSES'])
    
    # 转换元组类型
    if isinstance(config['IMAGE_SIZE'], list):
        config['IMAGE_SIZE'] = tuple(config['IMAGE_SIZE'])
    if isinstance(config['HEATMAP_SIZE'], list):
        config['HEATMAP_SIZE'] = tuple(config['HEATMAP_SIZE'])
    
    # 转换数值类型
    numeric_keys = ['INITIAL_LR', 'WARMUP_LR', 'MIN_LR', 'HEATMAP_LOSS_WEIGHT', 'OFFSET_LOSS_WEIGHT', 'WH_LOSS_WEIGHT', 'THRESHOLD', 'IOU_THRESHOLD', 'AVG_BOX_WIDTH', 'AVG_BOX_HEIGHT']
    for key in numeric_keys:
        if key in config:
            config[key] = float(config[key])
    
    # 转换整数类型
    integer_keys = ['BATCH_SIZE', 'NUM_WORKERS', 'EPOCHS', 'WARMUP_EPOCHS', 'MAX_DETECTIONS', 'QUICK_TEST_BATCHES', 'QUICK_TEST_EPOCHS']
    for key in integer_keys:
        if key in config:
            config[key] = int(config[key])
    
    # 转换布尔类型
    boolean_keys = ['QUICK_TEST']
    for key in boolean_keys:
        if key in config:
            config[key] = bool(config[key])

# 获取配置值
def get_config(key, default=None):
    """获取配置值
    
    Args:
        key: 配置键
        default: 默认值
    
    Returns:
        配置值
    """
    return config.get(key, default)

# 解析命令行参数
def parse_args():
    """解析命令行参数
    
    Returns:
        命令行参数
    """
    parser = argparse.ArgumentParser(description='Object Detection Training')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    return parser.parse_args()

# 初始化配置
args = parse_args()
load_config(args.config)

# 导出配置变量，保持向后兼容
DATA_DIR = get_config('DATA_DIR', 'data')
JPEGIMAGES_DIR = get_config('JPEGIMAGES_DIR', os.path.join(DATA_DIR, 'VOC2012', 'JPEGImages'))
ANNOTATIONS_DIR = get_config('ANNOTATIONS_DIR', os.path.join(DATA_DIR, 'VOC2012', 'Annotations'))
OUTPUT_DIR = get_config('OUTPUT_DIR', 'outputs')
VOC_CLASSES = get_config('VOC_CLASSES')
CLS_TO_IDX = get_config('CLS_TO_IDX')
NUM_CLASSES = get_config('NUM_CLASSES')
IMAGE_SIZE = get_config('IMAGE_SIZE', (512, 512))
HEATMAP_SIZE = get_config('HEATMAP_SIZE', (128, 128))
INPUT_CHANNELS = get_config('INPUT_CHANNELS', 3)
HIDDEN_CHANNELS = get_config('HIDDEN_CHANNELS', 64)
AVG_BOX_WIDTH = get_config('AVG_BOX_WIDTH', 162.34)
AVG_BOX_HEIGHT = get_config('AVG_BOX_HEIGHT', 203.76)
BATCH_SIZE = get_config('BATCH_SIZE', 2)
NUM_WORKERS = get_config('NUM_WORKERS', 2)
EPOCHS = get_config('EPOCHS', 50)
INITIAL_LR = get_config('INITIAL_LR', 0.0001)
WARMUP_EPOCHS = get_config('WARMUP_EPOCHS', 5)
WARMUP_LR = get_config('WARMUP_LR', 1e-6)
MIN_LR = get_config('MIN_LR', 1e-6)
LR_SCHEDULER = get_config('LR_SCHEDULER', 'cosine')
HEATMAP_LOSS_WEIGHT = get_config('HEATMAP_LOSS_WEIGHT', 1.0)
OFFSET_LOSS_WEIGHT = get_config('OFFSET_LOSS_WEIGHT', 1.0)
WH_LOSS_WEIGHT = get_config('WH_LOSS_WEIGHT', 0.1)
THRESHOLD = get_config('THRESHOLD', 0.5)
IOU_THRESHOLD = get_config('IOU_THRESHOLD', 0.3)
MAX_DETECTIONS = get_config('MAX_DETECTIONS', 20)
WEIGHTS_DIR = get_config('WEIGHTS_DIR', os.path.join(OUTPUT_DIR, 'weights'))
MODEL_PATH = get_config('MODEL_PATH', os.path.join(WEIGHTS_DIR, 'anchor_free_detector.pth'))
QUICK_TEST = get_config('QUICK_TEST', False)
QUICK_TEST_BATCHES = get_config('QUICK_TEST_BATCHES', 10)
QUICK_TEST_EPOCHS = get_config('QUICK_TEST_EPOCHS', 1)
