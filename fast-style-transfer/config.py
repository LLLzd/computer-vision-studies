import yaml
import os
import torch

def load_config(config_path='config.yaml'):
    """加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        config: 配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 处理设备配置
    if config['DEVICE'] == 'auto':
        if torch.backends.mps.is_available():
            config['DEVICE'] = 'mps'
        elif torch.cuda.is_available():
            config['DEVICE'] = 'cuda'
        else:
            config['DEVICE'] = 'cpu'
    
    # 确保目录存在
    os.makedirs(config['OUTPUT_DIR'], exist_ok=True)
    os.makedirs(config['MODEL_DIR'], exist_ok=True)
    os.makedirs(config['CONTENT_DIR'], exist_ok=True)
    os.makedirs(config['STYLE_DIR'], exist_ok=True)
    
    return config

# 加载配置
config = load_config()

# 导出配置变量
DATA_DIR = config['DATA_DIR']
CONTENT_DIR = config['CONTENT_DIR']
STYLE_DIR = config['STYLE_DIR']
OUTPUT_DIR = config['OUTPUT_DIR']
MODEL_DIR = config['MODEL_DIR']

IMAGE_SIZE = config['IMAGE_SIZE']
TRANSFER_SIZE = config['TRANSFER_SIZE']

TRANSFORMER_MODEL = config['TRANSFORMER_MODEL']
VGG_MODEL = config['VGG_MODEL']

BATCH_SIZE = config['BATCH_SIZE']
EPOCHS = config['EPOCHS']
LEARNING_RATE = config['LEARNING_RATE']

CONTENT_WEIGHT = config['CONTENT_WEIGHT']
STYLE_WEIGHT = config['STYLE_WEIGHT']
TV_WEIGHT = config['TV_WEIGHT']

DEFAULT_CONTENT_IMAGE = config['DEFAULT_CONTENT_IMAGE']
DEFAULT_STYLE_MODEL = config['DEFAULT_STYLE_MODEL']
DEFAULT_OUTPUT_IMAGE = config['DEFAULT_OUTPUT_IMAGE']

DEVICE = torch.device(config['DEVICE'])
