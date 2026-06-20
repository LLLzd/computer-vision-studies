"""
第一视角视频 Embedding 向量库 — 配置文件
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ---------- 路径 ----------
INPUTS_DIR = BASE_DIR / "inputs"
OUTPUTS_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# 向量库默认输出目录
DEFAULT_INDEX_DIR = OUTPUTS_DIR / "index"
DEFAULT_FRAMES_DIR = OUTPUTS_DIR / "frames"

# ---------- 模型 ----------
# 默认使用 2B 版本，M1 Mac 16G 可运行
MODEL_HF_ID = "Qwen/Qwen3-VL-Embedding-2B"
MODEL_LOCAL_DIR = MODELS_DIR / "Qwen3-VL-Embedding-2B"

# Embedding 维度（2B=2048, 8B=4096）
EMBEDDING_DIM = 2048

# ---------- 视频预处理（第一视角优化） ----------
# 抽帧间隔（秒）：ego 视频动作较快时可设为 0.5~1.0
FRAME_INTERVAL_SEC = 1.0

# 最大抽帧数（防止超长视频爆内存）
MAX_FRAMES = 64

# 输出帧最大边长（像素），降低可节省显存/内存
MAX_FRAME_EDGE = 768

# 是否保存预处理后的帧到磁盘（便于调试）
SAVE_PREPROCESSED_FRAMES = True

# 第一视角常见竖屏比例，可选 center_crop 或 letterbox
RESIZE_MODE = "letterbox"  # "letterbox" | "center_crop" | "stretch"

# ---------- Embedding ----------
# 检索任务 instruction（可按场景自定义）
DEFAULT_INSTRUCTION = (
    "Represent this egocentric video frame for visual retrieval and scene matching."
)

# 文本查询 instruction
QUERY_INSTRUCTION = (
    "Retrieve egocentric video frames relevant to the user's query."
)

# 批大小（M1 建议 1~2，CUDA 可增大）
EMBED_BATCH_SIZE = 1

# ---------- 向量库 ----------
TOP_K_DEFAULT = 5

# 支持的视频/图片扩展名
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
