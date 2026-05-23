"""
人脸偏好打分排序工具 - 配置文件
所有路径使用 pathlib，兼容 Windows / macOS。
"""

from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent

# 人脸图片目录（启动时自动扫描）
FACES_DIR = BASE_DIR / "static" / "faces"

# 排名数据持久化文件
RANKING_DATA_FILE = BASE_DIR / "ranking_data.json"

# 支持的图片扩展名（小写）
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}

# ---------- 对比流程配置 ----------

# 最大对比轮次，达到后自动结束
MAX_ITERATIONS = 50

# ---------- Elo 评分算法参数 ----------

# 初始 Elo 分数（所有图片起点相同）
ELO_INITIAL_RATING = 1500.0

# K 因子：单次对比对分数的影响幅度，越大收敛越快
ELO_K_FACTOR = 32.0

# ---------- 配对策略优化 ----------

# 最近 N 组配对不再重复出现，避免短时间内重复对比
RECENT_PAIR_HISTORY_SIZE = 10

# 配对选择时，优先挑选「对比次数较少」的图片（权重指数）
PAIR_SELECTION_UNDERCOMPARED_WEIGHT = 2.0

# Flask 服务配置
FLASK_HOST = "127.0.0.1"
FLASK_PORT = 5000
FLASK_DEBUG = True

# 当默认端口被占用时，是否自动尝试后续端口
FLASK_AUTO_PORT_FALLBACK = True

# 自动找端口时的最大搜索步数（例如 5000 -> 最多尝试到 5050）
FLASK_PORT_SEARCH_LIMIT = 50
