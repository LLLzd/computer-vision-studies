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

# 最大对比轮次：
# - <= 0: 不限制（推荐，支持无限对比）
# - > 0 : 达到后自动结束
MAX_ITERATIONS = 0

# ---------- TrueSkill 参数 ----------

# TrueSkill 初始参数（每张图片默认同起点）
TRUESKILL_MU = 25.0
TRUESKILL_SIGMA = TRUESKILL_MU / 3.0

# TrueSkill 观测噪声与动态波动参数
TRUESKILL_BETA = TRUESKILL_SIGMA / 2.0
TRUESKILL_TAU = TRUESKILL_SIGMA / 100.0

# sigma 合法区间，避免数值异常
TRUESKILL_SIGMA_MIN = TRUESKILL_SIGMA / 10.0
TRUESKILL_SIGMA_MAX = TRUESKILL_SIGMA * 1.5

# 有效得分定义：mu - 3 * sigma
EFFECTIVE_SCORE_SIGMA_FACTOR = 3.0

# ---------- 配对策略优化 ----------

# 最近 N 组配对不再重复出现，避免短时间内重复对比
RECENT_PAIR_HISTORY_SIZE = 10

# 配对选择时，优先挑选「对比次数较少」的图片（权重指数）
PAIR_SELECTION_UNDERCOMPARED_WEIGHT = 2.0

# 配对时额外考虑不确定性（sigma）权重
PAIR_SELECTION_UNCERTAINTY_WEIGHT = 1.2

# 配对时抑制短时间重复抽中同一图片
RECENT_FACE_COOLDOWN_SIZE = 8
RECENT_FACE_COOLDOWN_PENALTY = 0.35

# 倾向选择有效得分接近的样本对，提高比较信息量
PAIR_INFO_GAIN_GAP_SCALE = 8.0
PAIR_INFO_GAIN_WEIGHT = 0.8

# ---------- 新样本公平与抗刷分 ----------

# 低频次样本会获得更高更新权重（用于加速冷启动）
LOW_FREQUENCY_TARGET = 8
LOW_FREQUENCY_UPDATE_BOOST = 0.45

# 当双方有效得分差距过大时，降低单次结果冲击，抑制碾压刷分
CRUSH_GAP_SCALE = 10.0
CRUSH_PENALTY_STRENGTH = 0.55

# 更新强度边界，避免过小或过大
UPDATE_IMPACT_MIN = 0.55
UPDATE_IMPACT_MAX = 1.35

# ---------- 收敛判定 ----------

# 达到最少轮次后才允许触发收敛停止
CONVERGENCE_MIN_ROUNDS = 20

# 每张图至少参与次数，避免“假收敛”
CONVERGENCE_MIN_COMPARISONS_PER_FACE = 3

# 最近窗口内平均有效得分变化小于阈值判定收敛
CONVERGENCE_WINDOW = 12
CONVERGENCE_DELTA_THRESHOLD = 0.03

# Flask 服务配置
FLASK_HOST = "127.0.0.1"
FLASK_PORT = 5000
FLASK_DEBUG = True

# 当默认端口被占用时，是否自动尝试后续端口
FLASK_AUTO_PORT_FALLBACK = True

# 自动找端口时的最大搜索步数（例如 5000 -> 最多尝试到 5050）
FLASK_PORT_SEARCH_LIMIT = 50
