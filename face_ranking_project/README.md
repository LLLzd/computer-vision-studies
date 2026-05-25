# 人脸偏好打分排序工具

基于 **Python Flask + 原生前端** 的本地离线人脸偏好打分 Web 工具。通过成对对比收集偏好，使用 **TrueSkill(μ/σ) 天梯算法** 更新排名，最终将结果线性归一化到 **0–10 分**。

所有数据保存在本地 `ranking_data.json`，图片存放在 `static/faces`，**无任何网络上传**。

## 项目结构

```
face_ranking_project/
├── app.py                 # Flask 主程序
├── config.py              # 配置文件（轮次、TrueSkill 参数等）
├── ranking_engine.py      # TrueSkill 算法与配对逻辑
├── ranking_data.json      # 对比记录与分数（运行时自动更新）
├── requirements.txt       # Python 依赖
├── static/
│   ├── css/style.css      # 样式
│   ├── js/
│   │   ├── compare.js     # 对比页逻辑
│   │   └── results.js     # 结果页逻辑
│   └── faces/             # 本地人脸图片目录
└── templates/
    ├── index.html         # 对比页
    └── results.html       # 结果页
```

## 快速开始

### 1. 安装依赖

```bash
cd face_ranking_project
pip install -r requirements.txt
```

### 2. 添加人脸图片

将 `.jpg` / `.png` 等图片放入 `static/faces/` 目录，**至少需要 2 张**。

也可以使用内置抓取脚本从**百度图片**高速下载（默认关键词：女学生）：

```bash
python fetch_real_faces.py --limit 300 --workers 32
```

常用参数：

- `--limit 300`：目标成功下载数量
- `--word 女学生`：百度图片搜索关键词（可改成帅哥、男学生等）
- `--workers 32`：并发下载线程数（越大越快，按网络调整）
- `--rn 60`：每页候选数量
- `--output-dir static/faces`：输出目录

> 请遵守目标网站服务条款与当地法律法规，仅用于本地学习研究。

### 3. 启动服务

```bash
python app.py
```

浏览器访问：<http://127.0.0.1:5000>

> 若 5000 端口被占用，程序会自动切换到下一个可用端口（如 5001），以控制台输出地址为准。

## 功能说明

| 功能 | 说明 |
|------|------|
| 自动扫描 | 启动时扫描 `static/faces` 下所有有效图片 |
| 成对对比 | 每次随机展示 2 张不重复图片，左右布局 |
| TrueSkill 更新 | 点击「喜欢左侧/右侧」后实时更新 μ/σ |
| 轮次控制 | 默认无限轮次，可持续对比并随时查看结果；可用 `MAX_ITERATIONS` 设置上限 |
| 随时查看结果 | 对比页可随时进入结果页，返回后继续左右选择 |
| 全局排名 | 按有效得分 `μ-3σ` 排名，线性映射为 0–10 分（1 位小数） |
| 结果页 | 展示排名、得分、对比次数 |
| 导出 CSV | 一键导出 `face_rankings.csv` |
| 重置数据 | 清空所有对比记录，保留图片 |

## 配置项（config.py）

```python
MAX_ITERATIONS = 0                 # <=0 表示无限轮次
TRUESKILL_MU = 25.0                # 初始 μ
TRUESKILL_SIGMA = TRUESKILL_MU / 3 # 初始 σ
TRUESKILL_BETA = TRUESKILL_SIGMA / 2
TRUESKILL_TAU = TRUESKILL_SIGMA / 100
RECENT_PAIR_HISTORY_SIZE = 10      # 避免短期内重复配对
PAIR_SELECTION_UNDERCOMPARED_WEIGHT = 2.0  # 优先低频图片
CONVERGENCE_WINDOW = 12            # 收敛判定窗口
CONVERGENCE_DELTA_THRESHOLD = 0.03 # 收敛阈值
FLASK_AUTO_PORT_FALLBACK = True            # 端口占用时自动切换
FLASK_PORT_SEARCH_LIMIT = 50               # 向后最多尝试 50 个端口
```

## 算法说明

### TrueSkill 评分（1v1）

每张图维护 `(μ, σ)` 两个参数：

- `μ`：当前能力估计
- `σ`：不确定性，样本越少越大
- 有效得分：`μ - 3σ`（更保守，更抗刷分）

每次对比后，系统按 TrueSkill 数学公式更新胜负双方的 `μ/σ`，并结合低频样本校准与分差抑制策略，提高公平性与稳定性。

### 分数归一化

对比结束后，按有效得分 `μ-3σ` 线性映射：

- 第 1 名 → **10.0 分**
- 第 N 名 → **0.0 分**
- 中间名次等间距插值，保留 1 位小数

### 智能配对优化

- 最近 N 组配对不会重复出现
- 优先选择对比次数较少、σ 较高的图片组合，保证频次均匀与快速收敛
- 当最近窗口内分数变化很小时，状态会标记为“已收敛”（不强制停止）

## 快捷键

对比页支持键盘操作：

- `←` 喜欢左侧
- `→` 喜欢右侧

## 兼容性

- Windows / macOS / Linux
- Python 3.9+
- 损坏或无法读取的图片会自动跳过，不会导致程序崩溃

## 数据文件说明

`ranking_data.json` 由程序自动维护，通常包含以下信息：

- 图片基础信息（文件名、创建时间等）
- 每张图的当前 `μ/σ` 与有效得分
- 成对对比历史（谁胜谁负）
- 每张图参与对比次数

建议：

- 可以手动备份该文件（例如按日期复制一份）
- 不建议在服务运行时手改文件内容，避免 JSON 冲突

## 最小可跑流程（3 分钟）

```bash
cd face_ranking_project
pip install -r requirements.txt
python app.py
```

然后：

1. 打开浏览器访问控制台打印的地址
2. 可持续进行偏好选择（无限轮次）
3. 任意时刻点击「查看排名结果」查看当前排序并可继续返回对比

## 常见问题

### 1) 页面提示图片不足

- 原因：`static/faces/` 有效图片少于 2 张
- 处理：至少放入 2 张可读取图片（建议同一分辨率或近似比例）

### 2) 结果分数看起来差异很小

- 原因：对比次数仍不足或样本仍未收敛
- 处理：继续对比，或适当调高 `LOW_FREQUENCY_UPDATE_BOOST`

### 3) 想开始新一轮评测但保留旧结果

- 先备份 `ranking_data.json`
- 再使用页面「重置数据」按钮，重新开始一轮新对比
