# 人脸偏好打分排序工具

基于 **Python Flask + 原生前端** 的本地离线人脸偏好打分 Web 工具。通过成对对比收集偏好，使用 **Elo 评分算法** 更新排名，最终将结果线性归一化到 **0–10 分**。

所有数据保存在本地 `ranking_data.json`，图片存放在 `static/faces`，**无任何网络上传**。

## 项目结构

```
face_ranking_project/
├── app.py                 # Flask 主程序
├── config.py              # 配置文件（轮次、Elo 参数等）
├── ranking_engine.py      # Elo 算法与配对逻辑
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
| Elo 更新 | 点击「喜欢左侧/右侧」后实时更新 Elo 分数 |
| 轮次控制 | `config.py` 中 `MAX_ITERATIONS` 控制最大对比轮次 |
| 手动结束 | 对比页「结束对比」按钮可提前终止 |
| 全局排名 | 按 Elo 降序排名，线性映射为 0–10 分（1 位小数） |
| 结果页 | 展示排名、得分、对比次数 |
| 导出 CSV | 一键导出 `face_rankings.csv` |
| 重置数据 | 清空所有对比记录，保留图片 |

## 配置项（config.py）

```python
MAX_ITERATIONS = 50              # 最大对比轮次
ELO_INITIAL_RATING = 1500.0        # 初始 Elo 分数
ELO_K_FACTOR = 32.0                # K 因子（影响单次对比幅度）
RECENT_PAIR_HISTORY_SIZE = 10      # 避免短期内重复配对
PAIR_SELECTION_UNDERCOMPARED_WEIGHT = 2.0  # 优先对比次数少的图片
FLASK_AUTO_PORT_FALLBACK = True            # 端口占用时自动切换
FLASK_PORT_SEARCH_LIMIT = 50               # 向后最多尝试 50 个端口
```

## 算法说明

### Elo 评分

标准 Elo 成对比较算法：

- 期望胜率：`E_A = 1 / (1 + 10^((R_B - R_A) / 400))`
- 更新公式：`R_A' = R_A + K × (S_A - E_A)`（胜=1，负=0）

### 分数归一化

对比结束后，按 Elo 排名线性映射：

- 第 1 名 → **10.0 分**
- 第 N 名 → **0.0 分**
- 中间名次等间距插值，保留 1 位小数

### 智能配对优化

- 最近 N 组配对不会重复出现
- 优先选择对比次数较少的图片组合，保证频次均匀

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
- 每张图的当前 Elo 分数
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
2. 连续做 10~20 轮偏好选择
3. 点击「结束对比」进入结果页并导出 CSV

## 常见问题

### 1) 页面提示图片不足

- 原因：`static/faces/` 有效图片少于 2 张
- 处理：至少放入 2 张可读取图片（建议同一分辨率或近似比例）

### 2) 结果分数看起来差异很小

- 原因：对比轮次不足，Elo 尚未拉开
- 处理：增加轮次，或在 `config.py` 中适当调大 `ELO_K_FACTOR`

### 3) 想开始新一轮评测但保留旧结果

- 先备份 `ranking_data.json`
- 再使用页面「重置数据」按钮，重新开始一轮新对比
