# 第一视角视频 Embedding 向量库

使用 **Qwen3-VL-Embedding-2B** 对短第一视角（Ego-centric）视频做预处理与向量化，构建 FAISS 向量库，支持文本检索相似帧。

## 功能

- **视频预处理**：按时间间隔抽帧、等比缩放（letterbox / center_crop / stretch）
- **Embedding 编码**：Qwen3-VL Embedding 逐帧或整段视频编码
- **向量库**：FAISS 存储 + JSON 元数据（时间戳、帧号、路径）
- **文本检索**：自然语言查询最相似的视频帧

## 目录结构

```
ego_video_embedding_project/
├── config.py              # 配置（抽帧间隔、模型路径等）
├── download_model.py      # 下载模型
├── preprocess.py          # 仅预处理（不加载模型）
├── build_index.py         # 预处理 + Embedding + 建库
├── query.py               # 文本检索
├── inputs/                # 放入你的视频
├── outputs/
│   ├── frames/            # 预处理后的帧
│   └── index/             # 向量库
└── src/
    ├── preprocessing/     # 图像/视频预处理
    ├── embedder/          # Qwen3-VL Embedding 封装
    └── vector_store/      # FAISS 向量库
```

## 环境安装

```bash
cd study/ego_video_embedding_project

# 建议使用 study 目录已有 venv，或新建
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> **M1 Mac 提示**：默认使用 2B 模型，`batch_size=1`，`max_edge=768`。内存不足时可改为 `--max-edge 512`。

## 快速开始

### 1. 下载模型

```bash
# HuggingFace（需网络）
python download_model.py

# 国内可用 ModelScope
python download_model.py --source modelscope
```

### 2. 放入视频

将你的第一视角短视频放到 `inputs/`：

```bash
cp /path/to/your_video.MOV inputs/
```

### 3. 仅预处理（可选，快速查看抽帧效果）

```bash
python preprocess.py -i inputs/your_video.MOV -o outputs/frames --interval 1.0
```

输出：`outputs/frames/frame_*.jpg` + `manifest.json`

### 4. 构建向量库

```bash
python build_index.py -i inputs/your_video.MOV
```

常用参数：

| 参数 | 说明 | 默认 |
|------|------|------|
| `--interval` | 抽帧间隔（秒） | 1.0 |
| `--max-frames` | 最大帧数 | 64 |
| `--max-edge` | 帧最长边像素 | 768 |
| `--resize-mode` | letterbox / center_crop / stretch | letterbox |
| `--mode` | frames（逐帧）/ video（整段） | frames |

### 5. 文本检索

```bash
python query.py "打开冰箱拿东西"
python query.py "在桌上写字" --top-k 5
```

示例输出：

```
#1  score=0.7821  time=00:03  frame=90  path=outputs/frames/frame_000090_t3.00s.jpg
#2  score=0.7104  time=00:05  frame=150  path=outputs/frames/frame_000150_t5.00s.jpg
```

## 配置

编辑 `config.py` 可修改：

- `FRAME_INTERVAL_SEC`：抽帧间隔
- `MAX_FRAME_EDGE`：预处理分辨率
- `DEFAULT_INSTRUCTION`：Embedding 任务指令
- `EMBEDDING_DIM`：向量维度（2B=2048）

## 技术说明

- **模型**：[Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)
- **向量库**：FAISS IndexFlatIP（L2 归一化后等价余弦相似度）
- **推荐模式**：短 ego 视频用 `frames` 逐帧编码，检索粒度更细

## 参考

- [Qwen3-VL-Embedding GitHub](https://github.com/QwenLM/Qwen3-VL-Embedding)
- [Qwen3-VL-Embedding 技术报告](https://github.com/QwenLM/Qwen3-VL-Embedding/blob/main/assets/qwen3vlembedding_technical_report.pdf)
