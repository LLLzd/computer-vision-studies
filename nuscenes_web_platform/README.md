# NuScenes 本地网页数据平台

在浏览器中浏览本机 NuScenes 数据：左侧为「原始数据 / 真值 Label」两级导航，右侧为搜索、样本列表与可视化（相机原图、LiDAR BEV、3D 框 BEV、2D 投影、ego 定位 JSON），并支持按场景加载帧序列与简易播放。

## 依赖

- Python 3.10+
- 已下载的 NuScenes 目录（含 `samples/`、`sweeps/`、`v1.0-*`）
- 与 [demo_by_nuscenes](../demo_by_nuscenes) 同级的工具模块（通过 `PYTHONPATH` 自动加入 `demo_by_nuscenes` 以复用 `utils`）

## 配置

复制环境变量模板并按本机路径修改：

```bash
cp .env.example .env
```

必填：

- `NUSCENES_DATAROOT`：数据集根目录（例如 `.../data/nuscenes`）
- `NUSCENES_VERSION`：元数据版本目录名，如 `v1.0-mini` 或 `v1.0-trainval`

可选：`NUSCENES_WEB_HOST`（默认 `127.0.0.1`）、`NUSCENES_WEB_PORT`（默认 `8765`）。

若不创建 `.env`，[scripts/run.sh](scripts/run.sh) 会尝试使用与 `demo_by_nuscenes` 相邻的默认路径：`../demo_by_nuscenes/data/nuscenes` 与 `v1.0-mini`。

## 启动

```bash
cd nuscenes_web_platform
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
chmod +x scripts/run.sh
./scripts/run.sh
```

在浏览器打开：`http://127.0.0.1:8765/`（仅本机，勿将服务暴露到公网）。

## API 摘要

| 路径 | 说明 |
|------|------|
| `GET /api/health` | 健康检查 |
| `GET /api/meta` | 相机 / 雷达通道、类别列表 |
| `GET /api/scenes` | 场景列表 |
| `GET /api/scenes/{token}/samples` | 场景内样本分页 |
| `GET /api/samples/{token}` | 样本摘要 |
| `GET /api/samples/{token}/media?channel=` | 原始媒体 URL |
| `GET /api/samples/{token}/annotations` | 3D 标注 |
| `GET /api/samples/{token}/ego` | ego_pose |
| `GET /api/media/sample_data/{token}` | 受控文件下载 |
| `GET /api/clips/frames` | 场景内帧序列（用于播放） |
| `GET /api/search` | 按类别子串 + 场景过滤样本（结果内存缓存） |
| `GET /api/render/lidar_bev` | LiDAR BEV PNG |
| `GET /api/render/ann_bev` | 3D 框 BEV PNG |
| `GET /api/render/camera_2d` | 相机 + 3D 框 2D 投影 PNG |

媒体路径均由 devkit 解析并在数据集根目录下做 `realpath` 校验，避免路径穿越。

## 实现说明

- 后端：[backend/main.py](backend/main.py)（FastAPI）、[backend/visuals.py](backend/visuals.py)（Matplotlib 渲染）
- 前端静态页：[static/index.html](static/index.html)、[static/app.js](static/app.js)、[static/styles.css](static/styles.css)
- 大图 / 渲染结果使用进程内 LRU 缓存，减轻重复请求压力
