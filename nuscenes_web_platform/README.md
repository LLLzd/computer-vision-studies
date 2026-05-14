# NuScenes 本地网页数据平台

在浏览器中浏览本机 NuScenes 数据：左侧用**复选框叠加**选择六路相机与 LiDAR BEV、以及真值（LiDAR 上 3D 框 / 相机投影框）；主区为 **2×3 相机栅格**（顺序：左前、前、右前 / 左后、后、右后）+ **单张 LiDAR BEV**（点云经 `calibrated_sensor` 变换到 ego，与 BEV 框一致；可选叠加框）；支持按场景加载片段并**同步播放**所有已勾选图层。右侧为 ego_pose JSON、**world 平面轨迹图**（蓝线 + 当前帧红点）与标注摘要。

## 交互与性能

- 未勾选的相机或 LiDAR 格为**黑底**，不发起图像请求。
- 工具栏 **「结果列表」**：取消勾选后左栏完全隐藏，可视化区域占满剩余宽度。
- 相机投影框会过滤：无 lidar/radar 回波、3D 尺寸异常、投影过大/过小或占满画面的标注；LiDAR BEV 点云按**水平距离 0–100 m**着色（固定范围），并绘制 **100 m / 50 m** 参考虚线圆；子采样按样本 token 固定随机种子，播放时颜色不跳变。
- 播放时每一帧会刷新所有已勾选视图（多路原图或叠框图 + LiDAR 渲染），`trainval` 等大数据集上 CPU/IO 压力会明显上升，可调小「最大帧」或增大「步长」。

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
| `GET /api/meta` | 相机通道列表（栅格顺序与 devkit 一致）、类别等 |
| `GET /api/scenes` | 场景列表 |
| `GET /api/scenes/{token}/samples` | 场景内样本分页 |
| `GET /api/scenes/{token}/ego_trail` | 场景内关键帧 ego 轨迹 world (x,y)，供轨迹 SVG |
| `GET /api/samples/{token}` | 样本摘要 |
| `GET /api/samples/{token}/media?channel=` | 返回 `media_url` JSON |
| `GET /api/samples/{token}/raw_image/{channel}` | 直接返回相机原图字节（供 `<img src>`） |
| `GET /api/samples/{token}/annotations` | 3D 标注 |
| `GET /api/samples/{token}/ego` | ego_pose |
| `GET /api/media/sample_data/{token}` | 受控文件下载 |
| `GET /api/clips/frames` | 场景内帧序列（用于播放） |
| `GET /api/search` | 按类别子串 + 场景过滤样本（结果内存缓存） |
| `GET /api/render/lidar_bev?boxes=true|false` | LiDAR BEV PNG，可选叠加 BEV 3D 框 |
| `GET /api/render/camera_2d?channel=&boxes=true|false` | 相机 PNG；`boxes=false` 为原图 |
| `GET /api/render/ann_bev` | 仅 BEV 框（调试用，前端已合并进 LiDAR 图层） |

媒体路径均由 devkit 解析并在数据集根目录下做 `realpath` 校验，避免路径穿越。

## 实现说明

- 后端：[backend/main.py](backend/main.py)（FastAPI）、[backend/visuals.py](backend/visuals.py)（Matplotlib / Pillow 渲染）
- 前端静态页：[static/index.html](static/index.html)、[static/app.js](static/app.js)、[static/styles.css](static/styles.css)
- 大图 / 渲染结果使用进程内 LRU 缓存，缓存键包含 `boxes` 等查询参数
