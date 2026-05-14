# gsplat_m1_recon

一个专门为 **Mac Apple Silicon (M1, 16GB)** 设计的、可本地直接运行的 3D Gaussian Splatting 小场景重建工程。

目标场景：手机拍摄约 20 秒环绕小物体（杯子、摆件、玩具）视频，完成：

1. 视频抽帧
2. COLMAP 位姿估计与稀疏重建
3. 3DGS 风格高斯优化训练（MPS/低内存优化）
4. GT vs Render 对比预览
5. 模型导出（NPZ + PLY）

---

## 1. 工程特性

- **M1 兼容**：优先 `torch.mps`，不依赖 CUDA。
- **低内存模式**：限制高斯总量、可见 splat 数量、训练分辨率，避免 16GB 爆内存。
- **双模式**：
  - `quick`：约 2 分钟内跑通流程，快速验证。
  - `standard`：约 20~40 分钟，小物体获得可用重建结果。
- **小白友好**：核心脚本均含中文注释，参数可在 `config.yaml` 一处修改。

---

## 2. 项目文件树

```text
gsplat_m1_recon/
├── config.yaml                      # 主配置文件（路径、训练参数、低内存开关）
├── requirements.txt                 # 依赖清单
├── README.md
├── .gitignore
├── input/                           # 放手机视频，例如 object.mov
├── output/                          # 自动生成：帧、COLMAP、checkpoint、预览、导出
├── scripts/
│   ├── run_quick.sh                 # 一键 quick 模式
│   └── run_standard.sh              # 一键 standard 模式
└── src/
    ├── __init__.py
    ├── run_pipeline.py              # 全流程入口：抽帧->COLMAP->训练->预览->导出
    ├── train.py                     # 3DGS 训练主脚本（M1 参数优化）
    ├── preview.py                   # 导出 GT/Render 对比视频
    ├── compare_vis.py               # 五列对比图（GT|Pred|误差×8|热力图|SSIM）+ 新视角导出
    ├── eval_compare.py              # 仅从 checkpoint 重新导出对比图
    ├── render_trajectory_compare.py # 原视频等时长 | 沿 COLMAP 轨迹渲染 并排 MP4
    ├── export_model.py              # 导出 NPZ/PLY
    ├── core/
    │   ├── gaussian_model.py        # 高斯参数模型
    │   └── renderer.py              # 简化可微 splatting 渲染器
    ├── preprocess/
    │   ├── extract_frames.py        # 视频抽帧+清晰度过滤
    │   └── run_colmap.py            # COLMAP 自动化脚本
    └── utils/
        ├── config_utils.py          # 配置读取/目录创建
        ├── device_utils.py          # MPS 设备选择与内存统计
        ├── colmap_io.py             # 解析 COLMAP 文本模型
        ├── camera_pose.py           # 世界系绕 Y 旋转（新视角外参）
        └── natural_sort.py          # 帧文件名自然排序
```

---

## 3. 运行前准备

### 3.1 安装系统工具（COLMAP）

本工程使用外部 `colmap` 命令完成 SfM 位姿估计。

```bash
brew install colmap
colmap -h
```

如果 `colmap -h` 能正常输出帮助信息，即安装成功。

### 3.2 安装 Python 依赖（使用你已有 `.venv`）

在项目目录下执行：

```bash
cd /Users/rik/workspace/study/gsplat_m1_recon
/Users/rik/workspace/study/.venv/bin/python -m pip install -r requirements.txt
```

> 建议用 `python -m pip`，避免某些历史虚拟环境中 `pip` 脚本 shebang 路径失效的问题。

---

## 4. 输入数据要求（非常重要）

将手机视频放到：

```text
input/object.mov
```

拍摄建议（直接决定质量）：

- 物体尽量放在桌面中央，背景纹理适中（不要纯白墙）。
- 围绕物体 **缓慢环绕一圈**，时长约 15~25 秒。
- 画面稳定、尽量少运动模糊，光照均匀，避免强反光。
- 不要离物体太近，保证物体完整在画面中。

---

## 5. 一键执行命令

### 5.1 快速验证（约 2 分钟）

```bash
cd /Users/rik/workspace/study/gsplat_m1_recon
./scripts/run_quick.sh
```

用途：检查全链路可用性（抽帧、COLMAP、训练、预览、导出）。

### 5.2 标准训练（约 20~40 分钟）

```bash
cd /Users/rik/workspace/study/gsplat_m1_recon
./scripts/run_standard.sh
```

用途：小物体可用重建结果。

---

## 6. 手动分步执行（推荐先理解）

### Step 1: 抽帧

```bash
python src/preprocess/extract_frames.py \
  --video input/object.mov \
  --output output/frames \
  --stride 2 \
  --max_frames 220 \
  --resize_width 960 \
  --sharpness_threshold 35
```

### Step 2: COLMAP 位姿估计

```bash
python src/preprocess/run_colmap.py \
  --images output/frames \
  --workspace output/colmap
```

输出：
- `output/colmap/sparse_txt/cameras.txt`
- `output/colmap/sparse_txt/images.txt`
- `output/colmap/sparse_txt/points3D.txt`

### Step 3: 训练 3DGS（M1 优化）

先在 `config.yaml` 中设置模式：
- `train.mode: quick` 或 `standard`

再运行：

```bash
python src/train.py --config config.yaml
```

训练输出：
- `output/checkpoints/step_xxxxxx.pt`
- `output/previews/preview_xxxxxx.jpg`（仅当前随机视角的渲染）
- `output/previews/compare_trainstep_xxxxxx.jpg`（**5 列**：GT | Pred | \|err\|×8 灰度 | 误差 turbo 热力图+colorbar | SSIM+colorbar，训练中周期性保存）
- `output/previews/eval_final/same_view/`：训练结束自动生成多张同上 **5 列**对比（固定抽帧）
- `output/previews/eval_final/novel_view/`：以序列中间帧为基准，绕世界 Y 轴旋转得到的 **新视角纯渲染**（无 GT，用于看几何是否连贯）

仅已有 checkpoint、想重新导出对比图时：

```bash
python src/eval_compare.py --config config.yaml --checkpoint output/checkpoints/step_000120.pt
```

### Step 4: 生成对比预览视频（GT vs Render，短片段）

```bash
python src/preview.py \
  --config config.yaml \
  --checkpoint output/checkpoints/step_002200.pt \
  --output output/previews/compare.mp4 \
  --frames 120 \
  --fps 24
```

### Step 4b: 原视频等时长「左真右渲」轨迹对比（推荐训练完成后）

**左**：原视频帧。**右**：与训练时相同量级的 **splat 神经渲染图**（非点云、非高斯中心），在 `train.render_height_*` 对应分辨率上合成后再放大到与左图同尺寸；若在整幅 1080p 上直接 splat，`max_visible_splats` 上限会让画面像稀疏点云。

与输入 `object.mov` **同播放时长**；默认逐帧对应，亦可用 `--frame_stride N` 每 N 帧采样 1 次（输出 fps 自动为 `fps*M/n`，M 为写入帧数）。相机在 COLMAP 排序后的关键帧之间 **SE(3) 插值**（平移线性 + 旋转 slerp），内参缩放到渲染分辨率。默认使用 `output/checkpoints/` 下 **step 最大** 的模型。

```bash
python src/render_trajectory_compare.py \
  --config config.yaml \
  --output output/previews/trajectory_compare_full.mp4
```

指定 checkpoint 或视频路径：

```bash
python src/render_trajectory_compare.py \
  --config config.yaml \
  --checkpoint output/checkpoints/step_002200.pt \
  --video input/object.MOV \
  --output output/previews/trajectory_compare_full.mp4
```

> 全片逐帧渲染较慢；可用 `--render_height 320` 降低内部渲染高度以提速，或 **`--frame_stride 30`** 约 30 倍减少渲染次数（**总时长仍与源视频一致**）。

**30 倍稀疏采样示例：**

```bash
python src/render_trajectory_compare.py \
  --config config.yaml \
  --frame_stride 30 \
  --output output/previews/trajectory_compare_30x.mp4
```

### Step 5: 导出模型

```bash
python src/export_model.py \
  --checkpoint output/checkpoints/step_002200.pt \
  --export_dir output/export
```

导出内容：
- `output/export/gaussians_model.npz`：完整高斯参数（坐标/尺度/颜色/不透明度）
- `output/export/gaussians_points.ply`：可在点云工具中查看的点+颜色+alpha

---

## 7. 从“手机拍视频”到“重建完成”全流程指南

1. 手机拍 20 秒环绕视频，保存为 MOV。
2. 拷贝到 `input/object.mov`。
3. 安装依赖：
   - `brew install colmap`
   - `python -m pip install -r requirements.txt`
4. 先跑 `./scripts/run_quick.sh` 验证流程。
5. 再跑 `./scripts/run_standard.sh` 获取可用质量结果。
6. 查看：
   - `output/previews/compare_trainstep_*.jpg`：训练中 **5 列**对比（含误差热力图与 SSIM）
   - `output/previews/eval_final/same_view/`：训练结束 **5 列**对比（同上）
   - `output/previews/eval_final/novel_view/`：绕 Y 轴小角度 **新视角渲染**（无 GT）
   - `output/previews/compare.mp4`（左 GT，右渲染，短片段）
   - `output/previews/trajectory_compare_full.mp4`（运行 `render_trajectory_compare.py`：与 **原视频等时长** 左真右渲）
   - `output/export/gaussians_model.npz` / `gaussians_points.ply`

---

## 8. 关键参数说明（config.yaml）

### 数据相关

- `data.frame_stride`：抽帧间隔，越大帧越少，速度更快但细节可能下降。
- `data.max_frames`：最大训练帧数，M1 16GB 推荐 150~260。
- `data.resize_width`：抽帧宽度，越小越快越省内存。
- `data.sharpness_threshold`：清晰度阈值，过滤模糊帧。

### 训练相关

- `train.mode`：`quick` 或 `standard`。
- `train.low_memory`：低内存模式开关（16GB 推荐 `true`）。
- `train.max_gaussians_*`：高斯总量上限。
- `train.max_visible_splats_*`：每次渲染可见高斯上限，显著影响速度/内存。
- `train.render_height_*`：训练渲染高度，影响画质与速度。

### 评测可视化（eval）

- `eval.num_same_view_panels`：训练结束导出多少张「同视角 5 列对比」。
- `eval.novel_yaws`：新视角相对中间帧绕世界 Y 轴的偏角列表（度）；纯渲染无 GT。

---

## 9. 结果预期与限制说明

- 本工程是 **纯 Python + PyTorch + COLMAP** 的可落地版本，重点是本地易跑通和 M1 友好。
- 渲染器采用简化可微 splatting（非 CUDA 栅格器），速度和精度无法完全对齐官方高性能 C++/CUDA 实现。
- 通过合理拍摄与参数调优，可在小物体场景得到可用的重建与多视角预览结果。

---

## 10. 常见问题

### Q1: 训练太慢

- 用 `quick` 模式先验证。
- 降低 `render_height_standard`。
- 减小 `max_gaussians_standard` 和 `max_visible_splats_standard`。

### Q2: 内存占用过高

- 确保 `train.low_memory: true`。
- 降低 `data.max_frames` 到 120~180。
- 降低 `max_visible_splats_standard`。

### Q3: COLMAP 失败（点太少）

- 重拍视频，增加纹理和光照均匀性。
- 减少运动模糊，速度更慢一点拍摄。
- 尝试 `--camera_model OPENCV`（在 `run_colmap.py` 参数中传入）。
