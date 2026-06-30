# Row Delay Video

把普通视频转换成「按行带延迟播放」的新视频，产生类似水波纹的横向波动效果。

## 效果说明

不再对每一行单独延迟（高分辨率视频会太乱），而是把相邻若干行分成一组 **band**，每组共用同一个时间偏移：

- 第 1 组（行 0 ~ row_step-1）来自原视频第 `t` 帧
- 第 2 组（行 row_step ~ 2*row_step-1）来自原视频第 `t-1` 帧
- 第 k 组来自原视频第 `t-(k-1)` 帧

默认 `--row-step 10`：每 10 行延迟 1 帧。

例如 1920 行、`row_step=10` 时：

- 共 192 个 band
- 最大延迟 191 帧（而不是逐行延迟的 1919 帧）
- 视觉上会看到横向波纹带从上到下依次滞后

输出视频长度：`原帧数 + ceil(高度 / row_step) - 1`。

## 目录结构

```
row_delay_video/
├── process.py          # 主处理脚本
├── make_test_video.py  # 生成测试视频
├── inputs/             # 放入你的原视频
├── outputs/            # 输出结果
└── requirements.txt
```

## 安装

```bash
cd study/row_delay_video
pip install -r requirements.txt
```

也可使用 `study/.venv`：

```bash
source ../.venv/bin/activate
pip install -r requirements.txt
```

## 用法

```bash
# 默认每 10 行延迟 1 帧
python process.py inputs/your_video.mp4

# 更平滑的波纹：每 20 行延迟 1 帧
python process.py inputs/your_video.mp4 --row-step 20

# 逐行延迟（旧行为，高视频会很乱）
python process.py inputs/your_video.mp4 --row-step 1

# 指定输出路径
python process.py inputs/your_video.mp4 -o outputs/custom_name.mp4

# 开头/结尾缺帧时用黑色填充（默认是 clamp 到首/末帧）
python process.py inputs/your_video.mp4 --fill-mode black
```

## 测试

```bash
python make_test_video.py
python process.py inputs/test_moving_ball.mp4
```

输出在 `outputs/test_moving_ball_row_delay_s10.mp4`。

## 命令行 Tab 补全（可选）

之前用 argcomplete 注册全局 `python` 会**抢走所有 python 命令的补全**，导致连文件名都补全不了。现在改成项目专用命令 `rdv`，不影响普通 `python`。

**如果当前终端已经坏了：先关掉这个终端，开一个新的。**

在项目目录执行：

```bash
source enable_completion.zsh
```

然后用 `rdv` 代替 `python process.py`：

```bash
rdv inputs/object.MOV --<Tab>
rdv inputs/object.MOV --row-step 20
rdv inputs/object.MOV -o outputs/custom.mp4
```

`rdv` 就是 `python process.py` 的快捷方式，参数完全一样。

想永久生效，把下面这行写进 `~/.zshrc`：

```bash
source /Users/rik/workspace/study/row_delay_video/enable_completion.zsh
```
