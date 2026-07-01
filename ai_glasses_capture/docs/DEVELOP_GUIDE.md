# 最小 Demo 开发与操作指南

> 版本：v0.2  
> 日期：2026-06-30  
> 目标：**从零到在安卓手机上安装可运行的 APK**  
> 开发机：**MacBook Pro · Apple M5 · 48 GB 内存**  
> 关联文档：[需求](REQUIREMENTS.md) · [架构](ARCHITECTURE.md) · [UI 原型](../ui/preview.html)

本文档面向第一次做 Android 开发的场景，步骤按 **macOS + Apple Silicon** 编写。按顺序做完下面步骤，即可拿到一个可安装的 `.apk`，界面与 [`ui/preview.html`](../ui/preview.html) 原型一致（4 Tab：采集 / 设备 / 记录 / 我的）。

---

## 一、Demo 范围（最小版做什么）

| 功能 | Demo 是否包含 | 说明 |
|------|---------------|------|
| 4 Tab 界面 | ✅ | 对齐 HTML 原型，可切换、可输入 |
| 个人信息表单 | ✅ | 姓名、编号、备注，本地保存 |
| 设备下拉 + IP 输入 | ✅ | 可选手动填写流地址 |
| 连接状态展示 | ✅ | 检测 WiFi / 网络是否可用 |
| 开始 / 停止采集按钮 | ✅ | UI + 计时器；无眼镜时可 Mock |
| 记录列表 | ✅ | 展示本地已保存条目（可先 Mock 数据） |
| 真机 RTSP 拉流录制 | ⏳ Phase 1 | 需眼镜或电脑 Mock 流 |
| VPN 抓包 | ❌ | Phase 2，Demo 不做 |

**Demo 验收标准：** 手机能安装打开 App，四个页面正常切换，表单能输入，点击「开始采集」有计时反馈，Debug APK 编译成功。

---

## 二、环境准备

### 2.1 开发机（本机配置）

| 项目 | 本机 |
|------|------|
| 机型 | **MacBook Pro** |
| 芯片 | **Apple M5**（ARM 架构，Android Studio 原生运行，无需 Rosetta） |
| 内存 | **48 GB** — 远超 Android 开发推荐配置，Gradle 编译、Android 模拟器、ExoPlayer 调试均可并行，无需刻意省内存 |
| 操作系统 | **macOS**（建议 macOS 14 Sonoma 或更新） |
| 磁盘空间 | 预留 **15~20 GB** 给 Android Studio + SDK + Gradle 缓存 |

> 48 GB 内存的额外好处：可同时开 Android Studio + 模拟器 + 浏览器查文档 + Wireshark 抓包，编译几乎不会因内存 swap 卡顿。若愿意，可在 SDK Manager 里安装 **ARM64 系统镜像**，用本机模拟器调试，不必总插真机。

### 2.2 安装 Android Studio（Mac）

1. 打开 [Android Studio 官网](https://developer.android.com/studio)，下载 **Mac Apple Silicon** 版本（勿下 Intel 版）。
2. 将 `Android Studio.app` 拖入 **应用程序**，首次打开若提示安全限制：  
   **系统设置 → 隐私与安全性 → 仍要打开**。
3. 首次启动向导中勾选：
   - Android SDK
   - Android SDK Platform
   - **Android Virtual Device**（本机内存充足，建议勾选，便于无手机时调试 UI）
4. 打开 **Android Studio → Settings**（macOS 快捷键 `⌘ ,`）→ **Languages & Frameworks → Android SDK**：
   - **SDK Platforms**：勾选 **Android 15（API 35）**
   - **SDK Tools**：确保已安装
     - Android SDK Build-Tools
     - Android SDK Platform-Tools
     - Android Emulator
     - Android SDK Command-line Tools

5. **JDK**：Android Studio 自带 JetBrains Runtime 17，一般无需单独安装。若 Gradle 报 JDK 错误：  
   **Settings → Build, Execution, Deployment → Build Tools → Gradle → Gradle JDK** → 选 **jbr-17**。

6. 验证 `adb`（Platform-Tools 默认路径）：

```bash
# Apple Silicon Mac 默认 SDK 路径
export ANDROID_HOME="$HOME/Library/Android/sdk"
export PATH="$PATH:$ANDROID_HOME/platform-tools"

adb version
```

建议写入 `~/.zshrc` 以便终端长期使用：

```bash
echo 'export ANDROID_HOME=$HOME/Library/Android/sdk' >> ~/.zshrc
echo 'export PATH=$PATH:$ANDROID_HOME/platform-tools' >> ~/.zshrc
source ~/.zshrc
```

7. **（可选）Homebrew 安装辅助工具：**

```bash
brew install --cask android-platform-tools   # 仅 adb 时可用
brew install ffmpeg                            # Mock RTSP 测试流时用
```

### 2.3 手机 / AR 眼镜端

准备一台 **Android 10（API 29）及以上** 的设备：

- 普通安卓手机，或
- 运行 Android 系统的 AR 眼镜

**必须开启：**

| 设置项 | 路径（各品牌略有差异） |
|--------|------------------------|
| 允许安装未知来源 | 设置 → 安全 / 隐私 → **安装未知应用** → 对 QQ / 微信 / 文件管理器允许 |
| USB 调试（推荐） | 设置 → 关于手机 → **连点「版本号」7 次** → 返回 → 开发者选项 → **USB 调试** |
| 保持屏幕唤醒（可选） | 开发者选项 → **不锁定屏幕**（调试时方便） |

### 2.4 Mac 连接 Android 设备

| 方式 | MacBook Pro 操作 |
|------|------------------|
| **USB-C 数据线** | 用 **USB-C 转 USB-C** 或 **USB-C 转 USB-A** 线连手机（避免仅充电线）。连接后手机点「允许 USB 调试」。Mac **无需**像 Windows 那样装驱动。 |
| **无线调试** | 手机与 Mac 同一 WiFi；USB 先连一次执行 `adb tcpip 5555`，再 `adb connect 手机IP:5555`。Android 11+ 还可在开发者选项里开「无线调试」配对。 |
| **Android 模拟器** | Android Studio → Device Manager → Create Device → 选 **arm64-v8a** 镜像（Apple Silicon 原生，流畅）。48 GB 内存可给模拟器分配 4~8 GB RAM。 |

连接 AI 眼镜测试时：手机需能加入眼镜 WiFi 热点，或与眼镜在同一局域网。Mac 可同时连家庭 WiFi（Mock 流）而手机连眼镜热点——调试时注意填对 IP。

---

## 三、创建 Android 工程

### 3.1 New Project 参数（务必一致）

1. 打开 Android Studio → **New Project**
2. 选择 **Empty Activity**（或 **Empty Compose Activity**，若模板可用）
3. 填写：

| 字段 | 填写值 |
|------|--------|
| Name | `GlassCapture` |
| Package name | `com.arcollect.demo` |
| Save location | 建议 `study/ai_glasses_capture/android-app/` |
| Language | **Kotlin** |
| Minimum SDK | **API 29（Android 10）** |
| Build configuration | Kotlin DSL（`build.gradle.kts`） |

4. 点击 **Finish**，等待 **Gradle Sync** 完成。M5 + 48 GB 下首次 Sync 约 **3~10 分钟**（主要等网络下载依赖），后续增量编译通常 **几十秒内**。

### 3.2 Gradle Sync 失败时

| 现象 | 处理 |
|------|------|
| 下载超时 | 检查网络 / 代理；或在 `gradle.properties` 配置镜像 |
| JDK 报错 | Settings → Gradle → Gradle JDK 选 **jbr-17** |
| Compose 版本冲突 | 以 AI 给出的 `libs.versions.toml` / `build.gradle.kts` 为准整体替换 |
| 「不受信任的 SDK」 | macOS 弹窗点允许；或 Settings → Android SDK → SDK Tools 重新勾选 |

Sync 成功标志：底部状态栏显示 **Gradle build finished**，无红色报错。

---

## 四、接入 AI 生成的代码

> Phase 1 将由 AI 在 `android-app/` 目录生成完整工程文件。生成后按下列流程操作。

### 4.1 推荐方式：直接打开已有工程

若 `android-app/` 目录已包含完整 Gradle 工程：

1. Android Studio → **Open** → 选择 `study/ai_glasses_capture/android-app/`
2. 信任 Gradle 项目，等待 Sync
3. 跳过「New Project」，无需手动 Empty Activity

### 4.2 手动替换方式（Empty Activity 已建好）

按 AI 给出的**文件路径清单**，在 Android Studio 左侧 Project 视图中（选 **Android** 或 **Project** 模式）逐个操作：

| 操作 | 说明 |
|------|------|
| **替换** | 打开目标文件，全选删除，粘贴 AI 给出的完整内容 |
| **新建** | 右键包名 → New → Kotlin Class / File，路径与 AI 一致 |
| **删除** | AI 说明要删的默认 `MainActivity` 等，按指引删除 |

**常见需替换 / 新增的文件：**

```
android-app/
├── app/build.gradle.kts          ← 依赖（Compose、Room 等）
├── build.gradle.kts
├── settings.gradle.kts
├── gradle/libs.versions.toml     ← 版本目录（如有）
└── app/src/main/
    ├── AndroidManifest.xml       ← 权限、Application、Activity
    ├── java/com/arcollect/demo/
    │   ├── GlassCaptureApp.kt
    │   ├── MainActivity.kt
    │   ├── ui/...                 ← 四个 Screen + Theme
    │   └── ...
    └── res/
        ├── values/strings.xml
        └── xml/file_paths.xml      ← FileProvider（分享文件时用）
```

**注意：**

- 包名统一为 `com.arcollect.demo`，与 Manifest、`namespace` 一致。
- 粘贴后点击 **File → Sync Project with Gradle Files**。
- 有红色波浪线时，鼠标悬停看提示，多数是缺 import 或依赖未 Sync。

### 4.3 编译前自检清单

- [ ] Gradle Sync 无错误
- [ ] `MainActivity` 在 Manifest 中已声明且 `exported="true"`
- [ ] `minSdk = 29`，`compileSdk = 35`（或与 AI 文档一致）
- [ ] 手机已通过 USB 连接，顶部设备下拉能看到机型

---

## 五、在真机运行（开发调试）

比打 APK 更快的方式，适合改代码阶段：

1. 手机 USB 连接 Mac，弹出「允许 USB 调试」→ **允许**（可勾选「始终允许此计算机」）
2. Android Studio 顶部工具栏设备选你的手机或模拟器
3. 点击绿色 **Run ▶**（macOS 快捷键 `^R` 或菜单 Run → Run 'app'）
4. 等待安装，手机自动打开 App

终端确认设备已识别：

```bash
adb devices
# 应显示 xxxxxxxx    device
```

**日志查看：** 底部 **Logcat**，过滤器填 `com.arcollect.demo` 或 `GlassCapture`。

---

## 六、生成 APK（照着做就能拿到安装包）

### 方式 A：Debug APK（最简单，本地测试推荐）

无需签名证书，适合给自己手机安装测试。

**图形界面：**

1. 顶部菜单 **Build → Build Bundle(s) / APK(s) → Build APK(s)**
2. 等待编译，右下角通知 **APK(s) generated successfully** → 点 **locate**

**命令行（Mac 终端）：**

```bash
cd ~/workspace/study/ai_glasses_capture/android-app
./gradlew :app:assembleDebug
```

> Apple Silicon 上 Gradle 使用 ARM 原生 JVM，全量 Debug 编译在本机通常 **1~3 分钟** 内完成。

**APK 位置：**

```
android-app/app/build/outputs/apk/debug/app-debug.apk
```

### 方式 B：Signed APK（上架或长期测试）

1. **Build → Generate Signed Bundle / APK**
2. 选择 **APK** → Next
3. **Create new...** 新建 Keystore（仅本地测试可随便填）：
   - Key store path：选保存路径，如 `glasscapture-debug.jks`
   - Password / Alias：自设并**记住**
   - Validity：25 年
   - Certificate：姓名组织可填测试信息
4. 选择 **debug** 或 **release** 构建类型 → Finish
5. 输出在 `app/release/` 或提示的路径

> **提示：** Demo 阶段用 **方式 A** 即可。Release 签名请妥善备份 `.jks`，丢失无法更新同一包名 App。

---

## 七、把 APK 装到手机

### 7.1 传输文件

任选一种：

| 方式 | 操作 |
|------|------|
| QQ / 微信 | 发 `app-debug.apk` 到文件传输助手，手机下载 |
| 数据线 | 复制 APK 到手机 `Download/` 目录 |
| adb | `adb install -r app/build/outputs/apk/debug/app-debug.apk` |

### 7.2 安装

1. 在手机上找到 APK 文件，点击安装
2. 若提示「禁止安装未知应用」，去设置里对该应用（文件管理器 / 微信）**允许安装**
3. 安装完成 → 打开 **GlassCapture**

### 7.3 adb 安装（开发者推荐）

```bash
adb devices                    # 应列出 device
adb install -r app-debug.apk   # -r 表示覆盖安装
```

---

## 八、Demo 功能走查（安装后自测）

按 UI 原型逐项点击：

```
1. 我的 → 填写姓名「测试员」、编号 → 返回采集页，确认姓名已同步
2. 设备 → 下拉选「通用 RTSP 协议」→ IP 填 192.168.4.1 → 点「连接设备」
3. 采集 → 点「开始采集」→ 计时器走动 → 点「停止采集」
4. 记录 → 查看是否有新条目（Demo 可先显示 Mock 或占位）
```

**无 AI 眼镜时：** 在 Mac 上用 FFmpeg 推 Mock RTSP 流（见 [架构文档 §9.3](ARCHITECTURE.md#93-无眼镜时的-mock)），手机与 Mac 同一 WiFi，设备页填 `rtsp://<Mac的局域网IP>:8554/live`。

查看 Mac 本机 IP：

```bash
ipconfig getifaddr en0    # WiFi
```

---

## 九、常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| Gradle Sync 一直转圈 | 网络 / 首次下载 | 换网络、开代理、重启 Android Studio |
| Run 按钮灰色，无设备 | Mac 未识别手机 | 换 **数据线**（非仅充电线）；`adb kill-server && adb start-server`；检查手机是否点「允许调试」 |
| `adb: command not found` | PATH 未配置 | 按 **§2.2** 设置 `ANDROID_HOME` 并 `source ~/.zshrc` |
| 安装失败「版本不兼容」 | minSdk 高于手机系统 | 换 Android 10+ 设备，或降低 minSdk（不推荐低于 29） |
| 安装失败「解析包错误」 | APK 损坏或未下完 | 重新传输；用 adb install 看具体错误 |
| App 闪退 | 代码 / 权限问题 | Logcat 看 FATAL 行；检查 Manifest 权限 |
| SSID 显示 unknown | Android 10+ 限制 | 授予**位置权限**后重试 |
| Compose 预览空白 | 预览器 bug | 以真机 Run 为准；或用本机 ARM 模拟器 |
| 模拟器极慢 | 误用 x86 镜像 | Device Manager 中选带 **arm64-v8a** 标签的系统镜像 |

**查看崩溃日志：**

```bash
adb logcat | grep -E "FATAL|AndroidRuntime|arcollect"
```

---

## 十、开发迭代建议

```
第 1 天   按本文完成工程创建 + Debug APK 装手机
第 2~3 天 四个 Compose 页面对齐 preview.html
第 4~5 天 DataStore 存用户信息 + Room 记录列表
第 6~7 天 设备页 WiFi 状态 + 连接测试
第 2 周   ExoPlayer 接 Mock RTSP，真机录制 MP4
```

每完成一步可重新 **Build APK** 装到手机验证，不必等全部做完。

---

## 十一、文档与代码对照

| 文档 | 用途 |
|------|------|
| [REQUIREMENTS.md](REQUIREMENTS.md) | 做什么功能 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 怎么分层、用什么库 |
| [../ui/preview.html](../ui/preview.html) | 界面长什么样 |
| **本文档** | 怎么装环境、怎么打 APK、怎么装手机 |

---

## 十二、快速命令备忘（Mac 终端）

```bash
# 进入工程目录（按你的 workspace 路径调整）
cd ~/workspace/study/ai_glasses_capture/android-app

# 编译 Debug APK
./gradlew :app:assembleDebug

# 安装到已连接手机
adb install -r app/build/outputs/apk/debug/app-debug.apk

# 启动 App
adb shell am start -n com.arcollect.demo/.MainActivity

# 卸载（重装前清数据）
adb uninstall com.arcollect.demo

# 查看 Mac 局域网 IP（Mock RTSP 时用）
ipconfig getifaddr en0
```

---

## 十三、下一步

1. 确认本文 **第二节 ~ 第七节** 环境与本机 Android Studio 就绪  
2. 让 AI 生成 `android-app/` 最小 Demo 代码（Compose 四 Tab 壳）  
3. 按 **第四节** 打开或替换文件 → **第六节** 打 APK → **第七节** 装手机  
4. 按 **第八节** 走查通过后，再接入真实眼镜 WiFi 与 RTSP 拉流  

完成以上步骤，即完成 **Phase 0 → Phase 1 Demo** 的落地闭环。
