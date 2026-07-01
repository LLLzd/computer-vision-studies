# AI 眼镜数据采集 App — 技术方案与架构设计（Android）

> 版本：v0.2  
> 日期：2026-06-30  
> 平台：**Android APK**（Kotlin 原生）  
> UI 参照：[`../ui/preview.html`](../ui/preview.html)

---

## 1. 范围说明

| 包含 | 不包含 |
|------|--------|
| Android 原生 App（APK / AAB） | iOS |
| Jetpack Compose UI，按 HTML 原型实现 | Web 版 |
| WiFi 连接 + 直连流 / VPN 抓包 | 厂商 SDK 集成 |
| 本地解码与存储 | 跨平台框架（Flutter / RN） |

本 App 是独立的 Android 数据采集工具，通过 WiFi 与 AI 眼镜通信，自行实现协议解析与落盘，**不依赖第三方 SDK**。

---

## 2. 设计原则

| 原则 | 说明 |
|------|------|
| **简单优先** | MVP 先打通「连接 → 拉流 → 存 MP4」 |
| **UI 对齐原型** | 4 Tab（采集 / 设备 / 记录 / 我的），深色主题 + 渐变主色 |
| **协议可插拔** | 不同眼镜型号用 Adapter + JSON 配置扩展 |
| **Android 能力优先** | 直连流 MVP；VPN 抓包 Phase 2 |
| **本地优先** | 数据仅存手机，无后端 |
| **可扩展** | 分层清晰，后期可加云端同步而不改核心 |

---

## 3. 总体架构

```
┌─────────────────────────────────────────────────────────────┐
│              UI Layer — Jetpack Compose                      │
│  CaptureScreen │ DeviceScreen │ RecordsScreen │ ProfileScreen│
│  （对应 preview.html 四个 Tab）                               │
└──────────────────────────┬──────────────────────────────────┘
                           │ ViewModel
┌──────────────────────────▼──────────────────────────────────┐
│                     Domain Layer                             │
│  CaptureUseCase │ ConnectDeviceUseCase │ RecordRepository   │
│  UserProfileRepository │ CaptureSessionManager               │
└──────────────────────────┬──────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│ Transport     │  │ Media Pipeline│  │ Data Layer    │
│ DirectStream  │  │ MediaCodec    │  │ Room DB       │
│ VpnCapture    │  │ MediaMuxer    │  │ MediaStore    │
│ (Phase 2)     │  │ FFmpeg (可选) │  │ DataStore     │
└───────────────┘  └───────────────┘  └───────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│ Protocol Adapters（自研，非 SDK）       │
│ GenericRtsp │ GenericTcp │ ...        │
└───────────────────────────────────────┘
```

---

## 4. 技术选型

### 4.1 核心技术栈

| 层级 | 选型 | 说明 |
|------|------|------|
| 语言 | **Kotlin 2.x** | Android 官方推荐 |
| UI | **Jetpack Compose + Material 3** | 还原原型：圆角卡片、底部 Tab、渐变按钮 |
| 架构 | **MVVM + Clean Architecture** | ViewModel + UseCase + Repository |
| 导航 | **Navigation Compose** | 4 Tab + 可选详情页 |
| 依赖注入 | **Hilt** | 管理 Service / Repository 生命周期 |
| 本地 DB | **Room** | 采集记录元数据 |
| 偏好设置 | **DataStore** | 姓名、设备配置等 |
| 网络 | **OkHttp** | HTTP / 自定义 TCP |
| 拉流预览 | **ExoPlayer (Media3)** | RTSP / HTTP 流预览与录制 |
| 封装落盘 | **MediaMuxer + MediaCodec** | 系统 API，低 CPU |
| 抓包（Phase 2） | **VpnService** 自研 | 参考 PCAPdroid，非第三方 SDK |
| 异步 | **Kotlin Coroutines + Flow** | 采集状态、连接状态推送 UI |
| 后台采集 | **Foreground Service** | 采集中通知栏保活 |

### 4.2 为何选 Android 原生而非跨平台

- 抓包依赖 `VpnService`，必须 Kotlin/Java 原生实现
- `MediaCodec` / `MediaMuxer` 与系统深度集成，性能最好
- UI 原型已是移动端单端设计，Compose 足够还原
- 上架国内安卓商店以 APK/AAB 为主，无跨平台额外成本

---

## 5. UI 实现对照（Compose ↔ 原型）

| 原型 Tab | Compose Screen | 主要组件 |
|----------|----------------|----------|
| 采集 | `CaptureScreen` | 预览 `AndroidView(ExoPlayer)`、连接状态 Pill、计时器、开始/停止渐变按钮 |
| 设备 | `DeviceScreen` | `ExposedDropdownMenu` 设备选择、`OutlinedTextField` IP/流地址、连接按钮 |
| 记录 | `RecordsScreen` | `LazyColumn` 卡片列表、搜索框、时长 Badge |
| 我的 | `ProfileScreen` | 表单（姓名/编号/备注）、存储占用、深色模式 Switch |

**设计 Token（与 preview.html 一致）：**

```kotlin
object AppTheme {
    val Bg = Color(0xFF0A0B10)
    val Surface = Color(0xFF14161F)
    val Surface2 = Color(0xFF1C1F2B)
    val Accent = Color(0xFF6366F1)
    val Accent2 = Color(0xFF8B5CF6)
    val Success = Color(0xFF22C55E)
    val Danger = Color(0xFFEF4444)
}
```

底部导航使用 `NavigationBar` + 4 个 `NavigationBarItem`，图标与原型一一对应。

---

## 6. 核心模块设计

### 6.1 Transport 传输层

#### 模式 A：直连流（MVP，优先）

```
手机 ──WiFi──► 眼镜 (192.168.x.x)
              ├── RTSP   rtsp://192.168.4.1/live
              ├── HTTP   http://192.168.4.1/stream
              └── TCP    自定义包头 + H.264 NAL
```

**开发前必做：** 用 Wireshark 或手机安装 PCAPdroid，连接眼镜 WiFi 抓包 5 分钟，确认协议与端口。

MVP 实现路径：

1. `DeviceScreen` 配置 IP / 流 URL
2. `ExoPlayer` 建立连接，在 `CaptureScreen` 预览
3. 录制：`MediaMuxer` 或 ExoPlayer 录制 API 写入 MP4
4. 若仅为裸 H.264 TCP 流，用 `Socket` + NAL 重组 + `MediaMuxer`

#### 模式 B：VPN 抓包（Phase 2）

```
App 流量 ──► VpnService (TUN) ──► 过滤眼镜 IP ──► ProtocolParser
                                              └──► H.264 / AAC 提取
```

- `PacketCaptureService extends VpnService`
- 解析 IP/TCP/UDP 包头，仅保留目标为眼镜 IP 的 payload
- 与用户说明：需授权 VPN，用途为本地设备数据采集

### 6.2 Protocol Adapter（自研协议层）

```kotlin
interface GlassesProtocolAdapter {
    val deviceId: String
    suspend fun connect(config: DeviceConfig): ConnectionResult
    fun openMediaStream(): Flow<MediaPacket>
    suspend fun disconnect()
}

data class MediaPacket(
    val type: PacketType,   // VIDEO | AUDIO
    val payload: ByteArray,
    val timestampUs: Long,
    val codec: String       // "h264", "aac"
)
```

设备配置存于 `assets/protocols/*.json`：

```json
{
  "id": "generic_rtsp",
  "name": "通用 RTSP 眼镜",
  "transport": "rtsp",
  "defaultUrl": "rtsp://192.168.4.1/live",
  "videoCodec": "h264",
  "audioCodec": "aac"
}
```

通过 `AdapterFactory` 按 `deviceId` 实例化，**无需引入厂商 SDK**。

### 6.3 Media Pipeline 解码与封装

```
MediaPacket / RTSP Stream
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│ RingBuffer      │ ──► │ MediaMuxer       │ ──► .mp4
│ (采集中)         │     │ 或 ExoPlayer 录制 │
└─────────────────┘     └──────────────────┘
```

- **MVP**：ExoPlayer 拉 RTSP，`MediaMuxer` 或 `-c copy` 思路落盘，不重编码
- **抓包模式**：NAL 单元拼帧 → `MediaCodec` 解码预览（可选）→ `MediaMuxer` 写容器
- 若系统 API 不够用，可引入 **FFmpeg Android 二进制**（自行编译 so，仍非 SDK 封装）

### 6.4 Storage 存储

```
/data/data/com.example.glasscapture/
├── files/media/2026/06/30/张三_20260630_143022.mp4
├── databases/records.db      # Room
└── cache/temp_capture_*.mp4  # 采集中临时文件
```

**Room Entity — `CaptureRecord`：**

| 字段 | 类型 |
|------|------|
| id | Long PK |
| subjectName | String |
| subjectId | String? |
| deviceId | String |
| filePath | String |
| durationSec | Double |
| fileSize | Long |
| createdAt | Long |
| note | String? |

导出分享：通过 `FileProvider` + `Intent.ACTION_SEND` 调系统分享面板。

---

## 7. Android 工程目录结构

```
ai_glasses_capture/
├── docs/
├── ui/preview.html              # UI 原型（设计参照）
├── android-app/                 # Phase 1 创建
│   ├── app/
│   │   └── src/main/
│   │       ├── AndroidManifest.xml
│   │       ├── assets/protocols/
│   │       ├── java/com/example/glasscapture/
│   │       │   ├── GlassCaptureApp.kt
│   │       │   ├── ui/
│   │       │   │   ├── navigation/MainNavHost.kt
│   │       │   │   ├── capture/CaptureScreen.kt
│   │       │   │   ├── device/DeviceScreen.kt
│   │       │   │   ├── records/RecordsScreen.kt
│   │       │   │   ├── profile/ProfileScreen.kt
│   │       │   │   └── theme/AppTheme.kt
│   │       │   ├── domain/
│   │       │   │   ├── usecase/
│   │       │   │   └── model/
│   │       │   ├── data/
│   │       │   │   ├── local/Room*
│   │       │   │   └── repository/
│   │       │   └── core/
│   │       │       ├── transport/
│   │       │       │   ├── DirectStreamTransport.kt
│   │       │       │   └── adapter/
│   │       │       ├── media/MediaRecorder.kt
│   │       │       └── vpn/PacketCaptureService.kt  # Phase 2
│   │       └── res/
│   └── build.gradle.kts
├── protocols/                   # 协议 JSON（与 assets 同步）
└── tools/
    └── wireshark_filters.md
```

---

## 8. 权限与 Manifest

```xml
<!-- 网络 -->
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
<uses-permission android:name="android.permission.ACCESS_WIFI_STATE" />
<uses-permission android:name="android.permission.CHANGE_WIFI_STATE" />

<!-- Android 10+ 读取 WiFi SSID 需位置权限 -->
<uses-permission android:name="android.permission.ACCESS_FINE_LOCATION" />

<!-- 前台采集 -->
<uses-permission android:name="android.permission.FOREGROUND_SERVICE" />
<uses-permission android:name="android.permission.FOREGROUND_SERVICE_DATA_SYNC" />
<uses-permission android:name="android.permission.POST_NOTIFICATIONS" />

<!-- Phase 2 VPN 抓包 -->
<uses-permission android:name="android.permission.BIND_VPN_SERVICE" />
```

`targetSdkVersion` 建议跟随 Google 最新要求（2026 年约 **API 35**）。首次启动按顺序请求：通知 → 位置（用于 SSID）→ VPN（仅抓包模式）。

---

## 9. 真机调试

> 开发机：**MacBook Pro · Apple M5 · 48 GB** · macOS。详细步骤见 [DEVELOP_GUIDE.md](DEVELOP_GUIDE.md)。

### 9.1 环境准备

1. 安装 **Android Studio（Mac Apple Silicon 版）**
2. 使用自带 **JBR 17**；SDK Platform **API 35**
3. 终端配置：`export ANDROID_HOME=$HOME/Library/Android/sdk`
4. 手机开启 **USB 调试**；Mac 用 USB-C 数据线连接，**无需额外驱动**
5. Android Studio 选择设备 Run；48 GB 内存可同时开 **ARM64 模拟器** 做 UI 调试

### 9.2 调试命令

```bash
# 安装 debug APK
./gradlew :app:installDebug

# 查看日志
adb logcat -s GlassCapture

# 查看 App 存储
adb shell run-as com.arcollect.demo ls files/media/

# Mac 查看本机 IP（Mock 流）
ipconfig getifaddr en0
```

### 9.3 无眼镜时的 Mock

在 Mac 上推 RTSP 测试流，手机与 Mac 连同一 WiFi：

```bash
ffmpeg -re -f lavfi -i testsrc=size=1280x720:rate=30 \
  -f lavfi -i sine=frequency=1000:sample_rate=44100 \
  -c:v libx264 -c:a aac -f rtsp rtsp://0.0.0.0:8554/live
```

App 设备页填写 `rtsp://<Mac局域网IP>:8554/live`（IP 用 `ipconfig getifaddr en0` 查看）。

### 9.4 常见问题

| 问题 | 处理 |
|------|------|
| SSID 显示 `<unknown ssid>` | 检查位置权限；Android 10+ 必须授权 |
| USB 调试时 WiFi 不可用 | 部分机型可 USB + WiFi 并存；或用 `adb tcpip 5555` 无线调试 |
| ExoPlayer RTSP 失败 | 确认眼镜编码为 H.264 Baseline；必要时换 FFmpeg 拉流 |
| VPN 被系统杀 | 使用 Foreground Service + 通知栏 |

---

## 10. 打包与上架（Android）

### 10.1 构建 Release APK / AAB

```bash
# 签名 AAB（推荐上架 Google Play）
./gradlew :app:bundleRelease

# 或 APK（侧载 / 部分国内商店）
./gradlew :app:assembleRelease
```

使用 Android Studio：**Build → Generate Signed Bundle / APK**，配置 keystore（首次创建并妥善备份）。

### 10.2 国内安卓商店

| 商店 | 入口 | 备注 |
|------|------|------|
| **小米应用商店** | [dev.mi.com](https://dev.mi.com) | 常用；部分类目需软著 |
| **华为应用市场** | developer.huawei.com | 独立审核 |
| **OPPO / vivo** | 各开放平台 | 可后期用聚合分发 |
| **Google Play** | play.google.com/console | 一次性 $25；面向海外 |

### 10.3 上架材料清单

- [ ] 应用图标 512×512
- [ ] 截图 4~8 张（按原型页面截真机图）
- [ ] 隐私政策 URL（说明：数据仅存本地、权限用途）
- [ ] 应用描述：写明「配合 AI 眼镜 WiFi 采集音视频」
- [ ] 若使用 VPN：描述中说明**仅用于连接眼镜设备，非代理上网**
- [ ] 软件著作权登记（小米等商店可能要求，提前 1~2 个月申请）

### 10.4 内测分发

- **Debug APK** 直接发群 / 网盘侧载（需允许未知来源）
- **Firebase App Distribution** 或小米内测渠道
- 正式上架前在 2~3 款机型（含小米）真机回归

---

## 11. 分期实施计划

```
Week 1  创建 Android 工程 + Compose 4 Tab 空壳（对齐 preview.html）
Week 2  DeviceScreen：WiFi/SSID、IP 配置、连接状态
Week 3  ExoPlayer 预览 + MediaMuxer 录制 + Room 记录列表
Week 4  真机连眼镜联调（Wireshark 确认协议）
Week 5  Foreground Service、权限流程、UI 细节打磨
Week 6  签名打包、隐私政策、小米/Google Play 提交
```

**Phase 2（+4 周）：** VpnService 抓包、ProtocolParser、多设备 JSON 配置。

---

## 12. 后期扩展（仍限 Android）

| 方向 | 做法 |
|------|------|
| UI 更精美 | Compose 动画、`AnimatedVisibility`、Lottie |
| 更稳定 | WorkManager 重试、Crashlytics、采集断点续传 |
| 多眼镜型号 | `protocols/` 增加 Adapter 实现 |
| 云端备份 | 可选模块：OSS 上传，与本地 Repository 解耦 |
| 边录边分析 | 接入本地 TFLite 或后端 API（独立模块） |

---

## 13. 关键决策记录

| 决策 | 选择 | 原因 |
|------|------|------|
| 平台 | Android APK 原生 | 抓包与 Media API 必需原生能力 |
| UI | Jetpack Compose | 还原原型、开发效率高 |
| 传输 MVP | 直连 RTSP/HTTP | 实现快、稳定 |
| 抓包 | Phase 2 VpnService | MVP 不阻塞主链路 |
| 集成方式 | 自研 Adapter | 不依赖厂商 SDK |
| 存储 | MP4 + Room | 简单、可播放、可扩展 |

---

## 14. 参考资料

- [Jetpack Compose 文档](https://developer.android.com/develop/ui/compose)
- [Media3 / ExoPlayer](https://developer.android.com/media/media3)
- [Android VpnService](https://developer.android.com/reference/android/net/VpnService)
- [PCAPdroid 源码](https://github.com/emanuele-f/PCAPdroid) — VPN 抓包参考
- [Room 数据库](https://developer.android.com/training/data-storage/room)
- [小米开放平台](https://dev.mi.com/console/)
