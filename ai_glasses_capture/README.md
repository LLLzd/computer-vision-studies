# AI 眼镜数据采集 App（Android）

连接 AI 眼镜 WiFi，抓取并解码音视频数据，保存到 Android 手机本地的采集工具。

**平台范围：** Android APK 原生开发（Kotlin + Jetpack Compose），UI 按 [`ui/preview.html`](ui/preview.html) 原型实现。  
**开发机：** MacBook Pro · Apple M5 · 48 GB · macOS

## 文档

| 文档 | 说明 |
|------|------|
| [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) | 产品需求与功能清单 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Android 技术方案、架构与上架指南 |
| [docs/DEVELOP_GUIDE.md](docs/DEVELOP_GUIDE.md) | **最小 Demo 操作步骤与开发指南（从零到 APK）** |
| [ui/preview.html](ui/preview.html) | 界面原型（浏览器打开预览） |

## 快速预览界面

```bash
open ui/preview.html
```

## 项目阶段

- **Phase 0（当前）**：需求梳理、UI 原型、技术方案
- **Phase 1**：Android 工程 + WiFi 连接 + 流式拉取 + 本地存储（MVP）
- **Phase 2**：VPN 抓包解析、多设备协议适配
- **Phase 3**：UI 精修、稳定性、小米等应用商店上架
