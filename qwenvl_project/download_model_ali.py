from modelscope import snapshot_download

# 国内阿里云源（ModelScope），不需要登录，不需要 VPN，不会报错
#
# ── Qwen2.5-VL（视觉语言模型）────────────────────────────────────────────
#   3B / 7B / 32B / 72B — 各尺寸均有 Instruct 版（BF16 全精度）
#   3B / 7B / 72B — 另有 AWQ 量化版（-AWQ 后缀，显存占用更低）
#
# ── Qwen2.5（纯文本 LLM）────────────────────────────────────────────────
#   Qwen2.5：0.5B / 1.5B / 3B / 7B / 14B / 32B / 72B（Base + Instruct）
#   Qwen2.5-Coder：1.5B / 7B / 32B
#   Qwen2.5-Math：1.5B / 7B / 72B
#   API 专有（不开源权重）：Qwen2.5-Turbo、Qwen2.5-Plus（MoE）
#
# ── Qwen3（纯文本 LLM）──────────────────────────────────────────────────
#   稠密模型：0.6B / 1.7B / 4B / 8B / 14B / 32B
#   MoE 模型：30B-A3B（总 30B，每 token 激活 3B）
#             235B-A22B（总 235B，每 token 激活 22B）
#   2025-07 更新（2507）：235B-A22B / 30B-A3B / 4B 各拆分为
#     -Instruct-2507（直接回答）和 -Thinking-2507（逐步推理）两个独立权重
#
# ── Qwen3-VL（视觉语言模型，Qwen3 系列）─────────────────────────────────
#   2B / 4B / 8B / 32B — 各尺寸均有 Instruct 和 Thinking 版
#   MoE：30B-A3B / 235B-A22B — 同样有 Instruct 和 Thinking 版
#
model_dir = snapshot_download(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    cache_dir="./models"
)

print("✅ 模型下载成功！路径：", model_dir)
