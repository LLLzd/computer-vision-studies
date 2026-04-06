from modelscope import snapshot_download

# 国内阿里云源，不需要登录，不需要VPN，不会报错
model_dir = snapshot_download(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    cache_dir="./models"
)

print("✅ 模型下载成功！路径：", model_dir)