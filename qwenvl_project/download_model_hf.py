from huggingface_hub import snapshot_download

# 干净、简单、不报错、M1/Mac 亲测可用
snapshot_download(
    repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
    local_dir="./models/Qwen2.5-VL-3B-Instruct"
)

print("✅ 下载成功！")