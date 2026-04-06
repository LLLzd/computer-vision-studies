# 登录 Hugging Face，只需要粘贴你的 token 即可
from huggingface_hub import login
import getpass

# 从用户输入获取 token
print("请输入你的 Hugging Face token (以 hf_ 开头):")
YOUR_HF_TOKEN = getpass.getpass()

print("正在登录 Hugging Face...")
login(token=YOUR_HF_TOKEN)
print("✅ 登录成功！")