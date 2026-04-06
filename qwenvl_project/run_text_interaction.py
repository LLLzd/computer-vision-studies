from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch

# 你的模型路径（绝对正确）
model_path = "./models/Qwen/Qwen2___5-VL-3B-Instruct"

print("✅ 模型加载中...")

# 加载（官方原生正确方式）
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
).eval()

# 加载处理器
processor = AutoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True
)

# 初始化对话历史
messages = []

print("\n🤖 通义千问已就绪！")
print("💡 输入你的问题，按回车发送。")
print("🔚 输入 'exit' 或 'quit' 或直接按回车退出。")
print("=" * 60)

# 多轮对话循环
while True:
    # 获取用户输入
    user_input = input("\n👤 你：").strip()
    
    # 检查是否退出
    if user_input.lower() in ["exit", "quit"] or user_input == "":
        print("\n👋 再见！")
        break
    
    # 添加用户消息到对话历史
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": user_input},
        ]
    })
    
    # 应用官方聊天模板
    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 构造输入
    inputs = processor(
        text=[text_prompt],
        return_tensors="pt"
    ).to(model.device)
    
    # 生成
    print("🤖 思考中...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            top_p=1.0,
            temperature=1.0
        )
    
    # 输出结果
    response = processor.decode(outputs[0], skip_special_tokens=True)
    print("\n🤖 回答：", response)
    
    # 添加模型回复到对话历史
    messages.append({
        "role": "assistant",
        "content": [
            {"type": "text", "text": response},
        ]
    })

print("=" * 60)
