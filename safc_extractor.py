import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# 1. 指向你刚刚下好的本地模型库 (纯离线加载，绝对不卡网络！)
model_path = os.path.join(os.getcwd(), "models", "Qwen2.5-1.5B")

print(f"🚀 正在加载模型: {model_path}")
print("   ↳ 放心，完全本地加载，不走任何外网请求...")

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
# 使用 bfloat16 精度加载，充分利用你的 RTX 6000 算力，device_map="auto" 会自动调度双卡
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True
)
model.eval()

# 2. 构造一个经典的 GSM8K 数学题
question = "John buys 3 apples. Then he buys 5 more apples. How many apples does he have in total?"
messages = [
    {"role": "system", "content": "You are a helpful math assistant. Please reason step by step."},
    {"role": "user", "content": question}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

print("\n🧠 正在生成推理过程，并同步提取深层脑电波 (Hidden States)...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        return_dict_in_generate=True,
        output_hidden_states=True, # 💡 论文核心开关：强迫大模型吐出每一层的隐状态！
        pad_token_id=tokenizer.eos_token_id
    )

# 3. 提取生成的 Token
generated_token_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
print(f"\n📝 模型生成的完整推理回答:\n{generated_text}\n")

# 4. 执行 SAFC (语义锚定切块) 与 隐状态对齐
print("✂️ 正在执行 SAFC 语义锚定切块与探针向量提取...")
punctuations = [".", ",", "!", "?", "\n"] # 你的启发式标点符号边界

chunks = []
current_chunk_text = ""
current_chunk_token_count = 0

# outputs.hidden_states 包含了每生成一个 token 时，模型所有层的隐状态
hidden_states_tuple = outputs.hidden_states 

for i, token_id in enumerate(generated_token_ids):
    token_str = tokenizer.decode([token_id])
    current_chunk_text += token_str
    current_chunk_token_count += 1
    
    # 检测是否触发语义切块锚点
    if any(p in token_str for p in punctuations):
        # 💡 获取当前 token (切块边界) 最后一层的隐状态
        # hidden_states_tuple[i] 是第 i 步的隐状态
        # [-1] 取神经网络的最后一层
        # [0, -1, :] 取 batch_size=0, 最后一个序列位置的完整高维向量
        boundary_hidden_state = hidden_states_tuple[i][-1][0, -1, :]
        
        chunks.append({
            "text": current_chunk_text.strip(),
            "hidden_state": boundary_hidden_state,
            "token_count": current_chunk_token_count
        })
        # 清空容器，准备下一个切块
        current_chunk_text = ""
        current_chunk_token_count = 0

# 处理最后一段没有标点符号结尾的文本
if current_chunk_text.strip():
    boundary_hidden_state = hidden_states_tuple[-1][-1][0, -1, :]
    chunks.append({
        "text": current_chunk_text.strip(),
        "hidden_state": boundary_hidden_state,
        "token_count": current_chunk_token_count
    })

# 5. 展示极其性感的探针数据
print("\n📊 隐状态探针数据提取报告:")
for idx, chunk in enumerate(chunks):
    if chunk["text"]:
        vec_shape = chunk["hidden_state"].shape
        # 计算 L2-Norm 证明这是真实激活的高维向量
        vec_norm = torch.norm(chunk["hidden_state"]).item() 
        print(f"👉 [块 {idx+1}] (长度: {chunk['token_count']} tokens) 内容: {chunk['text']}")
        print(f"   ↳ 截获隐向量: {list(vec_shape)} 维, 向量 L2-Norm: {vec_norm:.4f}\n")

print("🎉 恭喜！LEAP 架构的底层特征提取管道已完全打通！")