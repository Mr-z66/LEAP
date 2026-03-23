import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os

# ================= 配置区 =================
model_path = os.path.join(os.getcwd(), "models", "Qwen2.5-1.5B")
save_path = "gsm8k_15b_hidden_states.pt"
num_samples = 1000  # 跑前 1000 道题

# 混合切分策略 (Hybrid Chunking) 的核心超参数
punctuations = [".", ",", "!", "?", "\n"]
min_tokens = 5   # 低于 5 个 token 不切，防噪声
max_tokens = 30  # 高于 30 个 token 强切，防漂移
# ==========================================

print("🚀 初始化 Qwen-1.5B 模型与分词器...")
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True
)
model.eval()

print("📚 正在加载 GSM8K 训练集...")
dataset = load_dataset("gsm8k", "main", split=f"train[:{num_samples}]")

all_extracted_data = [] # 用于存放这 1000 道题的所有切块数据

print(f"🔥 开始全自动批量提取 (共 {num_samples} 题)...")
for idx, data in enumerate(tqdm(dataset, desc="Processing GSM8K")):
    question = data["question"]
    true_answer = data["answer"]
    
    messages = [
        {"role": "system", "content": "You are a helpful math assistant. Please reason step by step."},
        {"role": "user", "content": question}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256, # 限制单题生成长度，提升收集效率
            return_dict_in_generate=True,
            output_hidden_states=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
    generated_token_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
    gen_hidden_states = outputs.hidden_states[1:] 
    
    # 💡 防弹补丁：强制对齐文本和隐状态的长度，取两者最小值，截断越界部分
    valid_length = min(len(generated_token_ids), len(gen_hidden_states))
    generated_token_ids = generated_token_ids[:valid_length]
    
    current_chunk_text = ""
    current_chunk_tokens = []
    question_chunks = []
    
    for i, token_id in enumerate(generated_token_ids):
        token_str = tokenizer.decode([token_id])
        current_chunk_text += token_str
        current_chunk_tokens.append(token_id)
        
        chunk_len = len(current_chunk_tokens)
        hit_punctuation = any(p in token_str for p in punctuations)
        
        # 💡 核心逻辑：触发混合切分条件
        # 条件 1：遇到了标点，且长度已经达标 (>= min_tokens)
        # 条件 2：一直没遇到标点，但长度已经触碰红线 (>= max_tokens)
        if (hit_punctuation and chunk_len >= min_tokens) or (chunk_len >= max_tokens):
            
            # 提取当前步、最后一层的隐状态，并立刻转移到 CPU 防止 GPU OOM
            boundary_hs = gen_hidden_states[i][-1][0, -1, :].clone().cpu()
            
            question_chunks.append({
                "chunk_text": current_chunk_text.strip(),
                "token_count": chunk_len,
                "hidden_state": boundary_hs  # Tensor shape: [1536]
            })
            
            # 清空缓存，准备切下一块
            current_chunk_text = ""
            current_chunk_tokens = []
            
    # 收尾：处理最后一段没被切分的残余文本
    if current_chunk_tokens:
        boundary_hs = gen_hidden_states[-1][-1][0, -1, :].clone().cpu()
        question_chunks.append({
            "chunk_text": current_chunk_text.strip(),
            "token_count": len(current_chunk_tokens),
            "hidden_state": boundary_hs
        })
        
    all_extracted_data.append({
        "question_id": idx,
        "question": question,
        "true_answer": true_answer,
        "chunks": question_chunks
    })

# 把心血保存到硬盘！
print(f"\n💾 正在将 {len(all_extracted_data)} 道题的隐状态特征持久化至磁盘...")
torch.save(all_extracted_data, save_path)
print(f"🎉 提取大业完成！数据集已安全保存至: {save_path}")