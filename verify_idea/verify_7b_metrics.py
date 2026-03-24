import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 核心配置
# ==========================================
DEVICE = "cuda"
MODEL_ID = "Qwen/Qwen2.5-Math-7B-Instruct"
MAX_NEW_TOKENS = 120 # 截取一段中等长度的推理用于可视化

print(f"📡 正在加载 7B 数学模型进行流形动力学验证...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
model.eval()

def run_latent_drift_visualization():
    # 使用一道具有明确步骤的经典代数题
    question = "Solve the equation: 3(x - 2) + 4 = 13. Please reason step by step."
    prompt = f"Question: {question}\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = inputs.input_ids
    past_key_values = None
    
    hidden_states = []
    tokens = []
    
    print("🚀 开始逐 Token 生成并提取隐状态...")
    for step in range(MAX_NEW_TOKENS):
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids, 
                past_key_values=past_key_values, 
                use_cache=True, 
                output_hidden_states=True
            )
        
        logits = outputs.logits[0, -1, :]
        past_key_values = outputs.past_key_values
        
        # 提取最后一层的隐状态 (维度 3584)
        last_hidden = outputs.hidden_states[-1][0, -1, :].cpu().to(torch.float32).numpy()
        hidden_states.append(last_hidden)
        
        # 获取当前预测的 Token
        next_id = torch.argmax(logits).unsqueeze(0).unsqueeze(0)
        
        # 记录 Token 文本用于 X 轴展示 (将换行符等特殊字符可视化)
        token_str = tokenizer.decode(next_id[0])
        if token_str == '\n':
            token_str = '[\\n]'
        elif token_str == ' ':
            token_str = '[space]'
            
        tokens.append(token_str)
        input_ids = next_id 
        
        if next_id.item() == tokenizer.eos_token_id:
            break

    print("📊 正在计算隐空间余弦漂移 (Cosine Drift)...")
    # 计算相邻 Token 之间的余弦相似度: S_t = cos(h_t, h_{t-1})
    similarities = []
    for i in range(1, len(hidden_states)):
        sim = cosine_similarity(
            hidden_states[i].reshape(1, -1), 
            hidden_states[i-1].reshape(1, -1)
        )[0, 0]
        similarities.append(sim)
        
    # ==========================================
    # 绘制高大上的学术级波形图
    # ==========================================
    print("🎨 正在生成 Latent Drift 轨迹图...")
    plt.figure(figsize=(20, 7)) # 宽一点以便放下所有 Token
    plt.style.use('seaborn-v0_8-darkgrid')
    
    x_axis = np.arange(1, len(similarities) + 1)
    
    # 画主折线
    plt.plot(x_axis, similarities, marker='.', markersize=8, color='#2980b9', linewidth=2, label='$S_t = \cos(h_t, h_{t-1})$')
    
    # 寻找谷底 (局部极小值) 并标注，这代表模型发生了认知跳跃
    trough_indices = []
    for i in range(1, len(similarities)-1):
        if similarities[i] < similarities[i-1] and similarities[i] < similarities[i+1]:
            # 设定一个相对明显的阈值，比如相似度掉到均值以下
            if similarities[i] < np.mean(similarities) - 0.5 * np.std(similarities):
                trough_indices.append(i)
                
    # 用红线标出这些“自然切块点”
    for idx in trough_indices:
        plt.axvline(x=idx+1, color='#e74c3c', linestyle='--', alpha=0.7)
        
    plt.xticks(x_axis, tokens[1:], rotation=75, fontsize=9)
    plt.title('Latent Drift Dynamics (Qwen2.5-Math-7B): Finding Natural Chunking Boundaries', fontsize=16, fontweight='bold')
    plt.ylabel('Cosine Similarity b/w Consecutive Tokens', fontsize=12)
    plt.xlabel('Generated Tokens', fontsize=12)
    
    # 优化显示范围
    plt.ylim(min(similarities) - 0.05, max(similarities) + 0.01)
    plt.xlim(0, len(similarities) + 1)
    plt.tight_layout()
    
    plt.savefig('latent_drift_chunking_verification.png', dpi=300)
    print(f"✅ 验证图已生成: latent_drift_chunking_verification.png")

if __name__ == "__main__":
    run_latent_drift_visualization()