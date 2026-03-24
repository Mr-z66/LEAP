import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 建议根据实际显存调整

import torch
import numpy as np
import re
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# 配置
DEVICE = "cuda"
MODEL_ID = "Qwen/Qwen2.5-Math-7B-Instruct" 
MAX_NEW_TOKENS = 512 # 7B 数学推理通常需要更长步数
MAX_CHUNK_LEN = 15   # 遵循 SAFC 目标长度 [cite: 22]

print(f"📡 正在加载 7B 数学模型 (需较大显存)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
model.eval()

# 获取换行符 ID 用于 SAFC 逻辑 [cite: 22, 25]
BOUNDARY_ID = tokenizer.encode("\n")[-1]

def run_7b_math_visual_verification(n=200):
    dataset = load_dataset("gsm8k", "main", split="test", trust_remote_code=True)
    
    correct_vectors = []    
    incorrect_last_vectors = [] 
    X_features = []
    y_labels = []

    print(f"🚀 正在采集 7B 隐状态并进行逻辑审计...")
    for i in tqdm(range(n)):
        prompt = f"Please reason step by step, and put your final answer within \\boxed{{}}.\nQuestion: {dataset[i]['question']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        input_ids = inputs.input_ids
        past_key_values = None
        chunk_hiddens = []
        traj_blocks_features = [] 
        out_ids = inputs.input_ids
        
        for step in range(MAX_NEW_TOKENS):
            with torch.no_grad():
                outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
            
            logits = outputs.logits[0, -1, :]
            past_key_values = outputs.past_key_values
            
            # 提取 7B 模型最后一层隐状态 (维度通常为 3584) [cite: 8]
            last_hidden = outputs.hidden_states[-1][0, -1, :].cpu()
            chunk_hiddens.append(last_hidden)
            
            next_id = torch.argmax(logits).unsqueeze(0).unsqueeze(0)
            out_ids = torch.cat([out_ids, next_id], dim=-1)
            
            # SAFC 语义锚定切块逻辑 [cite: 22]
            if next_id.item() == BOUNDARY_ID or len(chunk_hiddens) >= MAX_CHUNK_LEN:
                if len(chunk_hiddens) > 0:
                    h_C_k = torch.stack(chunk_hiddens).mean(dim=0).to(torch.float32).numpy()
                    traj_blocks_features.append(h_C_k)
                    chunk_hiddens = []
            
            if next_id.item() == tokenizer.eos_token_id: break
            input_ids = next_id 
            
        # 结果自动打标 (Proxy for LLM-as-a-Judge)
        gen_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        # 提取答案数字进行对比
        gt_ans = re.findall(r'-?\d+', dataset[i]['answer'].replace(',', ''))[-1]
        model_part = gen_text.split("boxed{")[-1] if "boxed{" in gen_text else gen_text
        model_ans_list = re.findall(r'-?\d+', model_part.replace(',', ''))
        is_correct = (model_ans_list[-1] == gt_ans) if model_ans_list else False
        
        if traj_blocks_features:
            if is_correct:
                for feat in traj_blocks_features:
                    X_features.append(feat)
                    y_labels.append(0)
                    correct_vectors.append(feat)
            else:
                # 仅将错题的最后一个块判定为逻辑崩塌 (y=1) [cite: 11]
                X_features.append(traj_blocks_features[-1])
                y_labels.append(1)
                incorrect_last_vectors.append(traj_blocks_features[-1])

    # 计算几何偏移
    print("\n📊 分析 7B 隐空间相似度差异...")
    anchor_vector = np.mean(correct_vectors, axis=0).reshape(1, -1)
    sim_correct = cosine_similarity(np.array(correct_vectors), anchor_vector).flatten()
    sim_incorrect = cosine_similarity(np.array(incorrect_last_vectors), anchor_vector).flatten()

    # 绘图
    plt.figure(figsize=(8, 6))
    plt.style.use('seaborn-v0_8-muted')
    means = [np.mean(sim_correct), np.mean(sim_incorrect)]
    stds = [np.std(sim_correct), np.std(sim_incorrect)]
    labels = ['Correct Blocks\n(Safe)', 'Collapse Blocks\n(Error)']
    
    bars = plt.bar(labels, means, yerr=stds, color=['#55a868', '#c44e52'], capsize=8, alpha=0.9)
    plt.title('Latent Consistency Comparison (Qwen2.5-Math-7B)', fontsize=13)
    plt.ylabel('Cosine Similarity to Logic Anchor', fontsize=11)
    plt.ylim(0, 1.1)
    
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{bar.get_height():.4f}', ha='center', fontweight='bold')

    plt.savefig('7b_math_similarity_analysis.png', dpi=300)
    print(f"✅ 核心实验图已生成: 7b_math_similarity_analysis.png")

    # 验证 AUROC 是否能稳在 0.93 附近
    X_train, X_test, y_train, y_test = train_test_split(np.array(X_features), np.array(y_labels), test_size=0.2, stratify=y_labels)
    probe = LogisticRegression(class_weight='balanced').fit(X_train, y_train)
    auc_score = roc_auc_score(y_test, probe.predict_proba(X_test)[:, 1])
    print(f"\n🎯 7B 隐层探针最终 AUROC: 【 {auc_score:.4f} 】")

if __name__ == "__main__":
    run_7b_math_visual_verification(200)