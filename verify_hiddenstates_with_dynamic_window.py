import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn.functional as F
import numpy as np
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

DEVICE = "cuda"
MODEL_ID = "Qwen/Qwen2.5-Math-7B-Instruct"
MAX_NEW_TOKENS = 512
# 动态切块的安全边界：防止切得太细或太粗
MIN_CHUNK_LEN = 5   
MAX_CHUNK_LEN = 40  

print(f"📡 正在加载 7B 模型并注入动态切块逻辑...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
)
model.eval()

def run_dynamic_chunk_probe(n=200):
    dataset = load_dataset("gsm8k", "main", split="test", cache_dir="./gsm8k_cache")
    X_features, y_labels = [], []

    print(f"🚀 开始执行【方差熵波峰】动态切块实验 (n={n})...")
    for i in tqdm(range(n)):
        prompt = f"Please reason step by step, and put your final answer within \\boxed{{}}.\nQuestion: {dataset[i]['question']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        input_ids, past_key_values = inputs.input_ids, None
        chunk_hiddens, traj_blocks = [], []
        out_ids = inputs.input_ids
        
        # 用于监测波峰的方差熵序列
        v_history = [] 

        for step in range(MAX_NEW_TOKENS):
            with torch.no_grad():
                outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
            
            logits = outputs.logits[0, -1, :]
            past_key_values = outputs.past_key_values
            
            # --- 核心：实时方差熵计算 ---
            probs = F.softmax(logits, dim=-1)
            log_p = torch.log(probs + 1e-10)
            ent = -torch.sum(probs * log_p)
            varentropy = torch.sum(probs * ((-log_p) - ent)**2).item()
            v_history.append(varentropy)
            
            # 提取隐状态
            last_hidden = outputs.hidden_states[-1][0, -1, :].cpu().to(torch.float32)
            chunk_hiddens.append(last_hidden)
            
            next_id = torch.argmax(logits).unsqueeze(0).unsqueeze(0)
            out_ids = torch.cat([out_ids, next_id], dim=-1)
            
            # --- 👑 核心创新点：方差熵波峰检测逻辑 ---
            is_peak = False
            if len(v_history) >= 3:
                # 简单的波峰判定：低 -> 高 -> 低 序列
                if v_history[-2] > v_history[-3] and v_history[-2] > v_history[-1]:
                    is_peak = True
            
            is_too_long = (len(chunk_hiddens) >= MAX_CHUNK_LEN)
            is_eos = (next_id.item() == tokenizer.eos_token_id)
            
            # 只有在过了最小块长度后，才允许波峰触发切块
            should_cut = (is_peak and len(chunk_hiddens) >= MIN_CHUNK_LEN) or is_too_long or is_eos
            
            if should_cut and len(chunk_hiddens) > 0:
                h_C_k = torch.stack(chunk_hiddens).mean(dim=0).numpy()
                traj_blocks.append(h_C_k)
                chunk_hiddens = []
                # 重置波峰历史，开始监测下一个决策周期
                v_history = [] 
                    
            if is_eos: break
            input_ids = next_id 
            
        # 结果校验
        gen_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        gt_ans = re.findall(r'-?\d+', dataset[i]['answer'].replace(',', ''))[-1]
        model_part = gen_text.split("boxed{")[-1] if "boxed{" in gen_text else gen_text
        model_part = model_part.replace(',', '') 
        model_ans_list = re.findall(r'-?\d+', model_part)
        is_correct = (model_ans_list[-1] == gt_ans) if model_ans_list else False
        
        # 打标与收集
        if traj_blocks:
            if is_correct:
                for b in traj_blocks:
                    X_features.append(b); y_labels.append(0)
            else:
                X_features.append(traj_blocks[-1]); y_labels.append(1)

    # 训练验证
    X, y = np.array(X_features), np.array(y_labels)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    probe = LogisticRegression(max_iter=1000, class_weight='balanced')
    probe.fit(X_train, y_train)
    auc_score = roc_auc_score(y_test, probe.predict_proba(X_test)[:, 1])
    
    print("\n" + "="*50)
    print(f"🎯 动态方差熵切块 + 隐层探针 AUROC: 【 {auc_score:.4f} 】")
    print(f"📊 块级样本数: {len(y)} | 正负样本比: {list(y).count(0)}:{list(y).count(1)}")
    print("="*50)

if __name__ == "__main__":
    run_dynamic_chunk_probe(200)