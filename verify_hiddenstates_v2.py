import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import numpy as np
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

DEVICE = "cuda"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # 0.5B 模型，适合快速验证，后续可以换成 7B
MAX_NEW_TOKENS = 256
MAX_CHUNK_LEN = 15

print(f"📡 正在加载 0.5B 模型 (约需 8GB 显存)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
)
model.eval()

BOUNDARY_ID = tokenizer.encode("\n")[-1]

def run_probe_mvp(n=200):
    print(f"📊 正在加载 GSM8K 测试集...")
    dataset = load_dataset("gsm8k", "main", split="test", trust_remote_code=True)
    
    # 存放高维隐状态 (X) 和 标签 (y)
    X_features = []
    y_labels = []

    print(f"🚀 开始采集高维隐状态 (样本数: {n})...")
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
                outputs = model(
                    input_ids=input_ids, 
                    past_key_values=past_key_values, 
                    use_cache=True,
                    output_hidden_states=True # 核心：开启隐层输出
                )
            
            logits = outputs.logits[0, -1, :]
            past_key_values = outputs.past_key_values
            
            # 提取 3584 维向量并立刻移到 CPU，防止显存爆炸
            last_hidden = outputs.hidden_states[-1][0, -1, :].cpu()
            chunk_hiddens.append(last_hidden)
            
            next_id = torch.argmax(logits).unsqueeze(0).unsqueeze(0)
            out_ids = torch.cat([out_ids, next_id], dim=-1)
            
            is_boundary = (next_id.item() == BOUNDARY_ID)
            is_too_long = (len(chunk_hiddens) >= MAX_CHUNK_LEN)
            is_eos = (next_id.item() == tokenizer.eos_token_id)
            
            if is_boundary or is_too_long or is_eos:
                if len(chunk_hiddens) > 0:
                    # 平均池化，融合成一个代表整个块的向量
                    h_C_k = torch.stack(chunk_hiddens).mean(dim=0).to(torch.float32).numpy()
                    traj_blocks_features.append(h_C_k)
                    chunk_hiddens = []
                    
            if is_eos: break
            input_ids = next_id 
            
        # 核对对错
        gen_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        gt_ans = re.findall(r'-?\d+', dataset[i]['answer'].replace(',', ''))[-1]
        model_part = gen_text.split("boxed{")[-1] if "boxed{" in gen_text else gen_text
        model_part = model_part.replace(',', '') 
        model_ans_list = re.findall(r'-?\d+', model_part)
        is_correct = (model_ans_list[-1] == gt_ans) if model_ans_list else False
        
        # ==========================================================
        # 👑 极简客观打标 (Proxy for PRM)
        # ==========================================================
        if not traj_blocks_features: continue
            
        if is_correct:
            # 算对的题：整条轨迹都是安全的，所有块提取特征，打 0
            for feat in traj_blocks_features:
                X_features.append(feat)
                y_labels.append(0)
        else:
            # 算错的题：只取最后 1 个块作为崩溃原爆点，打 1。前面的扔掉！
            X_features.append(traj_blocks_features[-1])
            y_labels.append(1)

    # ==========================================================
    # 🧠 秒级训练探针 (Logistic Regression)
    # ==========================================================
    print("\n" + "="*50)
    print("🏆 CARE v2 隐层探针 (Hidden State Probe) 训练与验证")
    print("="*50)
    
    X = np.array(X_features)
    y = np.array(y_labels)
    
    if len(set(y)) < 2:
        print("⚠️ 样本太单一，无法训练分类器。请增加 n 的数量！")
        return
        
    print(f"👉 成功采集块样本数: {len(y)} (安全块: {list(y).count(0)} | 崩溃块: {list(y).count(1)})")
    
    # 切分训练集和测试集 (80% 训练，20% 测试)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 训练极轻量探针 (加上 class_weight='balanced' 对抗样本不平衡)
    print("⚙️ 正在训练 Logistic Regression 探针...")
    probe = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    probe.fit(X_train, y_train)
    
    # 预测并计算 AUROC
    y_pred_proba = probe.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n🚀 见证奇迹的时刻：")
    print(f"🎯 隐层探针在未见数据上的 AUROC: 【 {auc_score:.4f} 】")
    
    if auc_score > 0.70:
        print("\n👑 结论：Idea 彻底验证成功！隐状态中存在极其强烈的崩溃信号！")
        print("下一步：可以安心用 PRM 去做更细粒度的离线打标，准备写论文了！")
    else:
        print("\n⚠️ 结论：信号不够强烈，可能需要引入更复杂的探针网络 (如 MLP) 或增大样本量。")

if __name__ == "__main__":
    run_probe_mvp(200) # 今晚先跑 200 道题光速验证