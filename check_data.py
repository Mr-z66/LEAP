import torch

# 加载数据
data_path = "gsm8k_15b_hidden_states.pt"
dataset = torch.load(data_path)

print(f"✅ 成功加载数据集，共包含 {len(dataset)} 道题目。")

# 随便挑一道题看看内部结构
sample = dataset[0]
print(f"\n📝 示例题目: {sample['question'][:50]}...")
print(f"✂️ 该题目被切成了 {len(sample['chunks'])} 个语义块。")

# 检查第一个切块的隐向量
first_chunk = sample['chunks'][0]
print(f"👉 第一个块的内容: {first_chunk['chunk_text']}")
print(f"   ↳ 隐向量维度: {first_chunk['hidden_state'].shape}") # 应该是 [1536]
print(f"   ↳ 存储位置: {first_chunk['hidden_state'].device}")   # 应该是 cpu