# LEAP

LEAP 当前首先实现的是一条“启发式切块 + 大模型打标 + 轻量探针验证”的监督学习 baseline。整条流程的目标不是马上做强化学习，而是先验证一个更基础的问题：

在启发式切块下，小模型隐藏状态能否判断当前推理前缀是否已经偏离正确逻辑轨道？

目前这条 baseline 的角色分工如下：

- `Qwen2.5-1.5B`：生成 GSM8K 的逐步推理轨迹，并提取隐藏状态
- 启发式切块器：按标点和长度阈值把生成轨迹切成多个 chunk
- `Qwen2.5-32B`：对每个 `prefix_text` 做 prefix-level 逻辑判断
- 线性 probe：在无泄漏分组评估下验证隐藏状态是否存在可学习信号

## 一、Baseline 目标

在进入自适应切块或强化学习之前，项目需要先把以下四件事做扎实：

- 构建干净、可追踪的 chunk 级数据
- 获取可靠的 prefix-level 监督标签
- 用按题分组的方式做无泄漏评估
- 能够做基础的误差分析和人工审计

只有这一步成立，后续再引入自适应切块或 RL 才不会把问题一起放大。

## 二、主要脚本说明

- [build_dataset.py](build_dataset.py)：用 `Qwen2.5-1.5B` 生成推理轨迹，提取隐藏状态，并按启发式规则切块
- [referee_32b_labeling.py](referee_32b_labeling.py)：用 `Qwen2.5-32B` 对每个前缀块做结构化逻辑打标
- [analyze_labeled_data.py](analyze_labeled_data.py)：统计标签分布、切块分布、judge 质量和首次错误位置
- [sample_judge_audit.py](sample_judge_audit.py)：随机导出一批样本，供人工审计 judge 标签
- [verify_idea/evaluate_probe_baseline.py](verify_idea/evaluate_probe_baseline.py)：训练并评估 probe，对比不同隐藏状态特征

## 三、Baseline 运行顺序

请按下面顺序执行。

### 1. 构建启发式切块数据

```powershell
python build_dataset.py
```

输出文件：

- `gsm8k_15b_hidden_states.pt`

该文件包含：

- 题目文本
- 模型生成的完整推理文本
- 最终答案是否正确
- 每个 chunk 的 token span
- `boundary_hidden_state`
- `mean_hidden_state`
- `prefix_text`

### 2. 使用 32B 裁判模型打标

```powershell
python referee_32b_labeling.py
```

输出文件：

- `gsm8k_labeled_training_data.pt`

该文件会在 chunk 数据基础上补充：

- `label`
- `judge_confidence`
- `judge_error_type`
- `judge_reason`
- `judge_raw_response`
- `judge_parse_status`

### 3. 分析标注后的数据分布

```powershell
python analyze_labeled_data.py
```

该脚本会输出：

- 正负标签比例
- 每题 chunk 数量分布
- 各类 `cut_reason` 分布
- judge 解析成功率
- judge confidence 统计
- 首次错误块通常出现在什么位置

### 4. 评估启发式监督学习 baseline

```powershell
python verify_idea\evaluate_probe_baseline.py
```

该脚本会对比两类特征：

- `mean_hidden_state`
- `boundary_hidden_state`

输出指标包括：

- AUROC
- AUPRC
- classification report

注意：

- 评估使用 `GroupShuffleSplit`
- 分组键是 `question_id`
- 这样可以避免同一道题拆出的多个 chunk 同时进入训练集和测试集，从而造成数据泄漏

### 5. 导出人工审计样本

```powershell
python sample_judge_audit.py
```

输出文件：

- `judge_audit_samples.jsonl`

你可以人工查看这些样本，核对 32B 打出来的 prefix-level 标签是否可信。

## 四、推荐的结果查看顺序

一轮 baseline 跑完之后，建议按下面顺序看结果：

1. `python analyze_labeled_data.py`
2. `python verify_idea\evaluate_probe_baseline.py`
3. `python sample_judge_audit.py`

这样可以更快判断当前问题主要出在哪一层：

- 是切块策略不合理
- 还是 judge 标签噪声太大
- 还是隐藏状态信号本身较弱
- 还是评估协议存在类别不平衡或分布问题

## 五、当前 baseline 的意义

这一版 baseline 主要是为了确认下面这件事是否成立：

隐藏状态中是否真的存在能够被轻量探针提取出来的“逻辑偏航信号”。

如果这一点成立，后续就可以继续推进：

1. 做更多特征和层选择的 ablation
2. 做更严格的 judge 人工一致性审计
3. 尝试监督式自适应切块
4. 最后再进入强化学习切块

## 六、后续建议路线

建议按下面的顺序推进，而不是直接跳到 PPO 一类的强化学习：

1. 启发式切块 + 监督学习 probe
2. 人工审计 judge 标签
3. 特征与层的对比实验
4. 监督式边界预测模型
5. 自适应切块的离线学习
6. 最后再尝试强化学习

核心原则是：

先把监督链路做干净，再把 RL 当成切块策略升级，而不是当前阶段的问题修复工具。
