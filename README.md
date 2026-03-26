# LEAP

LEAP 当前先做的是一条可验证、可审计的监督学习 baseline：

- 启发式切块
- 大模型做 `prefix_correct` 标注
- 小模型 hidden state 上训练轻量 probe

这条 baseline 的目标不是直接做强化学习，而是先验证一个更基础的问题：

> 在启发式 chunk 边界上，小模型 hidden state 里是否存在可被 probe 读出的 prefix-level 逻辑错误信号？

当前主线已经统一为：

- 主标注标准：`strict prefix_correct`
- 主标签文件：`gsm8k_labeled_training_data_strict.pt`
- 主特征：`boundary_hidden_state`
- 主评估重点：错误类指标，而不只看整体 AUROC / AUPRC

## 1. 当前实验链路

- `Qwen2.5-1.5B`
  生成 GSM8K 推理轨迹，提取 hidden states。
- 启发式切块器
  按标点和长度阈值把轨迹切成多个 chunk。
- `Qwen2.5-32B`
  对每个 `prefix_text` 做 `prefix_correct` 判断。
- 轻量 probe
  在无泄漏的按题分组评估下，验证 hidden state 中是否存在可学习信号。

## 2. 主要脚本

- [build_dataset.py](build_dataset.py)
  生成推理轨迹、hidden states 和 chunk 数据。
- [referee_32b_labeling.py](referee_32b_labeling.py)
  宽松版 judge，输出 `gsm8k_labeled_training_data.pt`。
- [referee_32b_labeling_strict.py](referee_32b_labeling_strict.py)
  严格版 judge，输出 `gsm8k_labeled_training_data_strict.pt`。
- [analyze_labeled_data.py](analyze_labeled_data.py)
  统计标签分布、切块分布、judge 输出质量，并支持宽松版和严格版对照。
- [sample_judge_audit.py](sample_judge_audit.py)
  随机导出审计样本，人工检查 judge 标签。
- [evaluate_probe_baseline.py](evaluate_probe_baseline.py)
  主 probe 评估脚本。默认使用 strict 数据，默认主特征为 `boundary_hidden_state`。
- [verify_idea/evaluate_probe_baseline.py](verify_idea/evaluate_probe_baseline.py)
  兼容旧路径的入口脚本，实际转发到根目录的主评估脚本。

## 3. 推荐运行顺序

### 3.1 构建 chunk 数据

```powershell
python build_dataset.py
```

输出：

- `gsm8k_15b_hidden_states.pt`

该文件包含：

- 问题文本
- 模型完整推理轨迹
- 最终答案及其是否正确
- 每个 chunk 的 token span
- `prefix_text`
- `boundary_hidden_state`
- `mean_hidden_state`

### 3.2 用 strict judge 做主标注

先小样本试跑：

```powershell
python referee_32b_labeling_strict.py --num-samples 100 --save-every 5
```

继续扩样本到 300：

```powershell
python referee_32b_labeling_strict.py --num-samples 300 --resume --save-every 5
```

如需在首个错误后停止该题后续 chunk 标注：

```powershell
python referee_32b_labeling_strict.py --num-samples 300 --resume --save-every 5 --stop-after-first-error
```

strict 输出文件：

- `gsm8k_labeled_training_data_strict.pt`

每个 chunk 会补充：

- `label`
- `judge_confidence`
- `judge_error_type`
- `judge_reason`
- `judge_raw_response`
- `judge_parse_status`

### 3.3 分析 strict 标签数据

```powershell
python analyze_labeled_data.py --strict
```

如果要直接对照 relaxed vs strict：

```powershell
python analyze_labeled_data.py --compare-default-pair
```

输出重点包括：

- 正负样本比例
- 每题 chunk 数分布
- `cut_reason` 分布
- judge JSON 解析成功率
- confidence 统计
- 首次错误 chunk 出现位置

### 3.4 用 strict 数据跑 probe

默认主线：

```powershell
python evaluate_probe_baseline.py
```

或使用兼容旧路径：

```powershell
python verify_idea/evaluate_probe_baseline.py
```

当前默认设置：

- 数据：`gsm8k_labeled_training_data_strict.pt`
- 主特征：`boundary_hidden_state`
- 切分方式：`GroupShuffleSplit`
- 分组键：`question_id`

如需同时和 `mean_hidden_state` 做对照：

```powershell
python evaluate_probe_baseline.py --features boundary_hidden_state mean_hidden_state
```

### 3.5 导出人工审计样本

```powershell
python sample_judge_audit.py
```

输出：

- `judge_audit_samples.jsonl`

用于人工检查 strict judge 是否：

- 抓到了合理的早期逻辑错误
- 没有把“只是没写完”的 chunk 大量误判为 0

### 3.6 运行 chunk 级离线调度模拟

第一版调度系统采用 `small -> large` 的 chunk 级接管：

- 小模型：使用已有 `Qwen2.5-1.5B` 轨迹作为默认生成路径
- 风险分数：用 probe 对每个 chunk 边界打分
- 大模型：当风险超过阈值时，由 `Qwen2.5-32B` 从当前 `prefix_text` 接管并完成后续生成

运行命令：

```powershell
python simulate_chunk_scheduler.py --probe-type mlp --mlp-hidden-layers 512,128,32 --thresholds 0.15,0.20,0.25
```

该脚本会输出：

- 小模型单跑准确率
- 大模型整题单跑准确率
- 调度系统准确率
- 触发率
- 首次触发位置
- 首次错误覆盖率
- 近错误覆盖率
- 潜在可挽救错误比例
- 近似大模型接管成本

建议先用单个 split 做调度原型验证，再根据最有潜力的阈值继续补更完整的系统实验。

## 4. 当前 probe 评估指标

当前主评估已经不只看整体 AUROC / AUPRC，而是重点关注错误类表现。

输出包括：

- `prefix_correct_auroc`
- `prefix_correct_auprc`
- `error_auprc`
- `balanced_accuracy`
- `error_precision`
- `error_recall`
- `error_f1`
- confusion matrix
- classification report

best split 默认按 `error_f1` 选，不再按单纯 AUROC 选。

## 5. 为什么当前主线选择 strict + boundary

目前已有的小样本结果表明：

- relaxed 版负类太少，grouped split 稳定性较差
- strict 版能抓到更多合理的 prefix-level 错误
- `boundary_hidden_state` 比 `mean_hidden_state` 更稳定、更强

因此当前主线统一为：

- 标注标准用 strict
- 主特征优先用 `boundary_hidden_state`
- 主指标优先看错误类检测能力

## 6. 拿到 300 题 strict 数据后的下一步

建议按下面顺序推进：

1. 先检查标签分布是否健康
   重点看 error ratio、questions with error、first-error 位置。
2. 跑 strict probe 主实验
   先只跑 `boundary_hidden_state`，确认错误类指标是否比 100 题更稳定。
3. 做一次 boundary vs mean 对照
   验证主特征选择是否稳固。
4. 做错误样本审计
   人工看一批 strict 负例，确认新增负类是否大多合理。
5. 记录主实验基线
   固定住 300 题 strict + boundary 的结果，作为后续扩展对照基线。
6. 再考虑下一层模型改动
   比如更强 probe、更多 hidden 层、更多特征组合，而不是立刻跳到 RL。

## 7. 当前 baseline 的意义

这条 baseline 主要是在验证：

> hidden state 中是否真的存在可被轻量 probe 提取出来的 prefix-level 逻辑偏航信号

如果这一点成立，后续才值得继续推进：

- 更系统的特征消融
- 更严格的人审一致性分析
- 自适应切块
- 更强的探针结构
- 最后再进入强化学习阶段

## 8. 当前建议

当前建议不要直接跳到 PPO 或其他 RL 方案，而是先把下面这条链路做扎实：

1. strict chunk labeling
2. 人工审计 judge 标签
3. boundary hidden state probe
4. 错误类指标驱动的评估
5. 稳定基线后再做更复杂方法

核心原则是：

先把监督链路做干净，再把 RL 当作后续策略升级，而不是当前阶段的补丁。
