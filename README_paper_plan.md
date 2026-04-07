# 论文推进说明

## 当前主线

目前面向论文的主线已经收束为：

- 监督信号：`strict label`
- 特征信号：`隐状态 + 熵/置信度`
- 调度机制：`chunk-level observe-and-rollback`
- 成本度量：`approximate FLOPs`

当前默认主线特征为：

```text
boundary+mean+relative_position+final_entropy+final_margin+final_top1_prob
```

这个默认配置目前写在：

- [train_probe_artifact_torch.py](/e:/软工/LEAP/probes/train_probe_artifact_torch.py)
- [simulate_observe_rollback_scheduler.py](/e:/软工/LEAP/schedulers/simulate_observe_rollback_scheduler.py)

也就是说，现在默认主线已经不是 heuristic 特征驱动，而是：

```text
hidden-state + uncertainty/confidence
```

启发式文本特征仍然保留在代码里，方便做消融，但已经不再作为默认主线的一部分。

## 当前探针训练配置

当前 PyTorch 探针训练流程已经包含以下基本防过拟合机制：

- 按 `question_id` 做 grouped split
- train / val / test 三段划分
- dropout
- weight decay
- early stopping
- 低熵错误 hard negative 重加权

当前默认超参数为：

- hidden layers：`128,32`
- dropout：`0.1`
- epochs：`60`
- batch size：`256`
- learning rate：`5e-4`
- weight decay：`1e-3`
- low-entropy error weight：`4.0`

## 当前探针会不会过拟合

结论是：

```text
当前训练流程已经具备基础的防过拟合设计，但还没有达到论文级别的稳健性。
```

原因如下。

### 已经具备的保护

- 不是 chunk 级随机打散，而是 question 级分组切分
- 有独立 validation split
- 有 early stopping
- 有 dropout 和 weight decay

### 仍然存在的风险

- 当前最有代表性的调度结果仍然主要来自较小 held-out 样本
- 结果还比较依赖单个 split / seed
- 当前最佳结果更多是单次实验，不是均值和方差
- validation 目前主要看 loss，还没有系统结合下游 routing 指标来选模型

所以更准确地说：

```text
目前不会被轻易定义成明显过拟合，但还需要多 seed、更大样本和更规范的验证流程，才能作为论文级结果。
```

## 全量实验前必须补的内容

### 1. 多随机种子

至少补 `3~5` 个 seed，并报告：

- 均值
- 标准差

建议统计：

- probe 分类指标
- 调度后的最终准确率
- FLOPs 比例

### 2. 用验证集选阈值

不要继续直接在测试集上扫阈值后挑最好结果。

规范流程应为：

- 在验证集上扫描 threshold
- 固定最佳 operating point
- 在测试集上只汇报一次正式结果

### 3. 做主线特征消融

至少要比较：

- `boundary`
- `mean`
- `boundary+mean`
- `boundary+mean+relative_position`
- `boundary+mean+relative_position+final_entropy+final_margin+final_top1_prob`

建议额外补一个：

- `last_non_punct`

这个消融非常重要，因为它直接对应：

```text
boundary_hidden 是否只是学到了标点边界
```

### 4. 补强 baseline

至少补：

- `SLM-only`
- `LLM-only`
- `7B-only`
- `entropy-only trigger`
- `top1-prob-only trigger`
- `random / fixed trigger`

### 5. 用真实 prompt token 做 FLOPs 统计

当前 FLOPs 分析脚本已经支持：

- token proxy
- approximate FLOPs

正式大样本实验前，应该重新导出新的 routing trace，并在每道题里保存真实的 `prompt_token_count`，而不是继续依赖固定的 `prompt-token-proxy`。

### 6. 扩大样本规模

在正式论文实验前，需要：

- 扩大测试集规模
- 必要时扩大训练集规模
- 在更大 held-out 集上重跑主线结果

## 建议的全量实验顺序

### 阶段 1：锁定主线

主线只保留：

- `strict label`
- `hidden + entropy/confidence`
- `observe-and-rollback`
- `approximate FLOPs`

不要在这一阶段再把 heuristic 特征重新放回正文主线。

### 阶段 2：训练默认主线 artifact

推荐命令：

```bash
python probes/train_probe_artifact_torch.py \
  --output-path probe_artifact_torch.pt \
  --label-key label \
  --hidden-layers 128,32 \
  --dropout 0.1 \
  --epochs 60 \
  --batch-size 256 \
  --learning-rate 5e-4 \
  --weight-decay 1e-3 \
  --low-entropy-error-final-entropy-max 1.0 \
  --low-entropy-error-final-top1-min 0.9 \
  --low-entropy-error-weight 4.0
```

### 阶段 3：用更高 token 上限跑调度

当前默认最大生成长度已经统一提高到 `768`，以减少“回答没说完就被截断判错”的问题。

推荐命令：

```bash
python schedulers/simulate_observe_rollback_scheduler.py \
  --probe-artifact-path probe_artifact_torch.pt \
  --thresholds 0.25,0.28,0.30,0.32,0.35 \
  --tail-bonus-weight 0.0 \
  --max-new-tokens 768 \
  --max-handoffs 2 \
  --large-handoff-chunks 2 \
  --cooldown-chunks 2 \
  --trace-export-path observe_rollback_traces_768.json
```

### 阶段 4：运行 FLOPs 分析

推荐命令：

```bash
python analysis/plot_threshold_accuracy_flops_compare.py \
  --trace-path observe_rollback_traces_768.json \
  --tail-bonus-weight 0.0 \
  --llm-accuracy 0.85 \
  --llm-token-proxy 302.05 \
  --cost-mode approx_flops \
  --prompt-token-proxy 120 \
  --output-dir analysis_outputs
```

后续如果 trace 里已经保存了真实 `prompt_token_count`，就可以去掉 `--prompt-token-proxy`。

### 阶段 5：做主线消融

先比较上面列出的主线特征组。

然后补：

- `7B-only`
- `entropy-only`
- `top1-prob-only`
- `random`

### 阶段 6：扩大样本并做多 seed

在主线稳定后：

- 扩大 held-out 测试题数量
- 跑多个随机种子
- 汇总均值和标准差

## 哪些内容放到后面做

下面这些内容是重要优化方向，但不应该阻塞当前论文主线：

- vLLM prefix caching
- cache-aware handoff
- 更复杂的 beneficial / rescue-aware 标签
- RL 风格 routing 目标

这些应该作为：

```text
主线稳定之后的优化模块或扩展方向
```

## 简短总结

当前论文主线可以概括成：

```text
基于 strict label 的 chunk 风险探测，通过隐状态与不确定度信号驱动 observe-and-rollback 调度，并在近似 FLOPs 约束下提升推理准确率。
```

全量实验前最重要的 6 件事是：

1. 多 seed 稳定性
2. 验证集选阈值
3. 主线特征消融
4. 强 baseline
5. 更大规模评测
6. 使用真实 prompt token 的 FLOPs 统计
