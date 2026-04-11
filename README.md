# LEAP

这个仓库当前服务于一条固定的主线实验：

- 小模型先生成 chunk 级推理轨迹
- 32B strict judge 做 chunk-level prefix correctness 标注
- probe 学习风险信号
- observe-and-rollback scheduler 动态接管

现在仓库已经整理成适合跨数据集复用的结构，后续跑 `GSM8K / SVAMP / MATH500` 时，尽量沿着同一条流程走。

## 仓库结构

- `core_package/`
  主线代码。
- `dataset/`
  原始数据、chunk 轨迹、strict 标注数据。
- `evaluation/`
  单模型评测、FLOPs、失败分析、可视化。
- `result/`
  probe artifact、trace、json 输出、图。
- `unsorted/`
  历史材料和非主线代码。

## 配置方式

统一默认参数放在：

- [core_package/config.py](e:/软工/LEAP/core_package/config.py)

这里集中管理了：

- 模型默认路径
- system prompt
- dataset build 默认参数
- strict label 默认参数
- probe 训练默认参数
- scheduler 默认参数
- evaluation 默认参数

推荐原则：

- 想改全局默认值，就改 `core_package/config.py`
- 想做单次实验覆盖，就继续在命令行里传参数

这样既统一口径，也不会影响你灵活做实验。

## 运行前准备

以下命令默认都在仓库根目录执行，也就是：

```powershell
cd e:\软工\LEAP
```

远程 Linux 环境对应一般是：

```bash
cd ~/care_experiment
```

建议先确保这些目录存在：

```bash
mkdir -p dataset result/artifacts result/analysis_outputs result/traces
```

## 主线默认配置

当前主线默认配置是：

- probe 特征：
  `boundary+mean+relative_position+final_entropy+final_margin+final_top1_prob`
- probe 训练：
  `hidden_layers=128,32`
  `dropout=0.1`
  `epochs=60`
  `batch_size=256`
  `learning_rate=5e-4`
  `weight_decay=1e-3`
- 低熵错误加权：
  `final_entropy <= 1.0`
  `final_top1_prob >= 0.9`
  `weight = 4.0`
- scheduler：
  `max_handoffs=2`
  `large_handoff_chunks=2`
  `cooldown_chunks=1`
  `tail_bonus_weight=0.0`

如果你要完全复现主线，优先保持这些不变。

## 一条完整实验链怎么跑

无论什么数据集，主线都尽量走这 5 步：

1. 构建 1.5B chunk hidden-state 轨迹
2. 用 32B strict judge 标注 chunk
3. 训练 probe
4. 跑单模型 baseline
5. 跑 scheduler

下面分数据集写具体操作。

---

## GSM8K

### 1. 构建 chunk 轨迹

```bash
python -m core_package.pipelines.build_dataset \
  --dataset-name gsm8k \
  --dataset-split "train[:500]" \
  --model-path models/Qwen2.5-1.5B \
  --save-path dataset/gsm8k_15b_hidden_states.pt \
  --max-new-tokens 256
```

### 2. strict 标注

```bash
python -m core_package.pipelines.referee_32b_labeling_strict \
  --input-path dataset/gsm8k_15b_hidden_states.pt \
  --output-path dataset/gsm8k_labeled_training_data_strict.pt \
  --model-path models/Qwen2.5-32B \
  --num-samples 500 \
  --save-every 10 \
  --include-reference-answer
```

支持断点续跑。第一次跑完一部分后，如果中断了，继续执行时加上：

```bash
python -m core_package.pipelines.referee_32b_labeling_strict \
  --input-path dataset/gsm8k_15b_hidden_states.pt \
  --output-path dataset/gsm8k_labeled_training_data_strict.pt \
  --model-path models/Qwen2.5-32B \
  --num-samples 500 \
  --save-every 10 \
  --include-reference-answer \
  --resume
```

说明：

- `--resume` 会读取已有的 `output-path`
- 已经处理过的 `question_id` 会自动跳过
- `--save-every` 控制每处理多少个新问题就落盘一次

如果你担心夜里中断，建议把 `--save-every` 调小到 `5`。

### 3. 训练 probe

```bash
python -m core_package.probes.train_probe_artifact_torch \
  --label-path dataset/gsm8k_labeled_training_data_strict.pt \
  --output-path result/artifacts/probe_artifact_torch.pt \
  --feature-key "boundary+mean+relative_position+final_entropy+final_margin+final_top1_prob" \
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

### 4. 跑 1.5B / 32B baseline

1.5B:

```bash
python -m evaluation.evaluate_model_only_accuracy \
  --label-path dataset/gsm8k_labeled_training_data_strict.pt \
  --artifact-path result/artifacts/probe_artifact_torch.pt \
  --trace-path does_not_exist.json \
  --model-path models/Qwen2.5-1.5B \
  --max-new-tokens 768 \
  --output-path result/analysis_outputs/qwen25_15b_only_heldout.json
```

32B:

```bash
python -m evaluation.evaluate_model_only_accuracy \
  --label-path dataset/gsm8k_labeled_training_data_strict.pt \
  --artifact-path result/artifacts/probe_artifact_torch.pt \
  --trace-path does_not_exist.json \
  --model-path models/Qwen2.5-32B \
  --max-new-tokens 768 \
  --output-path result/analysis_outputs/qwen25_32b_only_heldout.json
```

### 5. 跑 scheduler

```bash
python -m core_package.schedulers.simulate_observe_rollback_scheduler \
  --label-path dataset/gsm8k_labeled_training_data_strict.pt \
  --probe-artifact-path result/artifacts/probe_artifact_torch.pt \
  --small-baseline-path result/analysis_outputs/qwen25_15b_only_heldout.json \
  --thresholds 0.25,0.40,0.50 \
  --tail-bonus-weight 0.0 \
  --max-new-tokens 768 \
  --max-handoffs 2 \
  --large-handoff-chunks 2 \
  --cooldown-chunks 2 \
  --trace-export-path result/traces/observe_rollback_traces_mainline.json
```

---

## SVAMP

你现在已经下好了：

- `dataset/svamp/train.jsonl`
- `dataset/svamp/test.jsonl`

推荐优先跑规范版：

- `train=700` 用来构建轨迹、strict 标注、训练 probe
- `test=300` 用来做最终单模型和 scheduler 评测

### 1. 构建 chunk 轨迹

train:

```bash
python -m core_package.pipelines.build_dataset \
  --dataset-name svamp \
  --input-path dataset/svamp/train.jsonl \
  --num-samples 700 \
  --model-path models/Qwen2.5-1.5B \
  --save-path dataset/svamp_15b_hidden_states.pt \
  --max-new-tokens 512
```

test:

```bash
python -m core_package.pipelines.build_dataset \
  --dataset-name svamp \
  --input-path dataset/svamp/test.jsonl \
  --num-samples 300 \
  --model-path models/Qwen2.5-1.5B \
  --save-path dataset/svamp_test_15b_hidden_states.pt \
  --max-new-tokens 512
```

### 2. strict 标注

```bash
python -m core_package.pipelines.referee_32b_labeling_strict \
  --input-path dataset/svamp_15b_hidden_states.pt \
  --output-path dataset/svamp_labeled_training_data_strict.pt \
  --model-path models/Qwen2.5-32B \
  --num-samples 700 \
  --save-every 10 \
  --include-reference-answer
```

同样支持断点续跑：

```bash
python -m core_package.pipelines.referee_32b_labeling_strict \
  --input-path dataset/svamp_15b_hidden_states.pt \
  --output-path dataset/svamp_labeled_training_data_strict.pt \
  --model-path models/Qwen2.5-32B \
  --num-samples 700 \
  --save-every 10 \
  --include-reference-answer \
  --resume
```

### 3. 训练 probe

```bash
python -m core_package.probes.train_probe_artifact_torch \
  --label-path dataset/svamp_labeled_training_data_strict.pt \
  --output-path result/artifacts/probe_artifact_svamp_torch.pt \
  --feature-key "boundary+mean+relative_position+final_entropy+final_margin+final_top1_prob" \
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

### 4. 跑 1.5B / 32B baseline

这里改成在 `test=300` 上评测。

1.5B:

```bash
python -m evaluation.evaluate_model_only_accuracy \
  --label-path dataset/svamp_test_15b_hidden_states.pt \
  --artifact-path does_not_exist.pt \
  --trace-path does_not_exist.json \
  --model-path models/Qwen2.5-1.5B \
  --max-new-tokens 768 \
  --output-path result/analysis_outputs/qwen25_15b_only_svamp_test.json
```

32B:

```bash
python -m evaluation.evaluate_model_only_accuracy \
  --label-path dataset/svamp_test_15b_hidden_states.pt \
  --artifact-path does_not_exist.pt \
  --trace-path does_not_exist.json \
  --model-path models/Qwen2.5-32B \
  --max-new-tokens 768 \
  --output-path result/analysis_outputs/qwen25_32b_only_svamp_test.json
```

### 5. 跑 scheduler

```bash
python -m core_package.schedulers.simulate_observe_rollback_scheduler \
  --label-path dataset/svamp_labeled_training_data_strict.pt \
  --eval-data-path dataset/svamp_test_15b_hidden_states.pt \
  --probe-artifact-path result/artifacts/probe_artifact_svamp_torch.pt \
  --small-baseline-path result/analysis_outputs/qwen25_15b_only_svamp_test.json \
  --thresholds 0.25,0.40,0.50 \
  --tail-bonus-weight 0.0 \
  --max-new-tokens 768 \
  --max-handoffs 2 \
  --large-handoff-chunks 2 \
  --cooldown-chunks 2 \
  --trace-export-path result/traces/observe_rollback_traces_svamp_test_mainline.json
```

---

## 新数据集怎么接

以后新增一个数学数据集，优先按这个思路接：

### 情况 1：和 SVAMP 类似，已经是本地 `jsonl`

直接用：

```bash
python -m core_package.pipelines.build_dataset \
  --dataset-name jsonl \
  --input-path path/to/your_dataset.jsonl \
  --question-field your_question_field \
  --answer-field your_answer_field \
  --save-path dataset/your_dataset_15b_hidden_states.pt
```

如果你已经有单独的 official `train/test`，推荐像上面的 `SVAMP` 一样：

- train 上做 strict 标注和 probe 训练
- test 上只构建轨迹
- 用 scheduler 的 `--eval-data-path` 在 test 上做最终评测

后面步骤不变，只要把：

- `label-path`
- `output-path`
- `trace-export-path`

换成新数据集自己的文件名即可。

### 情况 2：题目类型比 GSM8K 更复杂

例如 `MATH500`，主线框架仍然能复用，但你要额外确认：

- `answer_extraction.py` 能不能正确抽取答案
- 是否需要等价判定而不是字符串完全相等
- 是否需要调整 `max_new_tokens`

也就是说：

- build / strict label / probe / scheduler 这条框架可以复用
- answer extraction / correctness judge 可能要单独适配

## 结果文件通常放哪里

建议保持下面这个习惯，不同数据集不要混名：

- chunk 轨迹：
  `dataset/<name>_15b_hidden_states.pt`
- strict 标注：
  `dataset/<name>_labeled_training_data_strict.pt`
- probe artifact：
  `result/artifacts/probe_artifact_<name>_torch.pt`
- 单模型输出：
  `result/analysis_outputs/qwen25_<model>_only_<name>_test.json`
- scheduler trace：
  `result/traces/observe_rollback_traces_<name>_test_mainline.json`

## 常见注意事项

### 1. 一定从仓库根目录运行

否则相对路径容易乱。

### 2. 不同数据集尽量用不同输出文件名

避免覆盖：

- probe artifact
- 单模型 json
- trace json

### 3. 默认参数统一在 `core_package/config.py`

如果你发现多个脚本的默认值不一致，优先检查这里。

### 4. scheduler 复现优先依赖 artifact 和 baseline json

这样 held-out question ids 更稳定。

### 5. MATH500 不建议直接照抄 GSM8K 判对逻辑

先做 extraction 和 equivalence 检查，再上主线。

### 6. strict 标注建议默认按“可恢复任务”来跑

推荐习惯：

- 始终显式传 `--output-path`
- 长任务时加 `--resume`
- `--save-every` 不要设太大，推荐 `5` 或 `10`

这样即使中断，也只会损失最近一小段进度。

## 你以后最常用的两个模式

### 模式 A：完全复现主线

- 不改 `config.py`
- 命令行只换数据路径和输出文件名

### 模式 B：新数据集实验

- 保持 probe / scheduler 主线参数不变
- 只改：
  - `--input-path`
  - `--save-path`
  - `--label-path`
  - `--output-path`
  - `--trace-export-path`

这是目前最稳的做法。
