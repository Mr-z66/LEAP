# LEAP

## Current Mainline Snapshot

As of 2026-04-28, the repository mainline has been aligned to the clean observe-and-rollback baseline that reproduced the best recent MATH500 result.

- Mainline probe feature spec: `boundary+mean`
- Mainline scheduler trigger rule: `require-consecutive-risk`
- Mainline adaptive handoff: enabled
- Mainline adaptive settings:
  - `min-large-handoff-chunks=1`
  - `max-adaptive-large-handoff-chunks=4`
  - `handoff-recovery-threshold=0.55`
  - `cooldown-chunks=2`
- Removed from the mainline scheduler:
  - sparse-risk trigger
  - multi-stable re-entry recovery
  - setup-suppression heuristics

### Current Best Clean MATH500 Setting

- Trace file: `result/traces/observe_rollback_traces_math500_vllm_hidden_only_t2048_adaptive_clean055.json`
- Threshold: `0.55`
- Small-only accuracy: `0.7200`
- Scheduled accuracy: `0.7800`
- Gain over small: `+0.0600`
- Trigger rate: `0.7000`

This is the current recommended clean baseline for MATH500 replication and follow-up ablations.

Exact reproduction command:

```bash
python -m core_package.schedulers.simulate_observe_rollback_scheduler \
  --label-path dataset/math500_labeled_data_strict_hf_t2048.pt \
  --eval-data-path dataset/math500_test_15b_hidden_states_hf_t2048.pt \
  --probe-artifact-path result/artifacts/probe_artifact_math500_hf_t2048_hidden_only.pt \
  --small-baseline-path result/analysis_outputs/qwen25_math_15b_only_math500_hf_t2048.json \
  --small-model-path models/Qwen2.5-Math-1.5B-Instruct \
  --large-model-path models/Qwen2.5-32B \
  --large-backend vllm \
  --vllm-base-url http://127.0.0.1:8000 \
  --vllm-api-key EMPTY \
  --vllm-model-name Qwen2.5-32B \
  --thresholds 0.55 \
  --max-new-tokens 2048 \
  --max-handoffs 2 \
  --large-handoff-chunks 2 \
  --adaptive-large-handoff \
  --min-large-handoff-chunks 1 \
  --max-adaptive-large-handoff-chunks 4 \
  --handoff-recovery-threshold 0.55 \
  --cooldown-chunks 2 \
  --require-consecutive-risk \
  --answer-type math500_qwen_boxed \
  --trace-export-path result/traces/observe_rollback_traces_math500_vllm_hidden_only_t2048_adaptive_clean055.json
```

This repository currently supports one main experimental workflow:

- a small model generates chunk-level reasoning trajectories
- a 32B judge produces strict chunk-level correctness labels
- a probe learns routing signals from those chunk features
- an observe-and-rollback scheduler decides when to hand off to the large model

The same workflow can be reused across `GSM8K`, `SVAMP`, and `MATH500` with minimal changes.

## Repository Structure

- `core_package/`
  main pipeline and scheduler code
- `dataset/`
  raw data, chunk trajectories, and strict labels
- `evaluation/`
  model-only evaluation, FLOPs analysis, failure analysis, and visualization
- `result/`
  probe artifacts, traces, JSON outputs, and figures
- `unsorted/`
  historical material and non-mainline code

## Configuration

Shared defaults live in:

- `core_package/config.py`

This file centralizes:

- default model paths
- system prompts
- dataset build defaults
- strict labeling defaults
- probe training defaults
- scheduler defaults
- evaluation defaults

Recommended practice:

- change `core_package/config.py` when you want to update global defaults
- override arguments on the command line for one-off experiments

## Before Running

All commands below assume you are in the repository root:

```powershell
cd <repo-root>
```

On the Linux server this is typically:

```bash
cd ~/care_experiment
```

Make sure these directories exist:

```bash
mkdir -p dataset result/artifacts result/analysis_outputs result/traces
```

## Mainline Defaults

The current mainline defaults are:

- probe features:
  `boundary+mean`
- probe training:
  `hidden_layers=128,32`
  `dropout=0.1`
  `epochs=60`
  `batch_size=256`
  `learning_rate=5e-4`
  `weight_decay=1e-3`
- low-entropy error weighting:
  `final_entropy <= 1.0`
  `final_top1_prob >= 0.9`
  `weight = 4.0`
- scheduler:
  `max_handoffs=2`
  `large_handoff_chunks=2`
  `cooldown_chunks=2`
  `tail_bonus_weight=0.0`

If you want to reproduce the mainline, keep these unchanged unless you are intentionally running an ablation.

## End-to-End Workflow

Across datasets, the mainline generally follows these five steps:

1. build 1.5B chunk hidden-state trajectories
2. label chunks with a 32B strict judge
3. train the probe
4. run model-only baselines
5. run the scheduler

Dataset-specific commands are listed below.

---

## GSM8K

### 1. 构建 chunk 轨迹

```bash
python -m core_package.pipelines.build_dataset \
  --dataset-name gsm8k \
  --dataset-split "train[:500]" \
  --model-path models/Qwen2.5-1.5B \
  --save-path dataset/gsm8k_15b_hidden_states.pt \
  --max-new-tokens  512
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
  --feature-key "boundary+mean" \
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
  --feature-key "boundary+mean" \
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
