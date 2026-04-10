# LEAP

主线仓库已经整理成 5 个顶层区块：

- `core_package/`
  主线代码。后续做 GSM8K 以外的数据集，也优先在这里扩展。
- `evaluation/`
  评测、作图、失败分析、导出脚本。
- `result/`
  trace、图、JSON、case export 等实验结果。
- `dataset/`
  数据文件与审计样本。
- `unsorted/`
  历史支线、旧实验、非主线代码。

## 当前主线

当前论文主线固定为：

- strict chunk-level label
- probe 特征：
  - `boundary+mean+relative_position+final_entropy+final_margin+final_top1_prob`
- observe-and-rollback
- 原始 handoff
- 新版答案抽取

主结果 operating points：

- 高准确率点：`threshold = 0.25`
- 高效率点：`threshold = 0.40` 或 `0.50`

## 目录说明

### `core_package/`

- `core_package/pipelines/`
  - `build_dataset.py`
  - `referee_32b_labeling_strict.py`
  - `count_labeled_questions.py`
- `core_package/probes/`
  - `train_probe_artifact_torch.py`
  - `evaluate_probe_baseline_torch.py`
- `core_package/schedulers/`
  - `simulate_observe_rollback_scheduler.py`
- `core_package/tooling/`
  - `download_model.py`
  - `check_data.py`
- `core_package/answer_extraction.py`

### `evaluation/`

- model-only baseline 评测
- FLOPs/threshold 作图
- 标签统计、失败分析、case 导出

### `result/`

- `result/analysis_outputs/`
- `result/traces/`
- `result/summaries/`
- `result/scheduler_case_exports/`

### `dataset/`

- `dataset/audits/`
- 后续可放：
  - `gsm8k_labeled_training_data_strict.pt`
  - 其他数据集的 `.pt/.jsonl`

### `unsorted/`

- `unsorted/legacy_experiments/`
- `unsorted/legacy_code/`

这里只保留历史材料，不再作为主线入口。

## 建议的主线命令

以下命令都在仓库根目录执行。

### 1. 构建 chunk hidden-state 数据

```powershell
python -m core_package.pipelines.build_dataset
```

### 2. strict 标注

```powershell
python -m core_package.pipelines.referee_32b_labeling_strict --num-samples 100 --save-every 5
```

### 3. 训练主线 probe

```powershell
python -m core_package.probes.train_probe_artifact_torch ^
  --label-path dataset/gsm8k_labeled_training_data_strict.pt ^
  --output-path result/artifacts/probe_artifact_torch.pt ^
  --feature-key "boundary+mean+relative_position+final_entropy+final_margin+final_top1_prob" ^
  --hidden-layers 128,32 ^
  --dropout 0.1 ^
  --epochs 60 ^
  --batch-size 256 ^
  --learning-rate 5e-4 ^
  --weight-decay 1e-3 ^
  --low-entropy-error-final-entropy-max 1.0 ^
  --low-entropy-error-final-top1-min 0.9 ^
  --low-entropy-error-weight 4.0
```

### 4. 评估 probe

```powershell
python -m core_package.probes.evaluate_probe_baseline_torch ^
  --data-path dataset/gsm8k_labeled_training_data_strict.pt ^
  --artifact-path result/artifacts/probe_artifact_torch.pt
```

### 5. 运行 scheduler

```powershell
python -m core_package.schedulers.simulate_observe_rollback_scheduler ^
  --probe-artifact-path result/artifacts/probe_artifact_torch.pt ^
  --small-baseline-path result/analysis_outputs/qwen25_15b_only_heldout100_newextract.json ^
  --thresholds 0.25,0.40,0.50 ^
  --tail-bonus-weight 0.0 ^
  --max-new-tokens 768 ^
  --max-handoffs 2 ^
  --large-handoff-chunks 2 ^
  --cooldown-chunks 2 ^
  --trace-export-path result/traces/observe_rollback_traces_mainline.json
```

### 6. 画 accuracy-FLOPs 图

```powershell
python -m evaluation.plot_threshold_accuracy_flops_compare ^
  --trace-path result/traces/observe_rollback_traces_mainline.json ^
  --tail-bonus-weight 0.0 ^
  --llm-accuracy 0.94 ^
  --llm-token-proxy 306.66 ^
  --cost-mode approx_flops ^
  --output-dir result/analysis_outputs
```
