# Strict Baseline

这条主线对应你最早跑通的 `strict prefix_correct` 路线。

## 目标

- 用 32B judge 给 chunk prefix 打 `strict` 标签
- 用 1.5B hidden state + confidence 特征训练 probe
- 用 single-handoff scheduler 做 small-to-large 接管

## 关键脚本

### 数据与标注
- `../pipelines/build_dataset.py`
- `../pipelines/referee_32b_labeling_strict.py`

### Probe
- `../probes/evaluate_probe_baseline.py`
- `../probes/train_probe_artifact.py`

### Scheduler
- `../schedulers/simulate_chunk_scheduler.py`

### Analysis
- `../analysis/analyze_labeled_data.py`
- `../analysis/audit_strict_label_quality.py`
- `../analysis/sample_judge_audit.py`
- `../analysis/analyze_scheduler_failures.py`
- `../analysis/export_missed_trigger_cases.py`

## 数据文件

- `gsm8k_15b_hidden_states.pt`
- `gsm8k_labeled_training_data_strict.pt`
- `probe_artifact.pt`
- `chunk_scheduler_cache.pt`

## 常用命令

### 1. 构建轨迹与 chunk
```powershell
python pipelines/build_dataset.py
```

### 2. strict 标注
```powershell
python pipelines/referee_32b_labeling_strict.py --num-samples 100 --save-every 5
```

### 3. probe 评估
```powershell
python probes/evaluate_probe_baseline.py --probe-type mlp --features boundary+mean+relative_position boundary+mean+relative_position+final_entropy+final_margin+final_top1_prob
```

### 4. 训练 artifact
```powershell
python probes/train_probe_artifact.py --feature-key boundary+mean+relative_position+final_entropy+final_margin+final_top1_prob
```

### 5. single-handoff 调度
```powershell
python schedulers/simulate_chunk_scheduler.py --probe-artifact-path probe_artifact.pt --thresholds 0.10,0.15,0.20,0.25 --tail-bonus-weight 0.05
```
