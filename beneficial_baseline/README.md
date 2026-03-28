# Beneficial Baseline

这条主线对应你后面单独拉出来的 `takeover_beneficial` 路线。

## 目标

- 不再学 `prefix_correct`
- 直接学习“这里接管值不值”
- 以 beneficial 标签训练 router
- 用 multi-handoff 或 single long handoff 做收益验证

## 关键脚本

### 标签构建
- `../pipelines/build_takeover_beneficial_labels.py`

### Probe
- `../probes/train_probe_artifact.py`
  这里通过 `--label-key takeover_beneficial` 复用训练器

### Scheduler
- `../schedulers/simulate_multi_handoff_scheduler.py`

### 独立环境与支线说明
- `../multi_beneficial_handoff/README.md`
- `../multi_beneficial_handoff/bootstrap_env.sh`
- `../multi_beneficial_handoff/run_smoke_test.sh`

## 数据文件

- `gsm8k_takeover_beneficial_labels.pt`
- `beneficial_probe_artifact.pt`
- `takeover_beneficial_cache.pt`

## 开始前建议清理

```powershell
Remove-Item gsm8k_takeover_beneficial_labels.pt -ErrorAction SilentlyContinue
Remove-Item beneficial_probe_artifact.pt -ErrorAction SilentlyContinue
Remove-Item takeover_beneficial_cache.pt -ErrorAction SilentlyContinue
```

## 常用命令

### 1. 构建 beneficial 标签
```powershell
python pipelines/build_takeover_beneficial_labels.py --small-model-path models/Qwen2.5-1.5B --model-path models/Qwen2.5-32B --feature-key boundary+mean+relative_position+final_entropy+final_margin+final_top1_prob --candidate-mode hybrid --top-k 2 --explore-positions middle,last --large-handoff-chunks 2 --max-new-tokens 256 --only-small-wrong --num-questions 20
```

### 2. 训练 beneficial probe
```powershell
python probes/train_probe_artifact.py --label-path gsm8k_takeover_beneficial_labels.pt --label-key takeover_beneficial --feature-key boundary+mean+relative_position+final_entropy+final_margin+final_top1_prob --output-path beneficial_probe_artifact.pt
```

### 3. 跑 multi-handoff
```powershell
python schedulers/simulate_multi_handoff_scheduler.py --probe-artifact-path beneficial_probe_artifact.pt --thresholds 0.01,0.02,0.05,0.10,0.15,0.20 --large-handoff-chunks 2 --max-handoffs 2 --num-test-questions 20
```
