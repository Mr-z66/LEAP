# LEAP

当前仓库现在有两层组织方式：

1. 按职责放代码
2. 按两条 baseline 给入口

## 先看这两个入口

- `strict_baseline/`
  对应 `strict prefix_correct -> probe -> single-handoff` 主线
- `beneficial_baseline/`
  对应 `takeover_beneficial -> beneficial probe -> multi/long handoff` 主线

如果你只是想跑实验，优先看这两个目录里的 `README.md` 和 `run_mainline.txt`。

## 代码目录

- `pipelines/`
  数据构建、judge 标注、beneficial 标签构建
- `probes/`
  probe 评估与 artifact 训练
- `schedulers/`
  single-handoff 与 multi-handoff 调度模拟
- `analysis/`
  标签分析、失败归因、bad case 导出
- `tooling/`
  下载模型、快速检查等工具脚本
- `multi_beneficial_handoff/`
  独立的 beneficial multi-handoff 支线环境与运行模板
- `verify_idea/`
  历史验证材料与兼容入口

## 主线脚本位置

### 数据与标注

- `pipelines/build_dataset.py`
- `pipelines/referee_32b_labeling.py`
- `pipelines/referee_32b_labeling_strict.py`
- `pipelines/build_takeover_beneficial_labels.py`

### Probe

- `probes/evaluate_probe_baseline.py`
- `probes/train_probe_artifact.py`

### Scheduler

- `schedulers/simulate_chunk_scheduler.py`
- `schedulers/simulate_multi_handoff_scheduler.py`

### 分析

- `analysis/analyze_labeled_data.py`
- `analysis/audit_strict_label_quality.py`
- `analysis/sample_judge_audit.py`
- `analysis/analyze_scheduler_failures.py`
- `analysis/export_missed_trigger_cases.py`

### 工具

- `tooling/download_model.py`
- `tooling/check_data.py`

## 常用命令

在仓库根目录执行。

### 1. 构建 hidden-state / chunk 数据

```powershell
python pipelines/build_dataset.py
```

### 2. strict 标注

```powershell
python pipelines/referee_32b_labeling_strict.py --num-samples 100 --save-every 5
```

### 3. probe 评估

```powershell
python probes/evaluate_probe_baseline.py
```

### 4. single-handoff 调度

```powershell
python schedulers/simulate_chunk_scheduler.py --thresholds 0.10,0.15,0.20,0.25
```
