# Multi Beneficial Handoff

这是一条与当前 `strict single-handoff` 主线隔离的实验支线，目标是专门验证：

- `takeover_beneficial` 标签是否适合做路由监督
- `multi-handoff` 或 `single long handoff` 是否能比当前 strict 主线更有效
- 新动作定义下的 probe / scheduler 是否真的带来最终收益

## 目录用途

- `requirements.txt`
  这条支线的最小 Python 依赖
- `bootstrap_env.sh`
  在 Linux 服务器上创建隔离虚拟环境并安装依赖
- `run_smoke_test.sh`
  第一轮小样本 smoke test 命令模板
- `configs/`
  留给后续支线配置文件
- `notes/`
  留给实验记录和失败分析

## 建议运行位置

推荐在服务器仓库根目录执行这些命令，例如：

```bash
cd ~/care_experiment
```

然后：

```bash
cd multi_beneficial_handoff
bash bootstrap_env.sh
source .venv/bin/activate
bash run_smoke_test.sh
```

## 当前支线约定

- 标签：`takeover_beneficial`
- 特征：
  `boundary+mean+relative_position+final_entropy+final_margin+final_top1_prob`
- 第一轮动作：
  - `large_handoff_chunks = 2`
  - `max_handoffs = 2`
- 第一轮目标：
  先验证这条线是否能产生正增益，而不是直接追求最终最优

## 清理建议

这条支线开始前，建议至少清掉：

```bash
rm -f gsm8k_takeover_beneficial_labels.pt
rm -f takeover_beneficial_cache.pt
rm -f beneficial_probe_artifact.pt
```

这样可以避免旧版 beneficial 标签和缓存污染新实验。
