# dataset

主线实验数据建议按数据集分目录存放。

当前推荐结构：

- `dataset/gsm8k/`
- `dataset/svamp/`
- `dataset/math500/`
- `dataset/livecodebench/`
- `dataset/audits/`

其中：

- 审计样本放在 `dataset/audits/`
- 代码评测集建议放在：
  - `dataset/livecodebench/v5/test5.jsonl`
  - `dataset/livecodebench/v6/test6.jsonl`

可以使用下面的脚本下载代码数据集：

```bash
bash dataset/download_code_datasets.sh
```
