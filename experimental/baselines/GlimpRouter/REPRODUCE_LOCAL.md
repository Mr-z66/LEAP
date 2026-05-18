# GlimpRouter Local Reproduction Notes

This checkout is placed at `experimental/GlimpRouter` so it does not disturb the parent `LEAP` worktree.

## 1. Environment

Use Linux or WSL with CUDA for the full vLLM reproduction. The official dependency set targets Python 3.12:

```bash
conda create -n glimp_router python=3.12 -y
conda activate glimp_router
pip install -r requirements.txt
```

On Windows PowerShell, create and run the environment inside WSL when using vLLM.

## 2. Hardware Note

The default large model is `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`, which will not fit on an 8 GB GPU with normal vLLM settings. For local smoke tests, point `GLIMP_LARGE_MODEL` and `GLIMP_SMALL_MODEL` at smaller OpenAI-compatible vLLM servers, or run the servers on a remote GPU machine and set the base URLs below.

## 3. Model Endpoints

The code now keeps the upstream defaults but can be configured without editing Python files:

```bash
export GLIMP_API_KEY=glimp_router
export GLIMP_LARGE_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
export GLIMP_SMALL_MODEL=Qwen/Qwen3-4B-Thinking-2507
export GLIMP_LARGE_PORT=17125
export GLIMP_SMALL_PORT=17130

# Optional: use these when the endpoints are remote or not localhost.
export GLIMP_32B_BASE_URL=http://localhost:17125/v1
export GLIMP_4B_BASE_URL=http://localhost:17130/v1
```

Edit `server/serve.sh` if you want the repository to start both local vLLM servers for you. The placeholders that must be changed are `CUDA_DEVICE` and `PORT`.

## 4. Datasets

Public math datasets are loaded from Hugging Face by name:

- `aime24`
- `aime25`
- `math500`

Local JSONL datasets can be configured with environment variables:

```bash
export GLIMP_GPQA_DATA=../data/gpqa/gpqa_diamond_test.jsonl
export GLIMP_LCBV5_DATA=../data/lcbv5/test5.jsonl
export GLIMP_LCBV6_DATA=../data/lcbv6/test6.jsonl
```

You can download the LiveCodeBench helper files with:

```bash
bash setup.sh
```

## 5. Run

From the `src` directory:

```bash
bash run.sh
```

The script writes `src/config.json`, stores step traces under the configured result directory, and logs to `src/logs/`.

For a smaller first smoke test, edit `src/run.sh`:

```bash
DATASET_NAME="aime25"
REPEAT_NUM=1
TOKEN_BUDGET=1024
```

## 6. Evaluate

After inference finishes, adjust `eval/math_eval.sh` so `ANSWER_PATH_PREFIX` matches the generated result directory, then run:

```bash
cd eval
bash math_eval.sh
```
