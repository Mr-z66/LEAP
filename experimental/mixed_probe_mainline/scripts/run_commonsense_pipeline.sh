#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/LEAP}"
DATA="${DATA:-/root/autodl-tmp/LEAP_data/dataset}"
SMALL="${SMALL:-/root/autodl-tmp/models/Qwen2.5-1.5B}"
LARGE="${LARGE:-/root/autodl-tmp/models/Qwen2.5-7B}"
TRAJ="$DATA/commonsense_trajectories"
LABELS="$DATA/commonsense_labels"
ART="/root/autodl-tmp/LEAP_data/result/artifacts/commonsense"
LOG="$ROOT/result/logs/commonsense"
TRACE="$ROOT/result/traces/commonsense"

mkdir -p "$TRAJ" "$LABELS" "$ART" "$LOG" "$TRACE"
cd "$ROOT"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate care_env
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec > >(tee -a "$LOG/pipeline.log") 2>&1

wait_for_gpu_headroom() {
  local required_mib="$1"
  while true; do
    local used_mib
    used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if (( used_mib + required_mib <= 90000 )); then return 0; fi
    echo "[gpu wait] used=${used_mib}MiB required=${required_mib}MiB ceiling=90000MiB"
    sleep 60
  done
}

build_one() {
  local dataset="$1" split="$2" count="$3"
  local output="$TRAJ/${dataset}_${split}_${count}_15b.pt"
  [[ -f "$output" ]] && { echo "[skip build] $output"; return; }
  wait_for_gpu_headroom 6000
  python -m core_package.pipelines.build_dataset \
    --dataset-name jsonl --input-path "$DATA/mixed_probe_splits/${dataset}_${split}.jsonl" \
    --question-field question --answer-field answer --answer-type multiple_choice_letter \
    --num-samples "$count" --model-path "$SMALL" --save-path "$output" \
    --chunking-method rsd_step_fallback --step-word $'\n\n' --min-step-tokens 12 \
    --target-step-tokens 64 --max-step-tokens 96 --force-step-tokens 160 --max-new-tokens 256
}

label_one() {
  local dataset="$1"
  local input="$TRAJ/${dataset}_calib_300_15b.pt"
  local output="$LABELS/${dataset}_calib_stage1_7b.pt"
  [[ -f "$output" ]] && { echo "[skip label] $output"; return; }
  wait_for_gpu_headroom 18000
  python -m core_package.pipelines.label_existing_trajectories \
    --input-path "$input" --output-path "$output" --num-samples 300 \
    --judge-model-path "$LARGE" --judge-backend hf --max-judge-tokens 192 \
    --lookahead-steps 1 --save-every 5 --include-reference-answer --resume
}

refine_one() {
  local dataset="$1"
  local input="$LABELS/${dataset}_calib_stage1_7b.pt"
  local output="$LABELS/${dataset}_calib_stage2_7b.pt"
  [[ -f "$output" ]] && { echo "[skip refine] $output"; return; }
  wait_for_gpu_headroom 18000
  python -m core_package.pipelines.refine_clean_step_labels_second_pass \
    --input-path "$input" --output-path "$output" \
    --judge-model-path "$LARGE" --judge-backend hf --max-judge-tokens 384 \
    --low-confidence-threshold 0.55 --refine-existing-errors
}

train_one() {
  local dataset="$1"
  local labels="$LABELS/${dataset}_calib_routing_conservative.pt"
  local artifact="$ART/probe_${dataset}_routing_boundary_mean.pt"
  local pos_weight
  pos_weight=$(python - "$labels" <<'PY'
import sys, torch
rows = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
labels = [int(row["routing_label"]) for row in rows]
print(max(labels.count(0) / max(labels.count(1), 1), 0.05))
PY
)
  if [[ -f "$artifact" ]]; then
    echo "[skip probe train] $artifact"
  else
    python -m core_package.probes.train_probe_artifact_torch \
      --label-path "$labels" --output-path "$artifact" --feature-key boundary+mean \
      --label-key routing_label --hidden-layers 128,32 --dropout 0.1 --epochs 80 \
      --batch-size 128 --learning-rate 5e-4 --weight-decay 1e-3 --pos-weight "$pos_weight"
  fi
  python -m core_package.probes.evaluate_probe_baseline_torch \
    --data-path "$labels" --artifact-path "$artifact" \
    --threshold-grid 0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90 \
    > "$LOG/probe_eval_${dataset}.log" 2>&1
  python - "$artifact" "$TRAJ/${dataset}_calib_300_15b.pt" "$TRAJ/${dataset}_calib_holdout.pt" <<'PY'
import sys, torch
artifact = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
ids = set(artifact["test_question_ids"])
rows = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
torch.save([row for row in rows if row["question_id"] in ids], sys.argv[3])
PY
}

schedule() {
  local dataset="$1" eval_path="$2" count="$3" thresholds="$4" trace_path="$5"
  wait_for_gpu_headroom 24000
  python -m core_package.schedulers.simulate_observe_rollback_scheduler \
    --label-path "$LABELS/${dataset}_calib_routing_conservative.pt" --eval-data-path "$eval_path" \
    --probe-artifact-path "$ART/probe_${dataset}_routing_boundary_mean.pt" \
    --small-model-path "$SMALL" --large-model-path "$LARGE" --large-backend hf \
    --thresholds "$thresholds" --num-test-questions "$count" --max-new-tokens 256 \
    --runtime-chunking rsd_step --runtime-step-word $'\n\n' --max-handoffs 1 --max-trigger-progress 0.75 \
    --handoff-mode takeover --require-consecutive-risk --rewrite-step-min-tokens 12 \
    --rewrite-step-target-tokens 64 --rewrite-step-force-tokens 160 \
    --rewrite-step-boundary-mode auto --large-handoff-chunks 8 --cooldown-chunks 2 \
    --large-extra-token-budget 128 --answer-repair-tokens 48 \
    --answer-type multiple_choice_letter --small-model-params-b 1.5 --large-model-params-b 7.0 \
    --trace-export-path "$trace_path"
}

echo "[pipeline] start $(date -Is)"
build_one commonsenseqa calib 300
build_one commonsenseqa test 300
build_one arc_challenge calib 300
build_one arc_challenge test 299
label_one commonsenseqa
label_one arc_challenge
refine_one commonsenseqa
refine_one arc_challenge

# Keep the trajectory-level second pass conservative: it may propagate a
# first-pass explicit error forward, but it may not move the first error to an
# earlier merely incomplete/verbose chunk.
python - "$LABELS" <<'PY'
import collections, pathlib, re, sys, torch
root = pathlib.Path(sys.argv[1])
for dataset in ("commonsenseqa", "arc_challenge"):
    stage1 = torch.load(root / f"{dataset}_calib_stage1_7b.pt", map_location="cpu", weights_only=False)
    stage2 = torch.load(root / f"{dataset}_calib_stage2_7b.pt", map_location="cpu", weights_only=False)
    first = {(str(row["question_id"]), int(row["chunk_id"])): row for row in stage1}
    groups = collections.defaultdict(list)
    for row in stage2:
        key = (str(row["question_id"]), int(row["chunk_id"]))
        source = first[key]
        row["label"] = int(source["label"])
        row["label_source"] = source.get("label_source", "judge")
        groups[str(row["question_id"])].append(row)
    propagated = 0
    for chunks in groups.values():
        chunks.sort(key=lambda row: int(row["chunk_id"]))
        if bool(chunks[0].get("is_final_correct", False)):
            continue
        anchors = [int(row["chunk_id"]) for row in chunks if int(row["label"]) == 0]
        if not anchors:
            gold = str(chunks[0].get("ground_truth_final_answer", "")).strip().upper()
            for row in chunks:
                answers = re.findall(r"(?im)\bAnswer\s*:\s*([A-E])\b", str(row.get("prefix_text", "")))
                if answers and gold in set("ABCDE") and answers[-1].upper() != gold:
                    anchors.append(int(row["chunk_id"]))
                    break
        if not anchors:
            chunks[-1]["label"] = -1
            chunks[-1]["label_source"] = "second_pass_unresolved_final_wrong"
            continue
        anchor = min(anchors)
        for row in chunks:
            if int(row["chunk_id"]) >= anchor and int(row["label"]) != 0:
                row["label"] = 0
                row["label_source"] = "second_pass_conservative_monotone"
                propagated += 1
    torch.save(stage2, root / f"{dataset}_calib_stage2_7b.pt")
    print(f"{dataset} conservative_stage2", collections.Counter(int(row["label"]) for row in stage2), "propagated", propagated)
PY

python - "$LABELS" <<'PY'
import collections, pathlib, sys, torch
root = pathlib.Path(sys.argv[1])
for dataset in ("commonsenseqa", "arc_challenge"):
    rows = torch.load(root / f"{dataset}_calib_stage2_7b.pt", map_location="cpu", weights_only=False)
    for row in rows:
        row["routing_label"] = 1 if int(row["label"]) == 1 else 0
    print(dataset, collections.Counter(int(row["routing_label"]) for row in rows))
    torch.save(rows, root / f"{dataset}_calib_routing_conservative.pt")
PY

train_one commonsenseqa
train_one arc_challenge

for dataset in commonsenseqa arc_challenge; do
  schedule "$dataset" "$TRAJ/${dataset}_calib_holdout.pt" 60 \
    0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80 "$TRACE/${dataset}_calib_threshold_sweep.json" \
    > "$LOG/scheduler_calib_${dataset}.log" 2>&1
  best=$(python - "$TRACE/${dataset}_calib_threshold_sweep.json" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1], encoding="utf-8"))
best = max(rows, key=lambda row: (row["scheduled_accuracy"], -row["avg_param_weighted_token_cost"]))
print(best["threshold"])
PY
)
  count=300; [[ "$dataset" == arc_challenge ]] && count=299
  schedule "$dataset" "$TRAJ/${dataset}_test_${count}_15b.pt" "$count" "$best" \
    "$TRACE/${dataset}_test_best_threshold.json" > "$LOG/scheduler_test_${dataset}.log" 2>&1
done

echo "[pipeline] complete $(date -Is)"
