#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

export ROOT_DIR
export ANSWER_AWARE_STOP="${ANSWER_AWARE_STOP:-1}"
export TRACE_TAG="${TRACE_TAG:-mixed_5to1_fixed4_15b_to_32b_hf_answer_aware}"

if [[ "${ANSWER_AWARE_STOP}" == "1" ]]; then
  export EXTRA_SCHEDULER_ARGS="${EXTRA_SCHEDULER_ARGS:-} --answer-aware-stop"
fi

echo "[experiment] answer_aware_stop=${ANSWER_AWARE_STOP}"
echo "[experiment] trace_tag=${TRACE_TAG}"

exec bash "${SCRIPT_DIR}/run_scheduler_eval_15b_to_32b_hf.sh"
