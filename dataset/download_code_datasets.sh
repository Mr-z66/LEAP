#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${ROOT_DIR}/dataset/livecodebench"

TARGET="${1:-all}"

LCBV5_URL="https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/test5.jsonl"
LCBV6_URL="https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/test6.jsonl"

download_file() {
  local url="$1"
  local out_path="$2"

  mkdir -p "$(dirname "$out_path")"

  if [[ -f "$out_path" ]]; then
    echo "Already exists: $out_path"
    return 0
  fi

  echo "Downloading: $url"
  echo "Output: $out_path"

  if command -v wget >/dev/null 2>&1; then
    wget -O "$out_path" "$url"
  elif command -v curl >/dev/null 2>&1; then
    curl -L "$url" -o "$out_path"
  else
    echo "Error: neither wget nor curl is available." >&2
    exit 1
  fi
}

case "$TARGET" in
  v5)
    download_file "$LCBV5_URL" "${DATASET_ROOT}/v5/test5.jsonl"
    ;;
  v6)
    download_file "$LCBV6_URL" "${DATASET_ROOT}/v6/test6.jsonl"
    ;;
  all)
    download_file "$LCBV5_URL" "${DATASET_ROOT}/v5/test5.jsonl"
    download_file "$LCBV6_URL" "${DATASET_ROOT}/v6/test6.jsonl"
    ;;
  *)
    echo "Usage: bash dataset/download_code_datasets.sh [v5|v6|all]" >&2
    exit 1
    ;;
esac

echo
echo "Done. Suggested environment variables:"
echo "  export GLIMP_LCBV5_DATA=\"${DATASET_ROOT}/v5/test5.jsonl\""
echo "  export GLIMP_LCBV6_DATA=\"${DATASET_ROOT}/v6/test6.jsonl\""
