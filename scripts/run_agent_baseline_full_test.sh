#!/usr/bin/env bash
set -euo pipefail

# Allow: bash scripts/run_agent_baseline_full_test.sh MODEL_NAME=foo
for arg in "$@"; do
  case "${arg}" in
    *=*)
      export "${arg%%=*}=${arg#*=}"
      ;;
    *)
      echo "Unrecognized argument: ${arg}" >&2
      echo "Usage: bash scripts/run_agent_baseline_full_test.sh [KEY=VALUE ...]" >&2
      exit 1
      ;;
  esac
done

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-9B}"
MODEL_DIR="${MODEL_DIR:-${MODEL_NAME##*/}}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:19000}"
TOOL_SERVER_URL="${TOOL_SERVER_URL:-http://127.0.0.1:11536}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output_full_test}"
PROVIDER="${PROVIDER:-}"
BASE_URL="${BASE_URL:-}"
API_KEY="${API_KEY:-}"

actor_endpoint_args=()
if [[ -n "${PROVIDER}" ]]; then
  actor_endpoint_args+=(--provider "${PROVIDER}")
fi
if [[ -n "${BASE_URL}" ]]; then
  actor_endpoint_args+=(--base_url "${BASE_URL}")
fi
if [[ -n "${API_KEY}" ]]; then
  actor_endpoint_args+=(--api_key "${API_KEY}")
fi

datasets=(
  2WikiMultiHopQA
  HotpotQA
  MuSiQue
  FRAMES
  PopQA
  Bamboogle
)

for dataset in "${datasets[@]}"; do
  python -m run_qa_agent \
    --model_name "${MODEL_NAME}" \
    --server_url "${SERVER_URL}" \
    "${actor_endpoint_args[@]}" \
    --tool_server_url "${TOOL_SERVER_URL}" \
    --dataset_name "${dataset}" \
    --use_full_test_data \
    --save_dir "${OUTPUT_ROOT}/${dataset}/${MODEL_DIR}" \
    --max_workers 8 \
    --max_round 20 \
    --enable_llm_judge
done

# bash scripts/run_agent_baseline_full_test.sh MODEL_NAME=Qwen/Qwen3.5-9B
# bash scripts/run_agent_baseline_full_test.sh MODEL_NAME=google/gemma-4-12B-it
# bash scripts/run_agent_baseline_full_test.sh MODEL_NAME=mistralai/Ministral-3-8B-Instruct-2512-BF16
