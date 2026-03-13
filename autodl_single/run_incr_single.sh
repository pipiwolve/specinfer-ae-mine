#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

cd "${REPO_ROOT}"

OUTPUT_DIR="${REPO_ROOT}/FlexFlow/inference/output/autodl_single/smoke"
PROMPT_FILE="${REPO_ROOT}/autodl_single/prompts/smoke_test.json"
mkdir -p "${OUTPUT_DIR}"

./FlexFlow/build/inference/incr_decoding/incr_decoding \
  -ll:cpu 8 \
  -ll:util 8 \
  -ll:gpu 1 \
  -ll:fsize 30000 \
  -ll:zsize 120000 \
  -cache-folder "${FF_CACHE_DIR}" \
  -llm-model huggyllama/llama-7b \
  -prompt "${PROMPT_FILE}" \
  --max-requests-per-batch 1 \
  --max-tokens-per-batch 128 \
  --max-sequence-length 128 \
  -tensor-parallelism-degree 1 \
  --fusion \
  -output-file "${OUTPUT_DIR}/incr_single.txt" \
  > "${OUTPUT_DIR}/incr_single.out" 2>&1

echo "Incremental smoke test finished: ${OUTPUT_DIR}/incr_single.out"
