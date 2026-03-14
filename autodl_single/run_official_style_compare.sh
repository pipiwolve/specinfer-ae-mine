#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROMPT_FILE="${REPO_ROOT}/autodl_single/prompts/official_style_en_50.json"
OUTPUT_DIR="${REPO_ROOT}/FlexFlow/inference/output/autodl_single/official_style_en_50"

LLM_MODEL="${FF_LLM_MODEL:-huggyllama/llama-7b}"
SSM_MODEL="${FF_SSM_MODEL:-jackfram/llama-68m}"
CACHE_DIR="${FF_CACHE_DIR:-/root/autodl-tmp/.cache/flexflow}"

NCPUS="${FF_OFFICIAL_CPUS:-4}"
NUTILS="${FF_OFFICIAL_UTILS:-4}"
NGPUS="${FF_OFFICIAL_GPUS:-1}"
FSIZE_MB="${FF_OFFICIAL_FSIZE_MB:-34000}"
ZSIZE_MB="${FF_OFFICIAL_ZSIZE_MB:-30000}"
MAX_REQUESTS="${FF_OFFICIAL_MAX_REQUESTS:-1}"
MAX_TOKENS="${FF_OFFICIAL_MAX_TOKENS:-64}"
MAX_SEQ_LEN="${FF_OFFICIAL_MAX_SEQ_LEN:-64}"
EXPANSION_DEGREE="${FF_OFFICIAL_EXPANSION_DEGREE:--1}"

mkdir -p "${OUTPUT_DIR}"

echo "Prompt file: ${PROMPT_FILE}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Models: LLM=${LLM_MODEL} SSM=${SSM_MODEL}"
echo "Cache dir: ${CACHE_DIR}"
echo "Runtime: cpu=${NCPUS} util=${NUTILS} gpu=${NGPUS} fsize=${FSIZE_MB} zsize=${ZSIZE_MB} max_tokens=${MAX_TOKENS} max_seq_len=${MAX_SEQ_LEN}"

stdbuf -oL -eL ./FlexFlow/build/inference/incr_decoding/incr_decoding \
  -ll:cpu "${NCPUS}" \
  -ll:util "${NUTILS}" \
  -ll:gpu "${NGPUS}" \
  -ll:fsize "${FSIZE_MB}" \
  -ll:zsize "${ZSIZE_MB}" \
  -cache-folder "${CACHE_DIR}" \
  -llm-model "${LLM_MODEL}" \
  -prompt "${PROMPT_FILE}" \
  --max-requests-per-batch "${MAX_REQUESTS}" \
  --max-tokens-per-batch "${MAX_TOKENS}" \
  --max-sequence-length "${MAX_SEQ_LEN}" \
  -tensor-parallelism-degree "${NGPUS}" \
  --fusion \
  -output-file "${OUTPUT_DIR}/incr_official_style.txt" \
  2>&1 | tee "${OUTPUT_DIR}/incr_official_style.out"

stdbuf -oL -eL ./FlexFlow/build/inference/spec_infer/spec_infer \
  -ll:cpu "${NCPUS}" \
  -ll:util "${NUTILS}" \
  -ll:gpu "${NGPUS}" \
  -ll:fsize "${FSIZE_MB}" \
  -ll:zsize "${ZSIZE_MB}" \
  -cache-folder "${CACHE_DIR}" \
  -llm-model "${LLM_MODEL}" \
  -ssm-model "${SSM_MODEL}" \
  -prompt "${PROMPT_FILE}" \
  --max-requests-per-batch "${MAX_REQUESTS}" \
  --max-tokens-per-batch "${MAX_TOKENS}" \
  --max-sequence-length "${MAX_SEQ_LEN}" \
  --expansion-degree "${EXPANSION_DEGREE}" \
  -tensor-parallelism-degree "${NGPUS}" \
  --fusion \
  -output-file "${OUTPUT_DIR}/spec_official_style.txt" \
  2>&1 | tee "${OUTPUT_DIR}/spec_official_style.out"

echo "Official-style comparison finished."
