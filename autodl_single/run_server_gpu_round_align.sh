#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

RUN_TAG="${FF_SERVER_RUN_TAG:-round_align}"
OUTPUT_DIR="${REPO_ROOT}/FlexFlow/inference/output/autodl_single/server_gpu_single_a100/${RUN_TAG}"
CACHE_DIR="${FF_CACHE_DIR:-/root/autodl-tmp/.cache/flexflow}"

NCPUS="${FF_SERVER_CPUS:-8}"
NUTILS="${FF_SERVER_UTILS:-8}"
NGPUS="${FF_SERVER_GPUS:-1}"
FSIZE_MB="${FF_SERVER_FSIZE_MB:-20000}"
ZSIZE_MB="${FF_SERVER_ZSIZE_MB:-60000}"
MAX_SEQ_LEN="${FF_SERVER_MAX_SEQ_LEN:-128}"
BATCH_SIZES="${FF_SERVER_BATCH_SIZES:-1 2 4 8 16}"
LLM_MODEL="${FF_LLM_MODEL:-huggyllama/llama-7b}"
SSM_MODEL="${FF_SSM_MODEL:-JackFram/llama-68m}"
MAX_TOKENS="${FF_SERVER_MAX_TOKENS:-}"
DISABLE_CACHE_FOLDER="${FF_SERVER_DISABLE_CACHE_FOLDER:-0}"
PROMPT_DIR="${FF_SERVER_PROMPT_DIR:-}"

if [[ -z "${PROMPT_DIR}" ]]; then
  if [[ -d "${REPO_ROOT}/FlexFlow/inference/prompt" ]]; then
    PROMPT_DIR="${REPO_ROOT}/FlexFlow/inference/prompt"
  elif [[ -d "${REPO_ROOT}/final_review_20260316_035130/prompts" ]]; then
    PROMPT_DIR="${REPO_ROOT}/final_review_20260316_035130/prompts"
  else
    echo "No prompt directory found. Set FF_SERVER_PROMPT_DIR explicitly." >&2
    exit 1
  fi
fi

mkdir -p "${OUTPUT_DIR}"

python "${REPO_ROOT}/autodl_single/write_manifest.py" \
  --output-dir "${OUTPUT_DIR}" \
  --experiment-name "server_gpu_single_a100_round_align" \
  --repo-root "${REPO_ROOT}" \
  --extra \
  "run_tag=${RUN_TAG}" \
  "llm_model=${LLM_MODEL}" \
  "ssm_model=${SSM_MODEL}" \
  "batch_sizes=${BATCH_SIZES}" \
  "prompt_dir=${PROMPT_DIR}" \
  "fsize_mb=${FSIZE_MB}" \
  "zsize_mb=${ZSIZE_MB}" \
  "max_seq_len=${MAX_SEQ_LEN}" \
  "max_tokens=${MAX_TOKENS:-<unset>}" \
  "disable_cache_folder=${DISABLE_CACHE_FOLDER}"

echo "Run tag: ${RUN_TAG}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Prompt dir: ${PROMPT_DIR}"
echo "Cache dir: ${CACHE_DIR}"
echo "Models: LLM=${LLM_MODEL} SSM=${SSM_MODEL}"
echo "Runtime: cpu=${NCPUS} util=${NUTILS} gpu=${NGPUS} fsize=${FSIZE_MB} zsize=${ZSIZE_MB} max_seq_len=${MAX_SEQ_LEN} max_tokens=${MAX_TOKENS:-<unset>}"
echo "Batch sizes: ${BATCH_SIZES}"
echo "Mode: round-aligned server benchmark"

cache_args=()
if [[ "${DISABLE_CACHE_FOLDER}" != "1" ]]; then
  cache_args=(-cache-folder "${CACHE_DIR}")
fi

max_token_args=()
if [[ -n "${MAX_TOKENS}" ]]; then
  max_token_args=(--max-tokens-per-batch "${MAX_TOKENS}")
fi

for bs in ${BATCH_SIZES}; do
  prompt_file="${PROMPT_DIR}/chatgpt_${bs}.json"
  if [[ ! -f "${prompt_file}" ]]; then
    echo "Missing prompt file: ${prompt_file}" >&2
    exit 1
  fi

  echo "===== batch size ${bs}: incr ====="
  stdbuf -oL -eL ./FlexFlow/build/inference/incr_decoding/incr_decoding \
    -ll:cpu "${NCPUS}" \
    -ll:util "${NUTILS}" \
    -ll:gpu "${NGPUS}" \
    -ll:fsize "${FSIZE_MB}" \
    -ll:zsize "${ZSIZE_MB}" \
    "${cache_args[@]}" \
    -llm-model "${LLM_MODEL}" \
    -prompt "${prompt_file}" \
    --max-requests-per-batch "${bs}" \
    "${max_token_args[@]}" \
    --max-sequence-length "${MAX_SEQ_LEN}" \
    -tensor-parallelism-degree "${NGPUS}" \
    --fusion \
    -output-file "${OUTPUT_DIR}/single_gpu-${bs}_incr_dec.txt" \
    2>&1 | tee "${OUTPUT_DIR}/single_gpu-${bs}_incr_dec.out"

  echo "===== batch size ${bs}: sequence specinfer ====="
  stdbuf -oL -eL ./FlexFlow/build/inference/spec_infer/spec_infer \
    -ll:cpu "${NCPUS}" \
    -ll:util "${NUTILS}" \
    -ll:gpu "${NGPUS}" \
    -ll:fsize "${FSIZE_MB}" \
    -ll:zsize "${ZSIZE_MB}" \
    "${cache_args[@]}" \
    -llm-model "${LLM_MODEL}" \
    -ssm-model "${SSM_MODEL}" \
    -prompt "${prompt_file}" \
    --max-requests-per-batch "${bs}" \
    "${max_token_args[@]}" \
    --max-sequence-length "${MAX_SEQ_LEN}" \
    --expansion-degree -1 \
    -tensor-parallelism-degree "${NGPUS}" \
    --fusion \
    -output-file "${OUTPUT_DIR}/single_gpu-${bs}_sequence_specinfer.txt" \
    2>&1 | tee "${OUTPUT_DIR}/single_gpu-${bs}_sequence_specinfer.out"

  echo "===== batch size ${bs}: tree specinfer ====="
  stdbuf -oL -eL ./FlexFlow/build/inference/spec_infer/spec_infer \
    -ll:cpu "${NCPUS}" \
    -ll:util "${NUTILS}" \
    -ll:gpu "${NGPUS}" \
    -ll:fsize "${FSIZE_MB}" \
    -ll:zsize "${ZSIZE_MB}" \
    "${cache_args[@]}" \
    -llm-model "${LLM_MODEL}" \
    -ssm-model "${SSM_MODEL}" \
    -prompt "${prompt_file}" \
    --max-requests-per-batch "${bs}" \
    "${max_token_args[@]}" \
    --max-sequence-length "${MAX_SEQ_LEN}" \
    -tensor-parallelism-degree "${NGPUS}" \
    --fusion \
    -output-file "${OUTPUT_DIR}/single_gpu-${bs}_tree_specinfer.txt" \
    2>&1 | tee "${OUTPUT_DIR}/single_gpu-${bs}_tree_specinfer.out"
done

python "${REPO_ROOT}/autodl_single/summarize_server_gpu_results.py" \
  --input-dir "${OUTPUT_DIR}"

cat "${OUTPUT_DIR}/summary.csv"
