#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

OUTPUT_DIR="${REPO_ROOT}/FlexFlow/inference/output/autodl_single/server_gpu_single_a100"
CACHE_DIR="${FF_CACHE_DIR:-/root/autodl-tmp/.cache/flexflow}"

NCPUS="${FF_SERVER_CPUS:-8}"
NUTILS="${FF_SERVER_UTILS:-8}"
NGPUS="${FF_SERVER_GPUS:-1}"
FSIZE_MB="${FF_SERVER_FSIZE_MB:-21890}"
ZSIZE_MB="${FF_SERVER_ZSIZE_MB:-80000}"
MAX_SEQ_LEN="${FF_SERVER_MAX_SEQ_LEN:-128}"
MAX_TOKENS="${FF_SERVER_MAX_TOKENS:-128}"
LLM_MODEL="${FF_LLM_MODEL:-huggyllama/llama-7b}"
SSM_MODEL="${FF_SSM_MODEL:-jackfram/llama-68m}"
BATCH_SIZES="${FF_SERVER_BATCH_SIZES:-1 2 4 8}"

mkdir -p "${OUTPUT_DIR}"

echo "Output dir: ${OUTPUT_DIR}"
echo "Cache dir: ${CACHE_DIR}"
echo "Models: LLM=${LLM_MODEL} SSM=${SSM_MODEL}"
echo "Runtime: cpu=${NCPUS} util=${NUTILS} gpu=${NGPUS} fsize=${FSIZE_MB} zsize=${ZSIZE_MB} max_seq_len=${MAX_SEQ_LEN} max_tokens=${MAX_TOKENS}"
echo "Batch sizes: ${BATCH_SIZES}"

for bs in ${BATCH_SIZES}; do
  prompt_file="${REPO_ROOT}/FlexFlow/inference/prompt/chatgpt_${bs}.json"
  if [[ ! -f "${prompt_file}" ]]; then
    echo "Missing prompt file: ${prompt_file}" >&2
    echo "Run ./download_dataset.sh first." >&2
    exit 1
  fi

  echo "===== batch size ${bs}: incr ====="
  stdbuf -oL -eL ./FlexFlow/build/inference/incr_decoding/incr_decoding \
    -ll:cpu "${NCPUS}" \
    -ll:util "${NUTILS}" \
    -ll:gpu "${NGPUS}" \
    -ll:fsize "${FSIZE_MB}" \
    -ll:zsize "${ZSIZE_MB}" \
    -cache-folder "${CACHE_DIR}" \
    -llm-model "${LLM_MODEL}" \
    -prompt "${prompt_file}" \
    --max-requests-per-batch "${bs}" \
    --max-tokens-per-batch "${MAX_TOKENS}" \
    --max-sequence-length "${MAX_SEQ_LEN}" \
    -tensor-parallelism-degree "${NGPUS}" \
    --fusion \
    -output-file "${OUTPUT_DIR}/server_small-${bs}_batchsize-incr_dec.txt" \
    2>&1 | tee "${OUTPUT_DIR}/server_small-${bs}_batchsize-incr_dec.out"

  echo "===== batch size ${bs}: sequence specinfer ====="
  stdbuf -oL -eL ./FlexFlow/build/inference/spec_infer/spec_infer \
    -ll:cpu "${NCPUS}" \
    -ll:util "${NUTILS}" \
    -ll:gpu "${NGPUS}" \
    -ll:fsize "${FSIZE_MB}" \
    -ll:zsize "${ZSIZE_MB}" \
    -cache-folder "${CACHE_DIR}" \
    -llm-model "${LLM_MODEL}" \
    -ssm-model "${SSM_MODEL}" \
    -prompt "${prompt_file}" \
    --max-requests-per-batch "${bs}" \
    --max-tokens-per-batch "${MAX_TOKENS}" \
    --max-sequence-length "${MAX_SEQ_LEN}" \
    --expansion-degree -1 \
    -tensor-parallelism-degree "${NGPUS}" \
    --fusion \
    -output-file "${OUTPUT_DIR}/server_small-${bs}_batchsize-sequence_specinfer.txt" \
    2>&1 | tee "${OUTPUT_DIR}/server_small-${bs}_batchsize-sequence_specinfer.out"

  echo "===== batch size ${bs}: tree specinfer ====="
  stdbuf -oL -eL ./FlexFlow/build/inference/spec_infer/spec_infer \
    -ll:cpu "${NCPUS}" \
    -ll:util "${NUTILS}" \
    -ll:gpu "${NGPUS}" \
    -ll:fsize "${FSIZE_MB}" \
    -ll:zsize "${ZSIZE_MB}" \
    -cache-folder "${CACHE_DIR}" \
    -llm-model "${LLM_MODEL}" \
    -ssm-model "${SSM_MODEL}" \
    -prompt "${prompt_file}" \
    --max-requests-per-batch "${bs}" \
    --max-tokens-per-batch "${MAX_TOKENS}" \
    --max-sequence-length "${MAX_SEQ_LEN}" \
    -tensor-parallelism-degree "${NGPUS}" \
    --fusion \
    -output-file "${OUTPUT_DIR}/server_small-${bs}_batchsize-tree_specinfer.txt" \
    2>&1 | tee "${OUTPUT_DIR}/server_small-${bs}_batchsize-tree_specinfer.out"
done

echo "Server GPU single-A100 benchmark finished."
