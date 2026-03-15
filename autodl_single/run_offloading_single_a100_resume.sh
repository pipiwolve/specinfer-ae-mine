#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

cd "${REPO_ROOT}"

OUTPUT_ROOT="${REPO_ROOT}/FlexFlow/inference/output/autodl_single/offloading_single_a100"
RUN_TAG="${FF_OFFLOAD_RUN_TAG:-default}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_TAG}"
mkdir -p "${OUTPUT_DIR}"

NCPUS="${FF_OFFLOAD_CPUS:-8}"
NUTILS="${FF_OFFLOAD_UTILS:-8}"
NGPUS="${FF_OFFLOAD_GPUS:-1}"
FSIZE_MB="${FF_OFFLOAD_FSIZE_MB:-21000}"
ZSIZE_MB="${FF_OFFLOAD_ZSIZE_MB:-50000}"
MAX_SEQ_LEN="${FF_OFFLOAD_MAX_SEQ_LEN:-256}"
BATCH_SIZES="${FF_OFFLOAD_BATCH_SIZES:-1 2 4 8}"
SMALL_LLM="${FF_OFFLOAD_SMALL_LLM:-facebook/opt-13b}"
LARGE_LLM="${FF_OFFLOAD_LARGE_LLM:-facebook/opt-30b}"
SSM_MODEL="${FF_OFFLOAD_SSM:-facebook/opt-125m}"
RUN_SMALL="${FF_OFFLOAD_RUN_SMALL:-1}"
RUN_LARGE="${FF_OFFLOAD_RUN_LARGE:-0}"
SMALL_RESERVE_MB="${FF_OFFLOAD_SMALL_RESERVE_MB:-500}"
LARGE_RESERVE_MB="${FF_OFFLOAD_LARGE_RESERVE_MB:-700}"
PROMPT_FILE="${FF_OFFLOAD_PROMPT_FILE:-${REPO_ROOT}/FlexFlow/inference/prompt/chatgpt_offloading.json}"
SEQUENCE_MODE="${FF_OFFLOAD_SEQUENCE_MODE:-1}"

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "Missing prompt file: ${PROMPT_FILE}" >&2
  echo "Run ./download_dataset.sh first." >&2
  exit 1
fi

run_spec_job() {
  local label="$1"
  local llm_model="$2"
  local reserve_mb="$3"
  local bs="$4"
  local output_prefix="${OUTPUT_DIR}/${label}_${bs}"
  local txt_file="${output_prefix}.txt"
  local out_file="${output_prefix}.out"

  if [[ -f "${out_file}" ]] && grep -q -- "----------inference finished--------------" "${out_file}"; then
    echo "Skipping completed job: ${label} batch=${bs}"
    return 0
  fi

  rm -f "${txt_file}" "${out_file}"

  echo "===== offloading job ${label} batch=${bs} ====="
  stdbuf -oL -eL ./FlexFlow/build/inference/spec_infer/spec_infer \
    -ll:cpu "${NCPUS}" \
    -ll:util "${NUTILS}" \
    -ll:gpu "${NGPUS}" \
    -ll:fsize "${FSIZE_MB}" \
    -ll:zsize "${ZSIZE_MB}" \
    -cache-folder "${FF_CACHE_DIR}" \
    -llm-model "${llm_model}" \
    -ssm-model "${SSM_MODEL}" \
    -prompt "${PROMPT_FILE}" \
    --max-requests-per-batch "${bs}" \
    --max-sequence-length "${MAX_SEQ_LEN}" \
    ${SEQUENCE_MODE:+--expansion-degree -1} \
    -offload \
    -offload-reserve-space-size "${reserve_mb}" \
    -output-file "${txt_file}" \
    2>&1 | tee "${out_file}"
}

echo "Run tag: ${RUN_TAG}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Prompt file: ${PROMPT_FILE}"
echo "Runtime: cpu=${NCPUS} util=${NUTILS} gpu=${NGPUS} fsize=${FSIZE_MB} zsize=${ZSIZE_MB} max_seq_len=${MAX_SEQ_LEN}"
echo "Batches: ${BATCH_SIZES}"
echo "Models: small=${SMALL_LLM} large=${LARGE_LLM} ssm=${SSM_MODEL}"

if [[ "${RUN_SMALL}" == "1" ]]; then
  for bs in ${BATCH_SIZES}; do
    run_spec_job "offloading_small" "${SMALL_LLM}" "${SMALL_RESERVE_MB}" "${bs}"
  done
fi

if [[ "${RUN_LARGE}" == "1" ]]; then
  for bs in ${BATCH_SIZES}; do
    run_spec_job "offloading_large" "${LARGE_LLM}" "${LARGE_RESERVE_MB}" "${bs}"
  done
fi

echo "Offloading benchmark finished."
