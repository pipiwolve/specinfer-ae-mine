#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

cd "${REPO_ROOT}"

OUTPUT_ROOT="${REPO_ROOT}/FlexFlow/inference/output/autodl_single/offloading_baseline_single_a100"
RUN_TAG="${FF_OFFLOAD_BASELINE_RUN_TAG:-default}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_TAG}"
mkdir -p "${OUTPUT_DIR}"

NCPUS="${FF_OFFLOAD_CPUS:-4}"
NUTILS="${FF_OFFLOAD_UTILS:-4}"
NGPUS="${FF_OFFLOAD_GPUS:-1}"
FSIZE_MB="${FF_OFFLOAD_FSIZE_MB:-21000}"
ZSIZE_MB="${FF_OFFLOAD_ZSIZE_MB:-50000}"
MAX_SEQ_LEN="${FF_OFFLOAD_MAX_SEQ_LEN:-128}"
MAX_TOKENS="${FF_OFFLOAD_MAX_TOKENS:-128}"
BATCH_SIZES="${FF_OFFLOAD_BATCH_SIZES:-1 2 4}"
LLM_MODEL="${FF_OFFLOAD_SMALL_LLM:-facebook/opt-13b}"
PROMPT_FILE="${FF_OFFLOAD_PROMPT_FILE:-${REPO_ROOT}/FlexFlow/inference/prompt/chatgpt_offloading.json}"
RESERVE_MB="${FF_OFFLOAD_SMALL_RESERVE_MB:-500}"

python "${REPO_ROOT}/autodl_single/write_manifest.py" \
  --output-dir "${OUTPUT_DIR}" \
  --experiment-name "offloading_baseline_single_a100" \
  --repo-root "${REPO_ROOT}" \
  --extra "run_tag=${RUN_TAG}" "llm_model=${LLM_MODEL}" "batch_sizes=${BATCH_SIZES}" "prompt_file=${PROMPT_FILE}"

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "Missing prompt file: ${PROMPT_FILE}" >&2
  echo "Run ./download_dataset.sh first." >&2
  exit 1
fi

run_job() {
  local bs="$1"
  local output_prefix="${OUTPUT_DIR}/offloading_baseline_${bs}"
  local txt_file="${output_prefix}.txt"
  local out_file="${output_prefix}.out"

  if [[ -f "${out_file}" ]] && grep -q -- "----------inference finished--------------" "${out_file}"; then
    echo "Skipping completed baseline batch=${bs}"
    return 0
  fi

  rm -f "${txt_file}" "${out_file}"

  echo "===== offloading baseline batch=${bs} ====="
  stdbuf -oL -eL ./FlexFlow/build/inference/incr_decoding/incr_decoding \
    -ll:cpu "${NCPUS}" \
    -ll:util "${NUTILS}" \
    -ll:gpu "${NGPUS}" \
    -ll:fsize "${FSIZE_MB}" \
    -ll:zsize "${ZSIZE_MB}" \
    -cache-folder "${FF_CACHE_DIR}" \
    -llm-model "${LLM_MODEL}" \
    -prompt "${PROMPT_FILE}" \
    --max-requests-per-batch "${bs}" \
    --max-tokens-per-batch "${MAX_TOKENS}" \
    --max-sequence-length "${MAX_SEQ_LEN}" \
    -offload \
    -offload-reserve-space-size "${RESERVE_MB}" \
    -output-file "${txt_file}" \
    2>&1 | tee "${out_file}"
}

echo "Run tag: ${RUN_TAG}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Prompt file: ${PROMPT_FILE}"
echo "Runtime: cpu=${NCPUS} util=${NUTILS} gpu=${NGPUS} fsize=${FSIZE_MB} zsize=${ZSIZE_MB} max_seq_len=${MAX_SEQ_LEN} max_tokens=${MAX_TOKENS}"
echo "Batches: ${BATCH_SIZES}"
echo "Model: ${LLM_MODEL}"

for bs in ${BATCH_SIZES}; do
  run_job "${bs}"
done

echo "Offloading baseline benchmark finished."
