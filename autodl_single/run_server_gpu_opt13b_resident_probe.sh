#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

cd "${REPO_ROOT}"

OUTPUT_ROOT="${REPO_ROOT}/FlexFlow/inference/output/autodl_single/server_gpu_single_a100"
RUN_TAG="${FF_SERVER_RUN_TAG:-opt13b_resident_probe}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_TAG}"
mkdir -p "${OUTPUT_DIR}"

CACHE_DIR="${FF_CACHE_DIR:-/root/autodl-tmp/.cache/flexflow}"
NCPUS="${FF_SERVER_CPUS:-4}"
NUTILS="${FF_SERVER_UTILS:-4}"
NGPUS="${FF_SERVER_GPUS:-1}"
FSIZE_MB="${FF_SERVER_FSIZE_MB:-21800}"
ZSIZE_MB="${FF_SERVER_ZSIZE_MB:-65000}"
MAX_SEQ_LEN="${FF_SERVER_MAX_SEQ_LEN:-64}"
MAX_TOKENS="${FF_SERVER_MAX_TOKENS:-}"
BATCH_SIZES="${FF_SERVER_BATCH_SIZES:-1}"
LLM_MODEL="${FF_LLM_MODEL:-facebook/opt-13b}"
SSM_MODEL="${FF_SSM_MODEL:-facebook/opt-125m}"
RUN_SEQUENCE="${FF_SERVER_RUN_SEQUENCE:-1}"
RUN_TREE="${FF_SERVER_RUN_TREE:-0}"
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

python "${REPO_ROOT}/autodl_single/write_manifest.py" \
  --output-dir "${OUTPUT_DIR}" \
  --experiment-name "server_gpu_opt13b_resident_probe" \
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
  "run_sequence=${RUN_SEQUENCE}" \
  "run_tree=${RUN_TREE}"

echo "Run tag: ${RUN_TAG}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Prompt dir: ${PROMPT_DIR}"
echo "Cache dir: ${CACHE_DIR}"
echo "Models: LLM=${LLM_MODEL} SSM=${SSM_MODEL}"
echo "Runtime: cpu=${NCPUS} util=${NUTILS} gpu=${NGPUS} fsize=${FSIZE_MB} zsize=${ZSIZE_MB} max_seq_len=${MAX_SEQ_LEN} max_tokens=${MAX_TOKENS:-<unset>}"
echo "Batch sizes: ${BATCH_SIZES}"
echo "Probe flags: sequence=${RUN_SEQUENCE} tree=${RUN_TREE}"

max_token_args=()
if [[ -n "${MAX_TOKENS}" ]]; then
  max_token_args=(--max-tokens-per-batch "${MAX_TOKENS}")
fi

run_job() {
  local label="$1"
  shift

  local out_file="${OUTPUT_DIR}/${label}.out"
  local txt_file="${OUTPUT_DIR}/${label}.txt"

  if [[ -f "${out_file}" ]] && grep -q -- "----------inference finished--------------" "${out_file}"; then
    echo "Skipping completed job: ${label}"
    return 0
  fi

  rm -f "${out_file}" "${txt_file}"

  stdbuf -oL -eL "$@" \
    -output-file "${txt_file}" \
    2>&1 | tee "${out_file}"
}

for bs in ${BATCH_SIZES}; do
  prompt_file="${PROMPT_DIR}/chatgpt_${bs}.json"
  if [[ ! -f "${prompt_file}" ]]; then
    echo "Missing prompt file: ${prompt_file}" >&2
    exit 1
  fi

  echo "===== resident probe batch size ${bs}: incr ====="
  run_job "single_gpu-${bs}_incr_dec" \
    ./FlexFlow/build/inference/incr_decoding/incr_decoding \
    -ll:cpu "${NCPUS}" \
    -ll:util "${NUTILS}" \
    -ll:gpu "${NGPUS}" \
    -ll:fsize "${FSIZE_MB}" \
    -ll:zsize "${ZSIZE_MB}" \
    -cache-folder "${CACHE_DIR}" \
    -llm-model "${LLM_MODEL}" \
    -prompt "${prompt_file}" \
    --max-requests-per-batch "${bs}" \
    "${max_token_args[@]}" \
    --max-sequence-length "${MAX_SEQ_LEN}" \
    -tensor-parallelism-degree "${NGPUS}" \
    --fusion

  if [[ "${RUN_SEQUENCE}" == "1" ]]; then
    echo "===== resident probe batch size ${bs}: sequence specinfer ====="
    run_job "single_gpu-${bs}_sequence_specinfer" \
      ./FlexFlow/build/inference/spec_infer/spec_infer \
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
      "${max_token_args[@]}" \
      --max-sequence-length "${MAX_SEQ_LEN}" \
      --expansion-degree -1 \
      -tensor-parallelism-degree "${NGPUS}" \
      --fusion
  fi

  if [[ "${RUN_TREE}" == "1" ]]; then
    echo "===== resident probe batch size ${bs}: tree specinfer ====="
    run_job "single_gpu-${bs}_tree_specinfer" \
      ./FlexFlow/build/inference/spec_infer/spec_infer \
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
      "${max_token_args[@]}" \
      --max-sequence-length "${MAX_SEQ_LEN}" \
      -tensor-parallelism-degree "${NGPUS}" \
      --fusion
  fi
done

python "${REPO_ROOT}/autodl_single/summarize_server_gpu_results.py" \
  --input-dir "${OUTPUT_DIR}"

cat "${OUTPUT_DIR}/summary.csv"
