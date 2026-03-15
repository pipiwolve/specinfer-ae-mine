#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

STAMP="${EXPORT_STAMP:-$(date +%Y%m%d_%H%M%S)}"
EXPORT_ROOT="${REPO_ROOT}/experiment_exports/final_review_${STAMP}"

OFFLOAD_SPEC_TAG="${OFFLOAD_SPEC_TAG:-opt13b_r1}"
OFFLOAD_BASELINE_TAG="${OFFLOAD_BASELINE_TAG:-opt13b_baseline_r1}"
OFFLOAD_COMPARE_CSV="${OFFLOAD_COMPARE_CSV:-${REPO_ROOT}/FlexFlow/inference/output/autodl_single/offloading_compare/opt13b_r1_vs_baseline.csv}"
SERVER_DEFAULT_DIR="${SERVER_DEFAULT_DIR:-${REPO_ROOT}/FlexFlow/inference/output/autodl_single/server_gpu_single_a100/default}"
SERVER_160M_DIR="${SERVER_160M_DIR:-${REPO_ROOT}/FlexFlow/inference/output/autodl_single/server_gpu_single_a100/focus_160m}"

mkdir -p "${EXPORT_ROOT}"

copy_path() {
  local src="$1"
  local dst="$2"
  if [[ -e "${src}" ]]; then
    mkdir -p "$(dirname "${dst}")"
    cp -R "${src}" "${dst}"
    echo "copied: ${src}"
  else
    echo "skip missing: ${src}"
  fi
}

copy_path "${REPO_ROOT}/FlexFlow/inference/output/autodl_single/stability_summary_en" \
  "${EXPORT_ROOT}/english_len64/stability_summary_en"
copy_path "${REPO_ROOT}/FlexFlow/inference/output/autodl_single/acceptance_analysis_en" \
  "${EXPORT_ROOT}/english_len64/acceptance_analysis_en"
copy_path "${REPO_ROOT}/FlexFlow/inference/output/autodl_single/latency_analysis_en" \
  "${EXPORT_ROOT}/english_len64/latency_analysis_en"
copy_path "${REPO_ROOT}/FlexFlow/inference/output/autodl_single/incr_batch_r1_en/results.csv" \
  "${EXPORT_ROOT}/english_len64/incr_results.csv"
copy_path "${REPO_ROOT}/FlexFlow/inference/output/autodl_single/spec_batch_r1_en/results.csv" \
  "${EXPORT_ROOT}/english_len64/spec_results.csv"

copy_path "${REPO_ROOT}/FlexFlow/inference/output/autodl_single/server_gpu_single_a100" \
  "${EXPORT_ROOT}/server_gpu_single_a100"

copy_path "${REPO_ROOT}/FlexFlow/inference/output/autodl_single/offloading_single_a100/${OFFLOAD_SPEC_TAG}" \
  "${EXPORT_ROOT}/offloading/spec_${OFFLOAD_SPEC_TAG}"
copy_path "${REPO_ROOT}/FlexFlow/inference/output/autodl_single/offloading_baseline_single_a100/${OFFLOAD_BASELINE_TAG}" \
  "${EXPORT_ROOT}/offloading/baseline_${OFFLOAD_BASELINE_TAG}"
copy_path "${OFFLOAD_COMPARE_CSV}" \
  "${EXPORT_ROOT}/offloading/compare.csv"

copy_path "${REPO_ROOT}/run_offloading_opt13b_r1.log" \
  "${EXPORT_ROOT}/offloading/run_offloading_opt13b_r1.log"
copy_path "${REPO_ROOT}/run_offloading_baseline_r1.log" \
  "${EXPORT_ROOT}/offloading/run_offloading_baseline_r1.log"
copy_path "${REPO_ROOT}/run_server.log" \
  "${EXPORT_ROOT}/server_gpu_single_a100/run_server.log"
copy_path "${REPO_ROOT}/run_server_focus_160m.log" \
  "${EXPORT_ROOT}/server_gpu_single_a100/run_server_focus_160m.log"

copy_path "${REPO_ROOT}/FlexFlow/inference/prompt/chatgpt_offloading.json" \
  "${EXPORT_ROOT}/prompts/chatgpt_offloading.json"
copy_path "${REPO_ROOT}/FlexFlow/inference/prompt/chatgpt_1.json" \
  "${EXPORT_ROOT}/prompts/chatgpt_1.json"
copy_path "${REPO_ROOT}/FlexFlow/inference/prompt/chatgpt_2.json" \
  "${EXPORT_ROOT}/prompts/chatgpt_2.json"
copy_path "${REPO_ROOT}/FlexFlow/inference/prompt/chatgpt_4.json" \
  "${EXPORT_ROOT}/prompts/chatgpt_4.json"

tar -czf "${EXPORT_ROOT}.tar.gz" -C "${REPO_ROOT}/experiment_exports" "final_review_${STAMP}"

echo "Export dir: ${EXPORT_ROOT}"
echo "Archive: ${EXPORT_ROOT}.tar.gz"
