#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

cd "${REPO_ROOT}"

OUTPUT_DIR="${REPO_ROOT}/FlexFlow/inference/output/autodl_single/smoke"
PROMPT_FILE="${REPO_ROOT}/autodl_single/prompts/smoke_test.json"
STDOUT_LOG="${OUTPUT_DIR}/spec_single.out"
WRAPPER_LOG="${OUTPUT_DIR}/spec_single.wrapper.log"
STRACE_LOG="${OUTPUT_DIR}/spec_single.strace.log"
mkdir -p "${OUTPUT_DIR}"

LL_CPU="${FF_SMOKE_CPU:-4}"
LL_UTIL="${FF_SMOKE_UTIL:-4}"
LL_GPU="${FF_SMOKE_GPU:-1}"
LL_FSIZE_MB="${FF_SMOKE_FSIZE_MB:-34000}"
LL_ZSIZE_MB="${FF_SMOKE_ZSIZE_MB:-30000}"
MAX_REQUESTS="${FF_SMOKE_MAX_REQUESTS:-1}"
MAX_TOKENS="${FF_SMOKE_MAX_TOKENS:-64}"
MAX_SEQ_LEN="${FF_SMOKE_MAX_SEQ_LEN:-64}"
TP_DEGREE="${FF_SMOKE_TP_DEGREE:-1}"
LLM_MODEL="${FF_SMOKE_LLM_MODEL:-huggyllama/llama-7b}"
SSM_MODEL="${FF_SMOKE_SSM_MODEL:-jackfram/llama-68m}"
OUTPUT_FILE="${OUTPUT_DIR}/spec_single.txt"

CMD=(
  ./FlexFlow/build/inference/spec_infer/spec_infer
  -ll:cpu "${LL_CPU}"
  -ll:util "${LL_UTIL}"
  -ll:gpu "${LL_GPU}"
  -ll:fsize "${LL_FSIZE_MB}"
  -ll:zsize "${LL_ZSIZE_MB}"
  -cache-folder "${FF_CACHE_DIR}"
  -llm-model "${LLM_MODEL}"
  -ssm-model "${SSM_MODEL}"
  -prompt "${PROMPT_FILE}"
  --max-requests-per-batch "${MAX_REQUESTS}"
  --max-tokens-per-batch "${MAX_TOKENS}"
  --max-sequence-length "${MAX_SEQ_LEN}"
  -tensor-parallelism-degree "${TP_DEGREE}"
  --fusion
  -output-file "${OUTPUT_FILE}"
)

{
  echo "Running speculative inference smoke test"
  echo "Command:"
  printf ' %q' "${CMD[@]}"
  printf '\n'
  echo "Logs:"
  echo "  stdout/stderr -> ${STDOUT_LOG}"
  echo "  wrapper       -> ${WRAPPER_LOG}"
  if [[ "${FF_ENABLE_STRACE:-0}" == "1" ]]; then
    echo "  strace        -> ${STRACE_LOG}"
  fi
  echo "Parameters:"
  echo "  ll:cpu=${LL_CPU}"
  echo "  ll:util=${LL_UTIL}"
  echo "  ll:gpu=${LL_GPU}"
  echo "  ll:fsize=${LL_FSIZE_MB}"
  echo "  ll:zsize=${LL_ZSIZE_MB}"
  echo "  max_requests=${MAX_REQUESTS}"
  echo "  max_tokens=${MAX_TOKENS}"
  echo "  max_seq_len=${MAX_SEQ_LEN}"
  echo "  tp_degree=${TP_DEGREE}"
  echo "  llm_model=${LLM_MODEL}"
  echo "  ssm_model=${SSM_MODEL}"
  echo "  cache_folder=${FF_CACHE_DIR}"
} | tee "${WRAPPER_LOG}"

set +e
if [[ "${FF_ENABLE_STRACE:-0}" == "1" ]]; then
  stdbuf -oL -eL strace -f -s 256 -o "${STRACE_LOG}" "${CMD[@]}" 2>&1 | tee "${STDOUT_LOG}"
  STATUS=${PIPESTATUS[0]}
else
  stdbuf -oL -eL "${CMD[@]}" 2>&1 | tee "${STDOUT_LOG}"
  STATUS=${PIPESTATUS[0]}
fi
set -e

{
  echo "Exit status: ${STATUS}"
  if [[ "${STATUS}" -eq 0 ]]; then
    echo "SpecInfer smoke test finished successfully."
  else
    echo "SpecInfer smoke test failed."
    echo "Inspect:"
    echo "  tail -n 120 ${STDOUT_LOG}"
    if [[ "${FF_ENABLE_STRACE:-0}" == "1" ]]; then
      echo "  tail -n 120 ${STRACE_LOG}"
    fi
  fi
} | tee -a "${WRAPPER_LOG}"

exit "${STATUS}"
