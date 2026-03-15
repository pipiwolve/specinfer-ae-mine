#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_VARIANT="${1:-68m}"

case "${MODEL_VARIANT}" in
  68m)
    export FF_SSM_MODEL="${FF_SSM_MODEL:-jackfram/llama-68m}"
    export FF_SERVER_RUN_TAG="${FF_SERVER_RUN_TAG:-focus_68m}"
    ;;
  160m)
    export FF_SSM_MODEL="${FF_SSM_MODEL:-jackfram/llama-160m}"
    export FF_SERVER_RUN_TAG="${FF_SERVER_RUN_TAG:-focus_160m}"
    ;;
  *)
    echo "Usage: bash autodl_single/run_server_gpu_focus.sh [68m|160m]" >&2
    exit 1
    ;;
esac

export FF_SERVER_CPUS="${FF_SERVER_CPUS:-8}"
export FF_SERVER_UTILS="${FF_SERVER_UTILS:-8}"
export FF_SERVER_GPUS="${FF_SERVER_GPUS:-1}"
export FF_SERVER_FSIZE_MB="${FF_SERVER_FSIZE_MB:-21890}"
export FF_SERVER_ZSIZE_MB="${FF_SERVER_ZSIZE_MB:-50000}"
export FF_SERVER_MAX_SEQ_LEN="${FF_SERVER_MAX_SEQ_LEN:-128}"
export FF_SERVER_MAX_TOKENS="${FF_SERVER_MAX_TOKENS:-128}"
export FF_SERVER_BATCH_SIZES="${FF_SERVER_BATCH_SIZES:-1 2 4}"

echo "Running focused server benchmark with SSM=${FF_SSM_MODEL}"
echo "Run tag=${FF_SERVER_RUN_TAG} batch_sizes=${FF_SERVER_BATCH_SIZES}"

bash autodl_single/run_server_gpu_single_a100.sh

python autodl_single/summarize_server_gpu_results.py \
  --input-dir "FlexFlow/inference/output/autodl_single/server_gpu_single_a100/${FF_SERVER_RUN_TAG}"

cat "FlexFlow/inference/output/autodl_single/server_gpu_single_a100/${FF_SERVER_RUN_TAG}/summary.csv"
