#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export SPECINFER_REPO_ROOT="${REPO_ROOT}"
export HF_HOME="${HF_HOME:-/autodl-tmp/.cache/huggingface}"
export FF_CACHE_DIR="${FF_CACHE_DIR:-/autodl-tmp/.cache/flexflow}"
export TMPDIR="${TMPDIR:-/autodl-tmp/tmp}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

export FF_HOME="${REPO_ROOT}/FlexFlow"
export FF_GPU_BACKEND="${FF_GPU_BACKEND:-cuda}"
export UCX_DIR="${UCX_DIR:-${REPO_ROOT}/ucx-1.15.0/install}"

mkdir -p "${HF_HOME}" "${FF_CACHE_DIR}" "${TMPDIR}"

if [[ -d "${FF_HOME}/build" ]]; then
  export LD_LIBRARY_PATH="${FF_HOME}/build:${UCX_DIR}/lib:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="${FF_HOME}/python:${FF_HOME}/build/python:${FF_HOME}/deps/legion/bindings/python:${PYTHONPATH:-}"
else
  export LD_LIBRARY_PATH="${UCX_DIR}/lib:${LD_LIBRARY_PATH:-}"
fi

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /root/miniconda3/etc/profile.d/conda.sh
fi

cat <<EOF
SPECINFER_REPO_ROOT=${SPECINFER_REPO_ROOT}
HF_HOME=${HF_HOME}
FF_CACHE_DIR=${FF_CACHE_DIR}
TMPDIR=${TMPDIR}
FF_HOME=${FF_HOME}
UCX_DIR=${UCX_DIR}
EOF
