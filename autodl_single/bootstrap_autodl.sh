#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p /autodl-tmp/work /autodl-tmp/.cache/huggingface /autodl-tmp/.cache/flexflow /autodl-tmp/tmp

apt-get update
apt-get install -y build-essential cmake ninja-build pkg-config openmpi-bin libopenmpi-dev git curl wget

if [[ ! -f /root/.cargo/env ]]; then
  curl https://sh.rustup.rs -sSf | sh -s -- -y
fi
# shellcheck disable=SC1091
source /root/.cargo/env

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /root/miniconda3/etc/profile.d/conda.sh
fi

if ! conda env list | awk '{print $1}' | grep -qx flexflow; then
  conda env create -f FlexFlow/conda/flexflow.yml
fi

conda activate flexflow
python -m pip install --upgrade pip
python -m pip install fastapi uvicorn matplotlib

if [[ ! -d "${REPO_ROOT}/ucx-1.15.0/install" ]]; then
  bash install_ucx.sh
fi

bash install_specinfer.sh

if [[ -f "${REPO_ROOT}/FlexFlow/build/deps/legion/runtime/legion/legion_defines.h" ]]; then
  cp "${REPO_ROOT}/FlexFlow/build/deps/legion/runtime/legion/legion_defines.h" \
    "${REPO_ROOT}/FlexFlow/deps/legion/runtime/legion/legion_defines.h"
fi

if [[ -f "${REPO_ROOT}/FlexFlow/build/deps/legion/runtime/legion/legion_config.h" ]]; then
  cp "${REPO_ROOT}/FlexFlow/build/deps/legion/runtime/legion/legion_config.h" \
    "${REPO_ROOT}/FlexFlow/deps/legion/runtime/legion/legion_config.h"
fi

if [[ -f "${REPO_ROOT}/FlexFlow/deps/legion/bindings/python/legion_cffi_build.py" ]]; then
  (
    cd "${REPO_ROOT}/FlexFlow/deps/legion/bindings/python"
    python legion_cffi_build.py || true
  )
fi

echo "Bootstrap finished. Run: source autodl_single/env.sh"
