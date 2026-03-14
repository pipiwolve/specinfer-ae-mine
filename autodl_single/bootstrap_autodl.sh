#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MINICONDA_INSTALLER_URL="${MINICONDA_INSTALLER_URL:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"
RUSTUP_DIST_SERVER="${RUSTUP_DIST_SERVER:-https://rsproxy.cn}"
RUSTUP_UPDATE_ROOT="${RUSTUP_UPDATE_ROOT:-https://rsproxy.cn/rustup}"
CONDA_SSL_VERIFY="${CONDA_SSL_VERIFY:-false}"

retry() {
  local attempts="$1"
  shift
  local n=1
  until "$@"; do
    if [[ "${n}" -ge "${attempts}" ]]; then
      return 1
    fi
    echo "Command failed. Retry ${n}/${attempts} after 10s: $*"
    sleep 10
    n=$((n + 1))
  done
}

mkdir -p /autodl-tmp/work /autodl-tmp/.cache/huggingface /autodl-tmp/.cache/flexflow /autodl-tmp/tmp

apt-get update
apt-get install -y build-essential cmake ninja-build pkg-config openmpi-bin libopenmpi-dev git curl wget

if [[ ! -x /root/miniconda3/bin/conda ]]; then
  rm -rf /root/miniconda3 /root/miniconda.sh
  retry 3 wget -O /root/miniconda.sh "${MINICONDA_INSTALLER_URL}"
  bash /root/miniconda.sh -b -p /root/miniconda3
fi

export PATH="/root/miniconda3/bin:${PATH}"

if [[ ! -f /root/.cargo/env ]]; then
  export RUSTUP_DIST_SERVER
  export RUSTUP_UPDATE_ROOT
  retry 3 bash -lc "curl --retry 5 --retry-delay 5 --connect-timeout 30 -sSf https://sh.rustup.rs | sh -s -- -y"
fi
# shellcheck disable=SC1091
source /root/.cargo/env

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /root/miniconda3/etc/profile.d/conda.sh
else
  source /root/miniconda3/bin/activate
fi

cat > /root/.condarc <<EOF
show_channel_urls: true
channel_priority: flexible
channels:
  - conda-forge
  - defaults
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
ssl_verify: ${CONDA_SSL_VERIFY}
remote_connect_timeout_secs: 30
remote_read_timeout_secs: 120
remote_max_retries: 10
remote_backoff_factor: 2
repodata_fns:
  - current_repodata.json
  - repodata.json
solver: classic
auto_activate_base: false
EOF

conda clean -i -y >/dev/null 2>&1 || true

if ! conda env list | awk '{print $1}' | grep -qx flexflow; then
  retry 3 conda create -n flexflow -y python=3.10 pip cffi pillow pybind11 jq pytest
fi

conda activate flexflow
mkdir -p /root/.pip
cat > /root/.pip/pip.conf <<EOF
[global]
index-url = ${PIP_INDEX_URL}
trusted-host = ${PIP_TRUSTED_HOST}
timeout = 120
EOF

retry 3 python -m pip install --upgrade pip
retry 3 python -m pip install \
  "qualname>=0.1.0" \
  "keras_preprocessing>=1.1.2" \
  "numpy>=1.16.0" \
  regex \
  onnx \
  "transformers>=4.31.0" \
  sentencepiece \
  einops \
  requests \
  jq \
  fastapi \
  uvicorn \
  matplotlib
retry 3 python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  torch \
  torchvision \
  torchaudio

if [[ ! -d "${REPO_ROOT}/ucx-1.15.0/install" ]]; then
  bash ./install_ucx.sh
fi

find "${REPO_ROOT}/FlexFlow" -path '*tokenizers-c*' -name 'lib.rs' -exec \
  sed -i 's/\*out_len = (\*handle)\.decode_str\.len();/\*out_len = (\&(\*handle)\.decode_str)\.len();/g' {} +

bash ./install_specinfer.sh

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
