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

conda config --remove-key channels >/dev/null 2>&1 || true
conda config --add channels conda-forge >/dev/null
conda config --add channels defaults >/dev/null
conda config --set auto_activate_base false >/dev/null
conda config --set channel_priority flexible >/dev/null
conda config --set default_channels "['https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main','https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r']" >/dev/null
conda config --set custom_channels.conda-forge https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud >/dev/null
conda config --set show_channel_urls true >/dev/null
conda config --set remote_connect_timeout_secs 30 >/dev/null
conda config --set remote_read_timeout_secs 120 >/dev/null
conda config --set remote_max_retries 10 >/dev/null
conda config --set remote_backoff_factor 2 >/dev/null
conda config --set repodata_fns '["current_repodata.json", "repodata.json"]' >/dev/null
conda clean -i -y >/dev/null 2>&1 || true

if ! conda env list | awk '{print $1}' | grep -qx flexflow; then
  if ! conda list -n base mamba >/dev/null 2>&1; then
    retry 3 conda install -n base -y mamba
  fi
  retry 3 mamba env create -f FlexFlow/conda/flexflow.yml
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
retry 3 python -m pip install fastapi uvicorn matplotlib

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
