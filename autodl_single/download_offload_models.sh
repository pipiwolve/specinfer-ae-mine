#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

cd "${REPO_ROOT}"

MODELS=(
  "facebook/opt-125m"
  "facebook/opt-13b"
)

if [[ "${FF_DOWNLOAD_OPT30B:-0}" == "1" ]]; then
  MODELS+=("facebook/opt-30b")
fi

for model in "${MODELS[@]}"; do
  echo "===== downloading ${model} ====="
  python ./FlexFlow/inference/utils/download_hf_model.py --cache-folder "${FF_CACHE_DIR}" --half-precision-only "${model}"
done

echo "Offloading model download finished."
