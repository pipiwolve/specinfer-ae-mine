#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

cd "${REPO_ROOT}"

RUN_TAG="${FF_OFFLOAD_RUN_TAG:-default}"
OUTPUT_DIR="${REPO_ROOT}/FlexFlow/inference/output/autodl_single/offloading_single_a100/${RUN_TAG}"

echo "===== offloading pipeline ====="
echo "repo: ${REPO_ROOT}"
echo "run tag: ${RUN_TAG}"
echo "output dir: ${OUTPUT_DIR}"
echo "cache dir: ${FF_CACHE_DIR}"
echo "time: $(date '+%Y-%m-%d %H:%M:%S')"

mkdir -p "${OUTPUT_DIR}"

echo "===== step 1/3: download models ====="
bash "${SCRIPT_DIR}/download_offload_models.sh"

echo "===== step 2/3: run benchmark ====="
bash "${SCRIPT_DIR}/run_offloading_single_a100_resume.sh"

echo "===== step 3/3: summarize ====="
python "${SCRIPT_DIR}/summarize_offloading_results.py" --input-dir "${OUTPUT_DIR}"

echo "===== done ====="
cat "${OUTPUT_DIR}/summary.csv"
