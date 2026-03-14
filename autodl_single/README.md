# SpecInfer AutoDL Single-GPU Kit

This directory turns the upstream artifact into a single-node, single-GPU workflow for an AutoDL `A100-PCIE-40GB x1` instance.

## What this kit does

- Keeps all large files on `/root/autodl-tmp`
- Downloads only `huggyllama/llama-7b` and `JackFram/llama-68m`
- Provides single-GPU smoke tests for incremental decoding and speculative inference
- Adds an OpenAI-compatible FastAPI server on top of FlexFlow Serve
- Adds batch runners for the 50-question medical workload
- Adds analysis scripts for latency and acceptance-style metrics
- Supports a second English medical QA workload for cleaner SpecInfer mechanism experiments

## Expected remote layout

Clone the repository to:

```bash
/root/autodl-tmp/work/specinfer-ae
```

The scripts assume the repository is on a data disk and that you `source autodl_single/env.sh` before running FlexFlow commands.

## Mainland mirror defaults

The kit defaults to these mirror-friendly settings:

- Miniconda installer: Tsinghua mirror
- Conda channels: Tsinghua mirror for `defaults` and `conda-forge`
- Pip index: Tsinghua PyPI mirror
- Rust toolchain: `rsproxy.cn`
- HuggingFace: `https://hf-mirror.com`
- UCX tarball: `ghfast.top` first, then GitHub origin as fallback
- Conda SSL verification: disabled by default for proxy-heavy AutoDL environments
- HuggingFace, Transformers, pip, torch, and conda package caches: all redirected to `/root/autodl-tmp/.cache`

Override them if your environment prefers a different mirror:

```bash
MINICONDA_INSTALLER_URL=...
PIP_INDEX_URL=...
PIP_TRUSTED_HOST=...
RUSTUP_DIST_SERVER=...
RUSTUP_UPDATE_ROOT=...
UCX_TARBALL_URL=...
HF_ENDPOINT=...
```

## Clean setup sequence

```bash
cd /root/autodl-tmp/work/specinfer-ae
bash autodl_single/bootstrap_autodl.sh
source autodl_single/env.sh
bash autodl_single/download_7b_68m.sh
bash autodl_single/run_incr_single.sh
bash autodl_single/run_spec_single.sh
```

What `bootstrap_autodl.sh` now does in practice:

- installs or repairs Miniconda
- configures mainland mirrors
- creates the `flexflow` conda env with a low-memory split strategy
- installs Python dependencies with pip, including the Python `jq` module
- builds UCX
- patches the `tokenizers-c` Rust source for newer Rust toolchains
- builds and installs SpecInfer

## Start the OpenAI-compatible API

```bash
cd /root/autodl-tmp/work/specinfer-ae
source autodl_single/env.sh
CONFIG_FILE="$PWD/autodl_single/configs/api_specinfer_single_a100.json" \
uvicorn autodl_single.openai_api:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/healthz
```

Chat completion test:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "huggyllama/llama-7b",
    "messages": [
      {"role": "system", "content": "You are a concise medical QA assistant."},
      {"role": "user", "content": "成年人低烧需要立刻去急诊吗？"}
    ],
    "max_tokens": 96,
    "stream": false
  }'
```

## Batch experiments

Run baseline incremental decoding on the English mechanism dataset:

```bash
cd /root/autodl-tmp/work/specinfer-ae
source autodl_single/env.sh
python autodl_single/run_medical_batch.py \
  --mode incr \
  --config autodl_single/configs/incr_single_a100.json \
  --dataset autodl_single/datasets/english_medical_qa_50.jsonl \
  --output-dir FlexFlow/inference/output/autodl_single/incr_batch_r1 \
  --repeats 1 \
  --max-length 64
```

Run speculative inference on the English mechanism dataset:

```bash
cd /root/autodl-tmp/work/specinfer-ae
source autodl_single/env.sh
python autodl_single/run_medical_batch.py \
  --mode spec \
  --config autodl_single/configs/specinfer_single_a100.json \
  --dataset autodl_single/datasets/english_medical_qa_50.jsonl \
  --output-dir FlexFlow/inference/output/autodl_single/spec_batch_r1 \
  --repeats 1 \
  --max-length 64

If the full 50-question dataset is not stable, split it first:

```bash
python autodl_single/split_dataset.py \
  --dataset autodl_single/datasets/english_medical_qa_50.jsonl \
  --parts 2 \
  --output-dir autodl_single/datasets/splits \
  --prefix english_medical_qa_25
```

Only merge when you actually ran multiple split outputs. If a single run already produced
`incr_batch_r1_en/results.csv` or `spec_batch_r1_en/results.csv`, skip merging.

Example merge for split runs:

```bash
python -m autodl_single.merge_batch_results \
  --inputs \
    FlexFlow/inference/output/autodl_single/incr_batch_r1_en_01 \
    FlexFlow/inference/output/autodl_single/incr_batch_r1_en_02 \
  --output-dir FlexFlow/inference/output/autodl_single/incr_batch_r1_en
```

Summarize stability and failure categories:

```bash
python autodl_single/summarize_batch_results.py \
  --inputs \
    FlexFlow/inference/output/autodl_single/incr_batch_r1/results.csv \
    FlexFlow/inference/output/autodl_single/spec_batch_r1/results.csv \
  --output-dir FlexFlow/inference/output/autodl_single/stability_summary
```

Select a stable subset for repeat-3 latency experiments:

```bash
python autodl_single/select_stable_subset.py \
  --baseline FlexFlow/inference/output/autodl_single/incr_batch_r1/results.csv \
  --spec FlexFlow/inference/output/autodl_single/spec_batch_r1/results.csv \
  --dataset autodl_single/datasets/english_medical_qa_50.jsonl \
  --output-dir FlexFlow/inference/output/autodl_single/stable_subset \
  --limit 15
```
```

Analyze latency:

```bash
python autodl_single/analyze_latency.py \
  --baseline FlexFlow/inference/output/autodl_single/incr_batch_r1/results.csv \
  --spec FlexFlow/inference/output/autodl_single/spec_batch_r1/results.csv \
  --output-dir FlexFlow/inference/output/autodl_single/latency_analysis
```

Analyze acceptance-style metrics:

```bash
python autodl_single/analyze_acceptance.py \
  --spec-results FlexFlow/inference/output/autodl_single/spec_batch_r1/results.csv \
  --baseline-results FlexFlow/inference/output/autodl_single/incr_batch_r1/results.csv \
  --log-glob '/root/autodl-tmp/spec_batch_r1*.log' \
  --output-dir FlexFlow/inference/output/autodl_single/acceptance_analysis
```

If your FlexFlow `.out` logs expose accepted or rejected token counts, add them via `--log-glob` and the analyzer will prefer real metrics over the fallback output-agreement proxy.

## Outputs

- Smoke test logs: `FlexFlow/inference/output/autodl_single/smoke`
- Batch outputs: `FlexFlow/inference/output/autodl_single/{incr_batch_r1,spec_batch_r1}`
- API logs: controlled by your `uvicorn` invocation
- Latency summary: `latency_ab.csv` and `latency_speedup.png`
- Acceptance summary: `acceptance_metrics.csv`, `acceptance_by_prompt.json`, and `acceptance_distribution.png`
- Stability summary: `failure_summary.csv`

## Recommended split by objective

- Use `autodl_single/datasets/english_medical_qa_50.jsonl` for Day 5 and Day 6 mechanism experiments.
- Keep `autodl_single/datasets/medical_qa_50.jsonl` for Day 4 Chinese stability and failure analysis.
- If the English full run is still unstable, split it and merge the outputs before analysis.

## Notes

- Do not run the upstream `download_models.sh` or `basic_test.sh` on a single A100 40GB box. They pull in large models and multi-node paths that are outside this kit's target.
- `JackFram/llama-68m` may be missing tokenizer files. `download_models_minimal.py` automatically falls back to the `llama-7b` tokenizer cache for the 68M SSM when needed.
