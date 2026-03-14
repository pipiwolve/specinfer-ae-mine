#!/bin/bash
ulimit -s unlimited

cd /root/specinfer-ae

mkdir -p ./FlexFlow/inference/output
mkdir -p ./FlexFlow/inference/prompt

# ====== 4090 单卡配置 ======
ncpus=8
ngpus=1
fsize=18000
zsize=12000

# 使用 HuggingFace 格式名，FlexFlow 自动从 ~/.cache/flexflow/weights/ 加载
llm_model_name="huggyllama/llama-7b"
ssm_model_name="JackFram/llama-68m"
bs=1
max_sequence_length=128

echo ">>> 1. Baseline test (incr_decoding)..."
./FlexFlow/build/inference/incr_decoding/incr_decoding \
  -ll:cpu $ncpus -ll:util $ncpus -ll:gpu $ngpus \
  -ll:fsize $fsize -ll:zsize $zsize \
  -llm-model $llm_model_name \
  -prompt ./FlexFlow/inference/prompt/chatgpt_1.json \
  --max-requests-per-batch $bs \
  --max-sequence-length $max_sequence_length \
  -tensor-parallelism-degree $ngpus \
  --fusion \
  -output-file ./FlexFlow/inference/output/baseline_7b.txt \
 > ./FlexFlow/inference/output/baseline_7b.out 2>&1

echo ">>> 2. SpecInfer test (tree-based speculative decoding)..."
./FlexFlow/build/inference/spec_infer/spec_infer \
  -ll:cpu $ncpus -ll:util $ncpus -ll:gpu $ngpus \
  -ll:fsize $fsize -ll:zsize $zsize \
  -llm-model $llm_model_name \
  -ssm-model $ssm_model_name \
  -prompt ./FlexFlow/inference/prompt/chatgpt_1.json \
  --max-requests-per-batch $bs \
  --max-sequence-length $max_sequence_length \
  -tensor-parallelism-degree $ngpus \
  --fusion \
  -output-file ./FlexFlow/inference/output/specinfer_7b.txt \
 > ./FlexFlow/inference/output/specinfer_7b.out 2>&1

echo "Done! Output:"
echo "  baseline:  ./FlexFlow/inference/output/baseline_7b.out"
echo "  specinfer: ./FlexFlow/inference/output/specinfer_7b.out"