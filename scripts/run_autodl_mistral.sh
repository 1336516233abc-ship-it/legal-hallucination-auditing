#!/usr/bin/env bash
# Run from the frozen source copy in a vLLM 0.10.2 Python environment.
set -euo pipefail

phase=${1:-}
model_key=${2:-mistral}
if [[ ( "$phase" != pilot && "$phase" != full ) || ( "$model_key" != mistral && "$model_key" != llama3 ) ]]; then
  echo "Usage: bash scripts/run_autodl_mistral.sh {pilot|full} [mistral|llama3]" >&2
  exit 2
fi

cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf}"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
export TOKENIZERS_PARALLELISM=false
output="$(pwd)/experiments/results/${model_key}_autodl_$phase"
mkdir -p "$output"
port="${VLLM_PORT:-8000}"
workers=8

python - <<'PY'
from importlib.metadata import version

import torch

if version("vllm") != "0.10.2":
    raise SystemExit("This runner requires vLLM 0.10.2")
if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
    raise SystemExit("Expected one CUDA card with native BF16 support")
device = torch.cuda.get_device_properties(0)
print(f"GPU: {device.name}, {device.total_memory / (1024 ** 3):.1f} GiB; CUDA: {torch.version.cuda}")
if "6000D" not in device.name or not 80 <= device.total_memory / (1024 ** 3) <= 86:
    raise SystemExit("Expected the AutoDL RTX 6000D 84 GB")
PY
nvidia-smi

snapshot=$(python - "$model_key" <<'PY'
import sys

from huggingface_hub import snapshot_download
import yaml

from experiments.run_baseline_pilot import REVISIONS

key = sys.argv[1]
with open("configs/model_config.yaml") as config:
    model = yaml.safe_load(config)["model"][key]["name"]
print(snapshot_download(model, revision=REVISIONS[key],
                        allow_patterns=["model-*.safetensors", "*.json", "tokenizer.model", "chat_template.jinja"]))
PY
)
export HF_HUB_OFFLINE=1

vllm serve "$snapshot" --served-model-name "$model_key-audit" --host 127.0.0.1 --port "$port" \
  --dtype bfloat16 --max-model-len 4608 --max-num-seqs "$workers" --gpu-memory-utilization 0.8 \
  --generation-config vllm --enable-prefix-caching >> "$output/vllm.log" 2>&1 &
server=$!
trap 'kill "$server" 2>/dev/null || true; wait "$server" 2>/dev/null || true' EXIT

deadline=$((SECONDS + 900))
until curl --silent --fail --max-time 2 "http://127.0.0.1:$port/health" > /dev/null; do
  if ! kill -0 "$server" 2>/dev/null || (( SECONDS >= deadline )); then
    echo "vLLM failed to become ready. See $output/vllm.log" >&2
    exit 1
  fi
  sleep 3
done

pilot=()
if [[ "$phase" == pilot ]]; then
  pilot+=(--pilot)
fi
python -m experiments.run_full_mistral --model "$model_key" --vllm-url "http://127.0.0.1:$port/v1" --workers "$workers" \
  --output "$output" --summary-output "$output/summary.json" "${pilot[@]}"
