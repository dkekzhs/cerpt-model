#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
PYTHON="$ROOT/.venv/bin/python"

if [ ! -x "$PYTHON" ]; then
  echo "Missing $PYTHON"
  echo "Run: uv venv --python 3.12 && uv sync --locked"
  read -r -p "Press Enter to close." _
  exit 1
fi

echo "CERPT Apple Silicon trainer"
echo "1) Resume or start 1B Korean base training"
echo "2) Korean chat SFT from a completed base checkpoint"
read -r -p "Select [1]: " choice
choice="${choice:-1}"

if [ "$choice" = "2" ]; then
  read -r -p "Base checkpoint directory: " resume_from
  read -r -p "SFT data directory [data/korean_basic_v6]: " data_dir
  read -r -p "Output directory [artifacts/cerpt-causal-korean-v6-sft-mps]: " output_dir
  "$PYTHON" scripts/train_mac.py --mode sft \
    --resume-from "$resume_from" \
    --data-dir "${data_dir:-data/korean_basic_v6}" \
    --output-dir "${output_dir:-artifacts/cerpt-causal-korean-v6-sft-mps}"
else
  default_root="artifacts/cerpt-cloud"
  read -r -p "Data directory [$default_root/data/korean_conversations_v7]: " data_dir
  read -r -p "Tokenizer directory [$default_root/tokenizers/cerpt-korean-32k]: " tokenizer_dir
  read -r -p "Output directory [$default_root/models/cerpt-causal-korean-v7-1b-30ep]: " output_dir
  "$PYTHON" scripts/train_mac.py --mode 1b \
    --data-dir "${data_dir:-$default_root/data/korean_conversations_v7}" \
    --tokenizer-dir "${tokenizer_dir:-$default_root/tokenizers/cerpt-korean-32k}" \
    --output-dir "${output_dir:-$default_root/models/cerpt-causal-korean-v7-1b-30ep}"
fi

read -r -p "Training finished. Press Enter to close." _
