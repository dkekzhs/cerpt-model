#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "CERPT Apple Silicon trainer"
echo "1) Korean chat SFT (recommended first)"
echo "2) 3B pretraining (requires a 32k tokenizer and pretraining_shards)"
read -r -p "Select [1]: " choice
choice="${choice:-1}"

if [ "$choice" = "2" ]; then
  read -r -p "Tokenizer directory: " tokenizer_dir
  python3 scripts/train_mac.py --mode 3b --tokenizer-dir "$tokenizer_dir"
else
  python3 scripts/train_mac.py --mode sft
fi

read -r -p "Training finished. Press Enter to close." _
