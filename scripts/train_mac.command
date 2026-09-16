#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="$PWD/.venv/bin/python"

if [ ! -x "$PYTHON" ]; then
  echo "Missing $PYTHON"
  echo "Run: uv venv --python 3.12 && uv sync --locked"
  read -r -p "Press Enter to close." _
  exit 1
fi

echo "CERPT Apple Silicon trainer"
echo "1) Korean-only 3B pretraining (downloads missing data and tokenizer)"
echo "2) Korean chat SFT from a completed causal base checkpoint"
read -r -p "Select [1]: " choice
choice="${choice:-1}"

if [ "$choice" = "2" ]; then
  read -r -p "Base checkpoint directory: " resume_from
  "$PYTHON" scripts/train_mac.py --mode sft --resume-from "$resume_from"
else
  echo "FineWeb2 Korean contains about 60.9M documents / 48.6B tokens."
  echo "The 5M default is a starter corpus; enter 0 for the full corpus (~98.5GB source)."
  read -r -p "Source documents [5000000]: " corpus_documents
  corpus_documents="${corpus_documents:-5000000}"
  "$PYTHON" scripts/train_mac.py --mode 3b --corpus-documents "$corpus_documents"
fi

read -r -p "Training finished. Press Enter to close." _
