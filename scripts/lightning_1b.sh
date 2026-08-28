#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -d /teamspace/studios/this_studio ]]; then
  DEFAULT_CLOUD_ROOT=/teamspace/studios/this_studio/cerpt-cloud
else
  DEFAULT_CLOUD_ROOT="$ROOT/artifacts/cerpt-cloud"
fi
CLOUD_ROOT="${CERPT_CLOUD_ROOT:-$DEFAULT_CLOUD_ROOT}"
RAW_DIR="$CLOUD_ROOT/raw"
DATA_DIR="$CLOUD_ROOT/data/korean_conversations_v7"
TOKENIZER_DIR="$CLOUD_ROOT/tokenizers/cerpt-korean-32k"
MODEL_DIR="$CLOUD_ROOT/models/cerpt-causal-korean-v7-1b-30ep"
SONGYS_SHA=4cf20d13fc46f5037fd1c531cd566e2dd9f72974

cd "$ROOT"

case "${1:-}" in
  prepare)
    OFFICE_ZIP="${2:?usage: lightning_1b.sh prepare /path/to/office.zip}"
    if [[ "${CERPT_ACK_OFFICE_LICENSE:-0}" != "1" ]]; then
      echo "Set CERPT_ACK_OFFICE_LICENSE=1 only after confirming the office archive's training and model-release terms." >&2
      exit 64
    fi
    uv sync --locked
    mkdir -p "$RAW_DIR" "$DATA_DIR" "$TOKENIZER_DIR"
    curl -fL "https://raw.githubusercontent.com/songys/Chatbot_data/$SONGYS_SHA/ChatbotData.csv" -o "$RAW_DIR/ChatbotData.csv"
    curl -fL "https://raw.githubusercontent.com/songys/Chatbot_data/$SONGYS_SHA/LICENSE" -o "$RAW_DIR/ChatbotData.LICENSE"
    uv run --locked --project . python scripts/prepare_korean_conversations.py \
      --current-data-dir data/korean_basic_v6 \
      --songys-csv "$RAW_DIR/ChatbotData.csv" \
      --office-zip "$OFFICE_ZIP" \
      --output-dir "$DATA_DIR" \
      --acknowledge-office-license
    uv run --locked --project . python scripts/train_korean_tokenizer.py \
      --data-dir "$DATA_DIR" \
      --output-dir "$TOKENIZER_DIR"
    ;;
  train)
    test -f "$DATA_DIR/train.jsonl"
    test -f "$TOKENIZER_DIR/tokenizer.json"
    uv run --locked --project . python scripts/train_causal_cloud.py \
      --data-dir "$DATA_DIR" \
      --tokenizer-dir "$TOKENIZER_DIR" \
      --output-dir "$MODEL_DIR" \
      --architecture-config configs/cerpt-causal-1b.json \
      --profile lightning-t4 \
      --epochs 30 \
      --batch-size 1 \
      --gradient-accumulation-steps 32 \
      --save-steps 100
    ;;
  upload)
    test -n "${HF_TOKEN:-}"
    test -f "$MODEL_DIR/TRAINING_COMPLETE"
    HF_REPO_ID="${HF_REPO_ID:-qweqwqw113/cerpt-causal-korean-v7-1b-30ep}"
    uv run --locked --project . hf upload "$HF_REPO_ID" "$MODEL_DIR/final" . --repo-type model
    uv run --locked --project . hf upload \
      "$HF_REPO_ID" \
      docs/model-cards/MODEL_CARD_CAUSAL_KOREAN_1B_30EP.md \
      README.md \
      --repo-type model
    uv run --locked --project . hf upload \
      "$HF_REPO_ID" "$MODEL_DIR/TRAINING_COMPLETE" TRAINING_COMPLETE --repo-type model
    uv run --locked --project . hf upload \
      "$HF_REPO_ID" "$MODEL_DIR/trainer_state.json" trainer_state.json --repo-type model
    ;;
  status)
    if [[ ! -d "$MODEL_DIR" ]]; then
      echo "not started"
    elif [[ -f "$MODEL_DIR/TRAINING_COMPLETE" ]]; then
      cat "$MODEL_DIR/TRAINING_COMPLETE"
    else
      LAST_CHECKPOINT="$(find "$MODEL_DIR" -maxdepth 1 -type d -name 'checkpoint-*' -print | sort -V | tail -1)"
      if [[ -n "$LAST_CHECKPOINT" ]]; then
        echo "$LAST_CHECKPOINT"
      else
        echo "not started"
      fi
    fi
    ;;
  *)
    echo "usage: lightning_1b.sh {prepare OFFICE_ZIP|train|upload|status}" >&2
    exit 64
    ;;
esac
