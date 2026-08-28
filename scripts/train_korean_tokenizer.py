# /// script
# requires-python = ">=3.10"
# dependencies = ["tokenizers>=0.15", "transformers>=5.0"]
# ///
# How to run: uv run --project . python scripts/train_korean_tokenizer.py --help

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cerpt.data.tokenizer import (
    CausalTokenizerSpec,
    build_causal_bpe_tokenizer,
    load_jsonl_texts,
    save_tokenizer,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an exact-size Korean BPE tokenizer for causal CERPT")
    parser.add_argument("--data-dir", type=Path, default=Path("data/korean_conversations_v7"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tokenizers/cerpt-korean-32k"))
    parser.add_argument("--final-vocab-size", type=int, default=32768)
    parser.add_argument("--workspace-slots", type=int, default=16)
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--model-max-length", type=int, default=4096)
    args = parser.parse_args()
    tokenizer = build_causal_bpe_tokenizer(
        load_jsonl_texts(args.data_dir / "train.jsonl"),
        CausalTokenizerSpec(
            final_vocab_size=args.final_vocab_size,
            workspace_tokens=args.workspace_slots * args.cycles,
            model_max_length=args.model_max_length,
        ),
    )
    save_tokenizer(tokenizer, args.output_dir)
    print(f"saved {len(tokenizer)} learned and base special tokens to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
