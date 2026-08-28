from __future__ import annotations

import argparse
import json
from pathlib import Path


def estimate(config: dict) -> int:
    hidden = int(config["hidden_size"])
    intermediate = int(config.get("intermediate_size") or hidden * 4)
    layers = int(config["num_hidden_layers"])
    vocab = int(config["vocab_size"])
    attention_heads = int(config["num_attention_heads"])
    key_value_heads = int(config.get("num_key_value_heads") or attention_heads)
    head_dim = int(config.get("head_dim") or hidden // attention_heads)
    operators = int(config.get("num_operators", 8))
    attention = 2 * hidden * head_dim * (attention_heads + key_value_heads)
    feed_forward = 3 * hidden * intermediate
    transformer_layer = attention + feed_forward + 2 * hidden
    output_embeddings = 0 if config.get("tie_word_embeddings", False) else vocab * hidden
    return (
        vocab * hidden
        + output_embeddings
        + layers * transformer_layer
        + hidden
        + hidden * operators + operators
        + hidden + 1
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate CERPT causal parameter count without allocating the model")
    parser.add_argument("--config", default="configs/cerpt-causal-3b.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    parameters = estimate(config)
    print(json.dumps({"config": args.config, "parameters": parameters, "billions": parameters / 1_000_000_000, "fp16_weight_gb": parameters * 2 / 1024**3}, indent=2))


if __name__ == "__main__":
    main()
