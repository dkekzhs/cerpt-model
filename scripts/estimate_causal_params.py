from __future__ import annotations

import argparse
import json
from pathlib import Path


def estimate(config: dict) -> int:
    hidden = int(config["hidden_size"])
    intermediate = int(config.get("intermediate_size") or hidden * 4)
    layers = int(config["num_hidden_layers"])
    vocab = int(config["vocab_size"])
    max_positions = int(config["max_position_embeddings"])
    slots = int(config.get("workspace_slots", 8))
    operators = int(config.get("num_operators", 8))
    transformer_layer = 4 * hidden * hidden + 2 * hidden * intermediate + 9 * hidden + intermediate
    return (
        2 * vocab * hidden
        + max_positions * hidden
        + layers * transformer_layer
        + transformer_layer
        + 2 * hidden
        + slots * hidden
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
