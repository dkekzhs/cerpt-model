from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cerpt.utils.device import select_device


def main() -> None:
    parser = argparse.ArgumentParser(description="One-click Apple Silicon CERPT training launcher")
    parser.add_argument("--mode", choices=["sft", "3b"], default="sft")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--tokenizer-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    device = select_device("mps")
    project = Path(__file__).resolve().parents[1]
    python = sys.executable
    environment = os.environ.copy()
    environment.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")

    if args.mode == "sft":
        command = [
            python, str(project / "scripts" / "sft_causal.py"),
            "--resume-from", str(project / "artifacts" / "cerpt-causal-korean-v5-10"),
            "--data-dir", args.data_dir or str(project / "data" / "korean_basic_v6"),
            "--output-dir", args.output_dir or str(project / "artifacts" / "cerpt-causal-korean-v6-sft-mps"),
            "--epochs", str(args.epochs or 5), "--batch-size", "64",
            "--max-length", "96", "--device", "mps", "--precision", "fp16",
            "--gradient-accumulation-steps", "4", "--gradient-checkpointing",
        ]
    else:
        if not args.tokenizer_dir:
            raise SystemExit("3b mode requires --tokenizer-dir pointing to a real 32k tokenizer")
        command = [
            python, str(project / "scripts" / "train_causal.py"),
            "--architecture-config", str(project / "configs" / "cerpt-causal-3b.json"),
            "--tokenizer-dir", args.tokenizer_dir,
            "--data-dir", args.data_dir or str(project / "data" / "pretraining_shards"),
            "--output-dir", args.output_dir or str(project / "artifacts" / "cerpt-causal-3b-mps"),
            "--epochs", str(args.epochs or 1), "--batch-size", "1",
            "--device", "mps", "--precision", "fp16",
            "--gradient-accumulation-steps", "32", "--gradient-checkpointing",
        ]
    print("Using Apple Silicon device:", device)
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=project, env=environment, check=True)


if __name__ == "__main__":
    main()
