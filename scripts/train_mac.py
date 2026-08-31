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
    parser.add_argument("--mode", choices=["sft", "1b"], default="1b")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--tokenizer-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()
    device = select_device("mps")
    project = Path(__file__).resolve().parents[1]
    python = sys.executable
    environment = os.environ.copy()
    environment.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")

    if args.mode == "sft":
        if not args.resume_from:
            raise SystemExit("sft mode requires --resume-from with a checkpoint trained by the new causal architecture")
        command = [
            python, str(project / "scripts" / "sft_causal.py"),
            "--resume-from", args.resume_from,
            "--data-dir", args.data_dir or str(project / "data" / "korean_basic_v6"),
            "--output-dir", args.output_dir or str(project / "artifacts" / "cerpt-causal-korean-v6-sft-mps"),
            "--epochs", str(args.epochs or 5), "--batch-size", "64",
            "--max-length", "96", "--device", "mps", "--precision", "fp32",
            "--gradient-accumulation-steps", "4", "--gradient-checkpointing",
        ]
    else:
        cloud_root = project / "artifacts" / "cerpt-cloud"
        command = [
            python, str(project / "scripts" / "train_causal_cloud.py"),
            "--architecture-config", str(project / "configs" / "cerpt-causal-1b.json"),
            "--profile", "mac-mps",
            "--tokenizer-dir", args.tokenizer_dir or str(cloud_root / "tokenizers" / "cerpt-korean-32k"),
            "--data-dir", args.data_dir or str(cloud_root / "data" / "korean_conversations_v7"),
            "--output-dir", args.output_dir or str(cloud_root / "models" / "cerpt-causal-korean-v7-1b-30ep"),
            "--epochs", str(args.epochs or 30), "--batch-size", "1",
            "--gradient-accumulation-steps", "32", "--save-steps", "100",
        ]
    print("Using Apple Silicon device:", device)
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=project, env=environment, check=True)


if __name__ == "__main__":
    main()
