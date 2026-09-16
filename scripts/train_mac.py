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
    parser.add_argument("--mode", choices=["sft", "3b"], default="3b")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--tokenizer-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument(
        "--corpus-documents",
        type=int,
        default=5_000_000,
        help="FineWeb2 Korean source documents to stream when data is missing; 0 uses the full corpus",
    )
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
        data_dir = Path(args.data_dir or project / "data" / "pretraining_shards")
        tokenizer_dir = Path(args.tokenizer_dir or project / "artifacts" / "tokenizers" / "cerpt-korean-32k")
        if not data_dir.is_absolute():
            data_dir = project / data_dir
        if not tokenizer_dir.is_absolute():
            tokenizer_dir = project / tokenizer_dir
        if not all((data_dir / name).is_file() for name in ("train.jsonl", "validation.jsonl", "metadata.json")):
            prepare_command = [
                python,
                str(project / "scripts" / "prepare_korean_pretraining.py"),
                "--output-dir",
                str(data_dir),
                "--max-documents",
                str(args.corpus_documents),
            ]
            print("Korean pretraining data is missing; streaming FineWeb2 kor_Hang.", flush=True)
            subprocess.run(prepare_command, cwd=project, env=environment, check=True)
        if not (tokenizer_dir / "tokenizer.json").is_file():
            tokenizer_command = [
                python,
                str(project / "scripts" / "train_korean_tokenizer.py"),
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(tokenizer_dir),
            ]
            print("Korean 32k tokenizer is missing; training it from the downloaded corpus.", flush=True)
            subprocess.run(tokenizer_command, cwd=project, env=environment, check=True)
        command = [
            python, str(project / "scripts" / "train_causal.py"),
            "--architecture-config", str(project / "configs" / "cerpt-causal-3b.json"),
            "--tokenizer-dir", str(tokenizer_dir),
            "--data-dir", str(data_dir),
            "--output-dir", args.output_dir or str(project / "artifacts" / "cerpt-causal-3b-mps"),
            "--epochs", str(args.epochs or 1), "--batch-size", "1",
            "--device", "mps", "--precision", "fp32",
            "--gradient-accumulation-steps", "32", "--gradient-checkpointing",
        ]
    print("Using Apple Silicon device:", device)
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=project, env=environment, check=True)


if __name__ == "__main__":
    main()
