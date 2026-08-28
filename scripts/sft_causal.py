from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cerpt.data.causal import CausalBatchFormatter, ConversationRow
from cerpt.models.cerpt_causal import CERPTForCausalLM
from cerpt.utils.device import autocast_context, select_device


class SFTTrainingRow(ConversationRow):
    task_type: str


@dataclass(frozen=True, slots=True)
class MissingSFTLossError(RuntimeError):
    def __str__(self) -> str:
        return "SFT model did not return a training loss"


@dataclass(frozen=True, slots=True)
class EmptySFTSplitError(ValueError):
    def __str__(self) -> str:
        return "SFT requires non-empty train and validation chat splits"


class JsonlDataset(Dataset[SFTTrainingRow]):
    def __init__(self, path: Path, task_type: str):
        with path.open(encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip() and json.loads(line).get("task_type") == task_type]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> SFTTrainingRow:
        return self.rows[index]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, optimizer, device, train: bool, precision: str, gradient_accumulation_steps: int) -> float:
    model.train(train)
    total_loss = 0.0
    count = 0
    if train:
        optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.set_grad_enabled(train), autocast_context(device, precision):
            output = model(**batch)
            if output.loss is None:
                raise MissingSFTLossError
            if train:
                (output.loss / gradient_accumulation_steps).backward()
                if (step + 1) % gradient_accumulation_steps == 0 or step + 1 == len(loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        size = batch["input_ids"].size(0)
        total_loss += output.loss.item() * size
        count += size
    return total_loss / max(count, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised fine-tune CERPT on response targets")
    parser.add_argument("--resume-from", required=True)
    parser.add_argument("--data-dir", default="data/korean_basic_v5")
    parser.add_argument("--output-dir", default="artifacts/cerpt-causal-korean-sft")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_set = JsonlDataset(data_dir / "train.jsonl", "korean_daily_chat")
    valid_set = JsonlDataset(data_dir / "validation.jsonl", "korean_daily_chat")
    if not train_set or not valid_set:
        raise EmptySFTSplitError

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.resume_from)
    model = CERPTForCausalLM.from_pretrained(args.resume_from)
    max_length = min(args.max_length, model.config.max_position_embeddings)
    formatter = CausalBatchFormatter(tokenizer, tuple(model.config.workspace_token_ids), max_length)
    collator = formatter.collate
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    device = select_device(args.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    best = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, True, args.precision, args.gradient_accumulation_steps)
        valid_loss = run_epoch(model, valid_loader, optimizer, device, False, args.precision, args.gradient_accumulation_steps)
        record = {"epoch": epoch, "train_loss": train_loss, "validation_loss": valid_loss, "device": str(device), "task_type": "korean_daily_chat"}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if valid_loss < best:
            best = valid_loss
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        model.save_pretrained(output_dir / "latest")
        tokenizer.save_pretrained(output_dir / "latest")
        (output_dir / "training_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "training_config.json").write_text(json.dumps({**vars(args), "task_type": "korean_daily_chat"}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved SFT checkpoint to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
