from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cerpt.models.cerpt_causal import CERPTForCausalLM
from cerpt.utils.device import autocast_context, select_device


class JsonlDataset(Dataset):
    def __init__(self, path: Path, task_type: str):
        with path.open(encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip() and json.loads(line).get("task_type") == task_type]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def make_collator(tokenizer, max_length: int):
    def collate(rows: list[dict]) -> dict[str, torch.Tensor]:
        prompts = [f"{row['input_text']}\n" for row in rows]
        texts = [f"{prompt}{row['target_text']} [EOS]" for prompt, row in zip(prompts, rows)]
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt", add_special_tokens=False)
        labels = encoded["input_ids"].clone().masked_fill(encoded["attention_mask"].eq(0), -100)
        prompt_lengths = [len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) for prompt in prompts]
        for index, prompt_length in enumerate(prompt_lengths):
            labels[index, : min(prompt_length, max_length)] = -100
        if not (labels != -100).any(dim=1).all():
            raise ValueError("SFT batch contains a response truncated away; increase --max-length")
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
        }

    return collate


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
                raise RuntimeError("SFT model did not return a loss")
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
        raise ValueError("SFT requires non-empty train and validation chat splits")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.resume_from)
    collator = make_collator(tokenizer, args.max_length)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    model = CERPTForCausalLM.from_pretrained(args.resume_from)
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
