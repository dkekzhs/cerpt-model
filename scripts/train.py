from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cerpt.data.synthetic import OPERATORS
from cerpt.data.tokenizer import build_tokenizer
from cerpt.models.cerpt import CERPTConfig, CERPTForConditionalGeneration


class JsonlDataset(Dataset):
    def __init__(self, path: Path):
        with path.open(encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def make_collator(tokenizer, max_length: int):
    def collate(rows: list[dict]) -> dict[str, torch.Tensor]:
        inputs = tokenizer([row["input_text"] for row in rows], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        # The custom offline tokenizer has no implicit post-processor, so add
        # an explicit EOS target to give greedy generation a stopping signal.
        targets = tokenizer([row["target_text"] + " [EOS]" for row in rows], padding=True, truncation=True, max_length=max_length, return_tensors="pt")["input_ids"]
        labels = targets.masked_fill(targets.eq(tokenizer.pad_token_id), -100)
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels,
            "cycle_valid_labels": torch.tensor([row["cycle_valid_labels"] for row in rows], dtype=torch.float32),
            "operator_labels": torch.tensor([row["operator_labels"] for row in rows], dtype=torch.long),
        }
    return collate


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, optimizer, device, train: bool) -> float:
    model.train(train)
    total_loss = 0.0
    count = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.set_grad_enabled(train):
            output = model(**batch)
            if output.loss is None:
                raise RuntimeError("model did not return a loss")
            if train:
                optimizer.zero_grad(set_to_none=True)
                output.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total_loss += output.loss.item() * batch["input_ids"].size(0)
        count += batch["input_ids"].size(0)
    return total_loss / max(count, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CERPT Phase-1 PoC")
    parser.add_argument("--data-dir", default="data/synthetic")
    parser.add_argument("--output-dir", default="artifacts/cerpt-small")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-encoder-layers", type=int, default=2)
    parser.add_argument("--num-decoder-layers", type=int, default=2)
    parser.add_argument("--workspace-slots", type=int, default=16)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from", default=None, help="load a previous CERPT checkpoint and continue to the requested total epoch")
    parser.add_argument("--start-epoch", type=int, default=None, help="last completed epoch when resuming from a checkpoint")
    args = parser.parse_args()
    seed_everything(args.seed)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_set = JsonlDataset(data_dir / "train.jsonl")
    valid_set = JsonlDataset(data_dir / "validation.jsonl")
    resume_dir = Path(args.resume_from) if args.resume_from else None
    if resume_dir:
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast.from_pretrained(resume_dir)
    else:
        texts = [text for row in train_set.rows for text in (row["input_text"], row["target_text"])]
        tokenizer = build_tokenizer(texts)
    tokenizer.save_pretrained(output_dir / "tokenizer")
    collator = make_collator(tokenizer, args.max_length)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    if resume_dir:
        model = CERPTForConditionalGeneration.from_pretrained(resume_dir)
        config = model.config
    else:
        config = CERPTConfig(
            vocab_size=len(tokenizer),
            hidden_size=args.hidden_size,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            workspace_slots=args.workspace_slots,
            num_cycles=args.cycles,
            max_position_embeddings=args.max_length,
        )
        model = CERPTForConditionalGeneration(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    best = float("inf")
    history_path = output_dir / "training_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
    if history:
        best = min(float(row["validation_loss"]) for row in history)
    start_epoch = int(history[-1]["epoch"]) + 1 if resume_dir and history else 1
    if args.start_epoch is not None:
        start_epoch = args.start_epoch + 1
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, True)
        valid_loss = run_epoch(model, valid_loader, optimizer, device, False)
        record = {"epoch": epoch, "train_loss": train_loss, "validation_loss": valid_loss, "device": str(device)}
        history.append(record)
        print(json.dumps(record))
        if valid_loss < best:
            best = valid_loss
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        # Keep a recoverable latest checkpoint even when validation loss does
        # not improve during a long run.
        model.save_pretrained(output_dir / "latest")
        tokenizer.save_pretrained(output_dir / "latest")
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "training_config.json").write_text(json.dumps({**vars(args), "operators": OPERATORS}, indent=2), encoding="utf-8")
    print(f"saved checkpoint to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
