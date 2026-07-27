from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cerpt.models.cerpt import CERPTForConditionalGeneration
from train import JsonlDataset, make_collator


def extract_answer(text: str) -> str:
    match = re.search(r"\banswer\s+([^\s]+)", text)
    if not match:
        return ""
    token = match.group(1)
    if token == "-":
        following = text[match.end() :].lstrip().split(None, 1)
        return "-" + following[0] if following else "-"
    return token


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CERPT generation and cycle gates")
    parser.add_argument("--model-dir", default="artifacts/cerpt-small")
    parser.add_argument("--data", default="data/synthetic/test.jsonl")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N rows; 0 means all")
    args = parser.parse_args()
    model_dir = Path(args.model_dir)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
    model = CERPTForConditionalGeneration.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    dataset = JsonlDataset(Path(args.data))
    if args.limit > 0:
        dataset.rows = dataset.rows[: args.limit]
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=make_collator(tokenizer, args.max_length))
    correct = 0
    total = 0
    gate_values = []
    examples = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        generated = model.generate(input_ids, attention_mask, max_new_tokens=args.max_new_tokens)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for text, row in zip(decoded, dataset.rows[total : total + len(decoded)]):
            predicted = extract_answer(text)
            correct += int(predicted == row["answer"])
            total += 1
            examples.append({"input": row["input_text"], "prediction": text, "expected": row["answer"], "correct": predicted == row["answer"]})
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=batch["labels"].to(device), cycle_valid_labels=batch["cycle_valid_labels"].to(device), operator_labels=batch["operator_labels"].to(device))
            gate_values.append(out.commit_gates.detach().cpu())
    mean_gate = torch.cat(gate_values).mean().item() if gate_values else 0.0
    result = {"exact_match": correct / max(total, 1), "correct": correct, "total": total, "mean_commit_gate": mean_gate, "examples": examples[:10]}
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
