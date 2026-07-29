from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cerpt.models.cerpt import CERPTForConditionalGeneration
from cerpt.verification.audit import ablate_input, extract_generated_answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure answer changes after input evidence ablation")
    parser.add_argument("--model-dir", default="artifacts/cerpt-small")
    parser.add_argument("--data", default="data/synthetic/test.jsonl")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.data).open(encoding="utf-8") if line.strip()][: args.limit]
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_dir)
    model = CERPTForConditionalGeneration.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    def predict(texts: list[str]) -> list[str]:
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        output = model.generate(**encoded, max_new_tokens=args.max_new_tokens)
        return [extract_generated_answer(text) for text in tokenizer.batch_decode(output, skip_special_tokens=True)]

    full_predictions = predict([row["input_text"] for row in rows])
    ablated_predictions = predict([ablate_input(row) for row in rows])
    full_correct = [prediction == row["answer"] for prediction, row in zip(full_predictions, rows)]
    ablated_correct = [prediction == row["answer"] for prediction, row in zip(ablated_predictions, rows)]
    result = {
        "count": len(rows),
        "full_accuracy": sum(full_correct) / max(len(rows), 1),
        "ablated_accuracy": sum(ablated_correct) / max(len(rows), 1),
        "causal_drop_rate": sum(a and not b for a, b in zip(full_correct, ablated_correct)) / max(sum(full_correct), 1),
        "prediction_sensitivity": sum(a != b for a, b in zip(full_predictions, ablated_predictions)) / max(len(rows), 1),
        "examples": [
            {"id": row["id"], "full": full, "ablated": ablated, "expected": row["answer"]}
            for row, full, ablated in zip(rows, full_predictions, ablated_predictions)
        ][:10],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
