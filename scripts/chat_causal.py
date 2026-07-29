from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cerpt.models.cerpt_causal import CERPTForCausalLM
from cerpt.verification.arithmetic import calculate_korean_arithmetic
from cerpt.utils.device import select_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a decoder-only CERPT checkpoint")
    parser.add_argument("--model-dir", default="artifacts/cerpt-causal-korean-base")
    parser.add_argument("--question", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
    model = CERPTForCausalLM.from_pretrained(model_dir)
    device = select_device(args.device)
    model.to(device).eval()
    print(f"CERPT causal loaded on {device}. Type 'exit' to quit.")

    questions = [args.question] if args.question is not None else None
    while questions is None or questions:
        question = questions.pop(0) if questions else input("질문> ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        verified = calculate_korean_arithmetic(question)
        if verified is not None:
            print("답:", verified["answer"])
            print("CERPT verifier: deterministic arithmetic", "→".join(verified["trace"]))
            continue
        prompt = f"[TASK_CHAT] 다음 질문에 답하세요. {question}\n"
        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            generated = model.generate(**encoded, max_new_tokens=args.max_new_tokens, eos_token_id=tokenizer.eos_token_id)
        new_tokens = generated[0, encoded["input_ids"].size(1) :]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if answer.lower().startswith("answer"):
            answer = answer[6:].strip(" :")
        print("답:", answer)
        if questions is not None:
            questions = []


if __name__ == "__main__":
    main()
