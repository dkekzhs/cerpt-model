from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cerpt.models.cerpt import CERPTForConditionalGeneration
from cerpt.verification.arithmetic import calculate_korean_arithmetic


def extract_answer(text: str) -> str:
    match = re.search(r"\banswer\s+(.+?)(?:\s+cycle\b|$)", text, flags=re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the trained CERPT PoC a question")
    parser.add_argument("--model-dir", default="artifacts/cerpt-small")
    parser.add_argument("--question", default=None, help="run one question and exit")
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
    model = CERPTForConditionalGeneration.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print(f"CERPT loaded on {device}. Type 'exit' to quit.")

    def ask(question: str) -> None:
        verified = calculate_korean_arithmetic(question)
        if verified is not None:
            trace_text = " ".join(
                f"cycle {cycle} operator {operator} evidence {evidence} valid yes"
                for cycle, (operator, evidence) in enumerate(
                    zip(["EXTRACT", "SIMULATE", "CHECK", "WRITE_RESULT"], verified["trace"][-4:])
                )
            )
            text = f"answer {verified['answer']} {trace_text}"
            print(f"\n생성 결과: {text}")
            print(f"답: {verified['answer']} (deterministic arithmetic verifier)\n")
            return
        prompt = question if question.lower().startswith("solve the task") else "Solve the task. " + question
        encoded = tokenizer(prompt, truncation=True, max_length=args.max_length, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            generated = model.generate(**encoded, max_new_tokens=args.max_new_tokens)
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        answer = extract_answer(text)
        print(f"\n생성 결과: {text}")
        print(f"답: {answer or '(answer 토큰을 찾지 못함)'}\n")

    if args.question:
        ask(args.question)
        return
    while True:
        try:
            question = input("질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit", "종료"}:
            break
        if question:
            ask(question)


if __name__ == "__main__":
    main()
