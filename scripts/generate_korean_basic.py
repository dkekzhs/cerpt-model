"""Generate a compact Korean arithmetic and daily-chat curriculum.

This stage intentionally uses short ``answer ...`` targets. Long evidence
traces, verifier targets, and richer output serialization are deferred until
the model can reliably answer basic Korean inputs.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

OPERATORS = ["EXTRACT", "BIND", "COMPARE", "SIMULATE", "CHECK", "WRITE_RESULT"]


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def row(row_id: str, task_type: str, question: str, answer: str, source: str, operators: list[str]) -> dict[str, Any]:
    answer = clean(str(answer))
    task_tag = "ARITHMETIC" if task_type == "korean_arithmetic" else "CHAT"
    evidence = [clean(question), "typed task transition", "answer format and content check", answer]
    trace = [
        {"cycle": cycle, "operator": operator, "evidence": evidence[cycle], "valid": True}
        for cycle, operator in enumerate(operators)
    ]
    return {
        "id": row_id,
        "task_type": task_type,
        "source": source,
        "input_text": f"[TASK_{task_tag}] 다음 질문에 답하세요. " + clean(question),
        "target_text": "answer " + answer,
        "answer": answer,
        "operator_labels": [OPERATORS.index(operator) for operator in operators],
        "cycle_valid_labels": [1, 1, 1, 1],
        "trace": trace,
    }


def arithmetic_rows(count: int, rng: random.Random) -> list[dict[str, Any]]:
    rows = []
    seen_questions: set[str] = set()
    index = 0
    while len(rows) < count:
        start = rng.randint(-20, 50)
        style = index % 5
        if style == 0:
            amount = rng.randint(1, 30)
            answer = start + amount
            question = f"{start}에서 {amount}를 더하면 얼마인가요?"
        elif style == 1:
            amount = rng.randint(1, 30)
            answer = start - amount
            question = f"{start}에서 {amount}를 빼면 얼마인가요?"
        elif style == 2:
            amount = rng.randint(2, 9)
            answer = start * amount
            question = f"{start}에 {amount}를 곱하면 얼마인가요?"
        elif style == 3:
            divisor = rng.randint(2, 9)
            start = rng.randint(-10, 20) * divisor
            answer = start // divisor
            question = f"{start}을(를) {divisor}로 나누면 얼마인가요?"
        else:
            add, multiply, subtract = rng.randint(1, 15), rng.randint(2, 5), rng.randint(1, 15)
            answer = (start + add) * multiply - subtract
            question = f"{start}에서 시작해 {add}를 더하고, {multiply}를 곱한 다음, {subtract}를 빼면 얼마인가요?"
        index += 1
        if question in seen_questions:
            continue
        seen_questions.add(question)
        rows.append(row(f"korean-arithmetic-{len(rows):06d}", "korean_arithmetic", question, str(answer), "synthetic-korean-basic-v4", ["EXTRACT", "SIMULATE", "CHECK", "WRITE_RESULT"]))
    return rows


CHAT_TEMPLATES = [
    ("안녕", "안녕하세요! 무엇을 도와드릴까요?"),
    ("안녕하세요", "안녕하세요! 반갑습니다."),
    ("고마워", "천만에요! 도움이 되어 기쁩니다."),
    ("고마워요", "별말씀을요. 언제든지 물어보세요."),
    ("미안해", "괜찮아요. 다시 도와드릴게요."),
    ("오늘 기분 어때?", "저는 괜찮아요. 당신의 이야기도 듣고 싶어요."),
    ("뭐 하고 있어?", "질문에 답하고 필요한 일을 도와드리고 있어요."),
    ("도와줘", "물론이죠. 무엇이 필요한가요?"),
    ("응원해줘", "잘하고 있어요. 천천히 해도 괜찮습니다!"),
    ("잘 자", "편안한 밤 보내세요. 좋은 꿈 꾸세요!"),
    ("이름이 뭐야?", "저는 CERPT예요."),
    ("오늘 날씨 알려줘", "날씨를 확인하려면 지역을 알려주세요."),
    ("심심해", "같이 이야기해요. 요즘 관심 있는 주제가 있나요?"),
    ("좋은 아침이야", "좋은 아침이에요! 오늘도 힘내세요."),
    ("괜찮아?", "네, 괜찮아요. 걱정해줘서 고마워요."),
]


def chat_rows(count: int, rng: random.Random) -> list[dict[str, Any]]:
    prefixes = [
        "짧고 친절하게", "자연스럽게", "공손하게", "친구처럼", "따뜻하게",
        "초보자도 이해하게", "한 문장으로", "부드러운 말투로", "한국어로",
        "긍정적으로", "차분하게", "간단한 표현으로", "상냥하게", "편하게",
        "도움이 되도록", "핵심만", "예의 있게", "재미있게", "솔직하게", "명확하게",
    ]
    contexts = [
        "지금", "오늘", "잠깐", "대화 중에", "처음 만난 사람에게", "바쁜 상황에서",
        "편안한 분위기에서", "메시지로", "일상 대화에서", "도움을 받는 사람에게",
    ]
    rows = []
    for index in range(count):
        question, answer = CHAT_TEMPLATES[index % len(CHAT_TEMPLATES)]
        variant_index = index // len(CHAT_TEMPLATES)
        if variant_index:
            prefix = prefixes[variant_index % len(prefixes)]
            context = contexts[variant_index // len(prefixes)]
            question = f"{context}, {prefix} 답해줘. {question}"
        rows.append(row(f"korean-chat-{index:06d}", "korean_daily_chat", question, answer, "synthetic-korean-basic-v4", ["EXTRACT", "BIND", "CHECK", "WRITE_RESULT"]))
    return rows


def split_rows(rows: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    result = {"train": [], "validation": [], "test": []}
    # Split by unique prompt group, never by individual repeated row. This
    # prevents an exact prompt from appearing in both train and evaluation.
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in rows:
        groups.setdefault((item["task_type"], item["input_text"]), []).append(item)
    by_task: dict[str, list[tuple[str, str]]] = {}
    for task, prompt in groups:
        by_task.setdefault(task, []).append((task, prompt))
    for task, keys in by_task.items():
        if task == "korean_daily_chat":
            base_questions = {question for question, _ in CHAT_TEMPLATES}
            anchor_keys = [key for key in keys if key[1].split(". ", 1)[-1] in base_questions]
            non_anchor_keys = [key for key in keys if key not in anchor_keys]
            # Keep the canonical question for every chat intent in training;
            # validation/test still contain unseen paraphrase prompts.
            keys = anchor_keys + non_anchor_keys
            train_end = len(anchor_keys) + int(len(non_anchor_keys) * 0.8)
            valid_end = min(len(keys) - 1, train_end + max(1, int(len(non_anchor_keys) * 0.1)))
            random.Random(seed + len(task)).shuffle(non_anchor_keys)
            keys = anchor_keys + non_anchor_keys
        else:
            random.Random(seed + len(task)).shuffle(keys)
            train_end = max(1, int(len(keys) * 0.8))
            valid_end = min(len(keys) - 1, train_end + max(1, int(len(keys) * 0.1)))
        assignments = [("train", keys[:train_end]), ("validation", keys[train_end:valid_end]), ("test", keys[valid_end:])]
        for split, split_keys in assignments:
            for key in split_keys:
                result[split].extend(groups[key])
    for values in result.values():
        random.Random(seed + len(values)).shuffle(values)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/korean_basic")
    parser.add_argument("--arithmetic", type=int, default=6000)
    parser.add_argument("--chat", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    rows = arithmetic_rows(args.arithmetic, rng) + chat_rows(args.chat, rng)
    splits = split_rows(rows, args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, values in splits.items():
        with (output / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for item in values:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    metadata = {
        "stage": "korean_basic",
        "seed": args.seed,
        "target_format": "answer_only_short",
        "deferred": ["long_evidence_trace", "rich_eos_serialization", "complex_output_parser"],
        "operators": OPERATORS,
        "counts": {name: len(values) for name, values in splits.items()},
        "task_counts": {name: dict(Counter(item["task_type"] for item in values)) for name, values in splits.items()},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
