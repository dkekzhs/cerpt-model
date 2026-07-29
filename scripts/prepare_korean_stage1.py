"""Prepare small, openly licensed Korean reasoning data for CERPT Stage 1.

The downloaded files remain under ``data/korean_stage1/raw``.  This script
creates CERPT-compatible JSONL files without copying long chain-of-thought
verbatim into the generation target; concise evidence is kept in ``trace``.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

OPERATORS = ["EXTRACT", "BIND", "COMPARE", "SIMULATE", "CHECK", "WRITE_RESULT"]
DEFAULT_ROOT = Path("data/korean_stage1")


def clean(value: Any, limit: int | None = None) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if limit is not None and len(text) > limit:
        return text[: max(0, limit - 1)].rstrip() + "…"
    return text


def trace_row(cycle: int, operator: str, evidence: str, valid: bool = True) -> dict[str, Any]:
    evidence = clean(evidence, 500)
    return {
        "cycle": cycle,
        "operator": operator,
        "state_delta": evidence,
        "evidence": evidence,
        "valid": valid,
        "causal": cycle > 0,
    }


def render_target(answer: str, trace: list[dict[str, Any]]) -> str:
    parts = [f"answer {clean(answer, 160)}"]
    for row in trace:
        parts.append(
            f"cycle {row['cycle']} operator {row['operator']} evidence "
            f"{clean(row['evidence'], 220)} valid {'yes' if row['valid'] else 'no'}"
        )
    return " ".join(parts)


def make_row(
    row_id: str,
    task_type: str,
    question: str,
    answer: str,
    evidence: Iterable[str],
    operators: list[str],
    source: str,
    source_license: str,
    raw_index: int,
) -> dict[str, Any]:
    trace = [trace_row(i, op, item) for i, (op, item) in enumerate(zip(operators, evidence))]
    answer = clean(answer, 160)
    return {
        "id": row_id,
        "task_type": task_type,
        "source": source,
        "source_license": source_license,
        "raw_index": raw_index,
        "input_text": "Solve the task. " + clean(question, 2600),
        "target_text": render_target(answer, trace),
        "answer": answer,
        "operator_labels": [OPERATORS.index(op) for op in operators],
        "cycle_valid_labels": [1] * len(operators),
        "trace": trace,
    }


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_openmath(root: Path) -> list[dict[str, Any]]:
    files = sorted((root / "raw" / "openmath").glob("**/openmath_*.jsonl"))
    if not files:
        raise FileNotFoundError("OpenMath JSONL not found under data/korean_stage1/raw/openmath")
    rows = []
    for index, item in enumerate(read_jsonl(files[0])):
        problem = clean(item.get("problem", ""), 2600)
        answer = clean(item.get("expected_answer", ""), 160)
        solution = clean(item.get("generated_solution", ""), 500)
        trace = [
            problem,
            "풀이 전개: " + solution,
            f"정답 후보 {answer}를 풀이 결론과 대조",
            f"최종 답: {answer}",
        ]
        rows.append(make_row(
            f"openmath-ko-{index:06d}", "math", problem, answer, trace,
            ["EXTRACT", "SIMULATE", "CHECK", "WRITE_RESULT"],
            "neuralfoundry-coder/OpenMathReasoning-mini-ko", "CC-BY-4.0", index,
        ))
    return rows


def load_hrmcr(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((root / "raw" / "hrmcr").glob("HRMCR-*.csv")):
        subset = path.stem.removeprefix("HRMCR-").lower()
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for index, item in enumerate(csv.DictReader(handle)):
                question = clean(item.get("question", ""), 2600)
                answer = clean(item.get("answer", ""), 160)
                solution = clean(item.get("solution", ""), 500)
                rows.append(make_row(
                    f"hrmcr-{subset}-{index:04d}", "logic", question, answer,
                    [question, "단계별 추론: " + solution,
                     f"풀이와 정답 {answer}를 검증", f"최종 답: {answer}"],
                    ["EXTRACT", "SIMULATE", "CHECK", "WRITE_RESULT"],
                    "HAERAE-HUB/HRMCR", "Apache-2.0", index,
                ))
    return rows


def load_strategyqa(root: Path) -> list[dict[str, Any]]:
    path = root / "raw" / "strategyqa" / "ko-strategy-qa_full.json"
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    entries = payload.items() if isinstance(payload, dict) else enumerate(payload)
    rows = []
    for index, (key, item) in enumerate(entries):
        facts = item.get("facts", []) or []
        decomposition = item.get("decomposition", []) or []
        fact_text = "\n".join("- " + clean(f, 500) for f in facts[:8])
        question = clean(item.get("question", ""), 1000)
        context = f"{question}\nContext facts:\n{clean(fact_text, 2200)}"
        answer = clean(item.get("answer", ""), 40)
        decomposition_text = " | ".join(clean(part, 180) for part in decomposition[:4])
        rows.append(make_row(
            f"strategyqa-ko-{clean(key, 60)}", "multihop_logic", context, answer,
            ["질문과 조건 추출: " + question,
             "근거 결합: " + clean(fact_text, 500),
             "하위 질문 검증: " + clean(decomposition_text, 500),
             f"최종 답: {answer}"],
            ["EXTRACT", "BIND", "CHECK", "WRITE_RESULT"],
            "NomaDamas/Ko-StrategyQA", "Apache-2.0", index,
        ))
    return rows


def split_rows(rows: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["task_type"], []).append(row)
    result = {"train": [], "validation": [], "test": []}
    for group in groups.values():
        rng.shuffle(group)
        n = len(group)
        train_end = int(n * 0.8)
        valid_end = train_end + int(n * 0.1)
        result["train"].extend(group[:train_end])
        result["validation"].extend(group[train_end:valid_end])
        result["test"].extend(group[valid_end:])
    for name in result:
        rng.shuffle(result[name])
    return result


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CERPT Korean Stage 1 data")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = args.root
    rows = load_openmath(root) + load_hrmcr(root) + load_strategyqa(root)
    splits = split_rows(rows, args.seed)
    root.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        write_jsonl(root / f"{name}.jsonl", split)
    metadata = {
        "stage": 1,
        "description": "Korean reasoning basics mapped to CERPT typed workspace traces",
        "seed": args.seed,
        "operators": OPERATORS,
        "split_ratio": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "sources": {
            "neuralfoundry-coder/OpenMathReasoning-mini-ko": {"license": "CC-BY-4.0", "task": "math"},
            "HAERAE-HUB/HRMCR": {"license": "Apache-2.0", "task": "logic"},
            "NomaDamas/Ko-StrategyQA": {"license": "Apache-2.0", "task": "multihop_logic"},
        },
        "counts": {name: len(split) for name, split in splits.items()},
        "task_counts": {name: dict(Counter(row["task_type"] for row in split)) for name, split in splits.items()},
        "raw_files": [str(path.relative_to(root)) for path in sorted((root / "raw").glob("**/*")) if path.is_file() and ".cache" not in path.parts],
    }
    (root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"counts": metadata["counts"], "task_counts": metadata["task_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
