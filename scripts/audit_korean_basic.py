from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cerpt.verification.arithmetic import calculate_korean_arithmetic


REQUIRED = {"id", "task_type", "input_text", "target_text", "answer", "operator_labels", "cycle_valid_labels", "trace"}


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def audit(data_dir: Path) -> dict:
    splits = {name: read_rows(data_dir / f"{name}.jsonl") for name in ("train", "validation", "test")}
    all_rows = [row for rows in splits.values() for row in rows]
    errors: list[str] = []
    for split, rows in splits.items():
        for index, row in enumerate(rows):
            missing = REQUIRED - row.keys()
            if missing:
                errors.append(f"{split}[{index}] missing {sorted(missing)}")
            if row.get("target_text") != f"answer {row.get('answer', '')}":
                errors.append(f"{split}[{index}] target/answer mismatch")
            if len(row.get("operator_labels", [])) != len(row.get("trace", [])):
                errors.append(f"{split}[{index}] operator/trace length mismatch")
            if len(row.get("cycle_valid_labels", [])) != len(row.get("trace", [])):
                errors.append(f"{split}[{index}] validity/trace length mismatch")

    id_counts = Counter(row.get("id") for row in all_rows)
    duplicate_ids = {key: count for key, count in id_counts.items() if count > 1}
    input_hashes: dict[str, set[str]] = defaultdict(set)
    for split, rows in splits.items():
        for row in rows:
            digest = hashlib.sha256(row.get("input_text", "").encode("utf-8")).hexdigest()
            input_hashes[digest].add(split)
    leaked_inputs = sum(1 for values in input_hashes.values() if len(values) > 1)
    within_split_duplicates = sum(len(rows) - len({row.get("input_text") for row in rows}) for rows in splits.values())
    task_types = sorted({row.get("task_type") for row in all_rows})
    missing_task_splits = {
        split: [task for task in task_types if not any(row.get("task_type") == task for row in rows)]
        for split, rows in splits.items()
    }

    arithmetic_rows = [row for row in all_rows if row.get("task_type") == "korean_arithmetic"]
    arithmetic_verified = 0
    arithmetic_unparsed: list[str] = []
    arithmetic_mismatches: list[dict] = []
    for row in arithmetic_rows:
        question = row["input_text"]
        result = calculate_korean_arithmetic(question)
        if result is None:
            arithmetic_unparsed.append(question)
        elif result["answer"] == str(row["answer"]):
            arithmetic_verified += 1
        else:
            arithmetic_mismatches.append({"id": row["id"], "expected": row["answer"], "computed": result["answer"]})

    return {
        "counts": {split: len(rows) for split, rows in splits.items()},
        "task_counts": {split: dict(Counter(row.get("task_type") for row in rows)) for split, rows in splits.items()},
        "total_rows": len(all_rows),
        "schema_errors": errors,
        "duplicate_ids": len(duplicate_ids),
        "cross_split_duplicate_inputs": leaked_inputs,
        "within_split_duplicate_rows": within_split_duplicates,
        "unique_inputs_by_split": {split: len({row.get("input_text") for row in rows}) for split, rows in splits.items()},
        "missing_task_splits": missing_task_splits,
        "unique_inputs": len(input_hashes),
        "arithmetic_rows": len(arithmetic_rows),
        "arithmetic_verified": arithmetic_verified,
        "arithmetic_unparsed": len(arithmetic_unparsed),
        "arithmetic_answer_mismatches": arithmetic_mismatches,
        "quality_ok": not errors and not duplicate_ids and not leaked_inputs and not any(missing_task_splits.values()) and not arithmetic_unparsed and not arithmetic_mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit CERPT Korean curriculum JSONL files")
    parser.add_argument("--data-dir", default="data/korean_basic_v2")
    args = parser.parse_args()
    report = audit(Path(args.data_dir))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["quality_ok"] else 1)


if __name__ == "__main__":
    main()
