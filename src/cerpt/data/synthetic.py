"""Deterministic synthetic tasks with explicit state/evidence traces.

The generator intentionally keeps the answer checkable by a small deterministic
program.  It is a first-stage curriculum, not a replacement for natural data.
"""

from __future__ import annotations

import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

OPERATORS = ["EXTRACT", "BIND", "COMPARE", "SIMULATE", "CHECK", "WRITE_RESULT"]


def _arithmetic(rng: random.Random, index: int) -> dict[str, Any]:
    start = rng.randint(-9, 12)
    ops = [("add", rng.randint(1, 9)), ("multiply", rng.randint(2, 4)), ("subtract", rng.randint(1, 9))]
    rng.shuffle(ops)
    value = start
    states = [value]
    for name, operand in ops:
        if name == "add":
            value += operand
        elif name == "subtract":
            value -= operand
        else:
            value *= operand
        states.append(value)
    templates = [
        f"Start with {start}. Then {ops[0][0]} {ops[0][1]}, {ops[1][0]} {ops[1][1]}, and {ops[2][0]} {ops[2][1]}. What is the final value?",
        f"A running number is {start}. Apply these operations in order: {ops[0][0]} {ops[0][1]}; {ops[1][0]} {ops[1][1]}; {ops[2][0]} {ops[2][1]}. Give the result.",
    ]
    evidence = [
        f"initial value {start}",
        f"after {ops[0][0]} {ops[0][1]} value {states[1]}",
        f"after {ops[1][0]} {ops[1][1]} value {states[2]}",
        f"after {ops[2][0]} {ops[2][1]} value {states[3]}",
    ]
    return _example(index, "arithmetic", rng.choice(templates), str(value), evidence, ["EXTRACT", "SIMULATE", "CHECK", "WRITE_RESULT"])


def _binding(rng: random.Random, index: int) -> dict[str, Any]:
    x, offset, scale = rng.randint(1, 9), rng.randint(1, 8), rng.randint(2, 4)
    y, z = x + offset, (x + offset) * scale
    first = rng.choice(["x", "m"])
    second = "y" if first == "x" else "n"
    question = f"Let {first} be {x}. Let {second} equal {first} plus {offset}. Let z equal {second} times {scale}. What is z?"
    evidence = [f"{first}={x}", f"{second}={first}+{offset}={y}", f"z={second}*{scale}={z}"]
    return _example(index, "binding", question, str(z), evidence, ["EXTRACT", "BIND", "SIMULATE", "WRITE_RESULT"])


def _graph(rng: random.Random, index: int) -> dict[str, Any]:
    nodes = list("ABCDE")
    rng.shuffle(nodes)
    edges = [(nodes[i], nodes[i + 1]) for i in range(4)]
    start_index = rng.randint(0, 1)
    steps = rng.randint(1, 3)
    start = nodes[start_index]
    end_index = min(start_index + steps, 4)
    answer = nodes[end_index]
    edge_text = ", ".join(f"{a} to {b}" for a, b in edges)
    question = f"A graph has directed edges {edge_text}. Starting at {start}, follow outgoing edges for {steps} step(s). Which node do you reach?"
    path = nodes[start_index : end_index + 1]
    evidence = [f"start node {start}"] + [f"step {i}: {path[i - 1]} to {path[i]}" for i in range(1, len(path))]
    return _example(index, "graph", question, answer, evidence, ["EXTRACT", "SIMULATE", "CHECK", "WRITE_RESULT"])


def _constraints(rng: random.Random, index: int) -> dict[str, Any]:
    items = list("WXYZ")
    rng.shuffle(items)
    relations = [(items[0], items[1]), (items[1], items[2]), (items[2], items[3])]
    relation_text = ", ".join(f"{a} must come before {b}" for a, b in relations)
    question = f"Four items are ordered by these rules: {relation_text}. Which item must come first?"
    evidence = [f"constraint {a} before {b}" for a, b in relations] + [f"there is no predecessor for {items[0]}"]
    return _example(index, "constraints", question, items[0], evidence, ["EXTRACT", "COMPARE", "CHECK", "WRITE_RESULT"])


def _example(index: int, task: str, question: str, answer: str, evidence: list[str], operators: list[str]) -> dict[str, Any]:
    trace = []
    for cycle, (operator, item) in enumerate(zip(operators, evidence)):
        trace.append({
            "cycle": cycle,
            "operator": operator,
            "state_delta": item,
            "evidence": item,
            "valid": True,
            "causal": cycle > 0,
        })
    target = _render_target(answer, trace)
    return {
        "id": f"{task}-{index:06d}",
        "task_type": task,
        "input_text": f"Solve the task. {question}",
        "target_text": target,
        "answer": answer,
        "operator_labels": [OPERATORS.index(x) for x in operators],
        "cycle_valid_labels": [1] * len(operators),
        "trace": trace,
    }


def _render_target(answer: str, trace: list[dict[str, Any]]) -> str:
    trace_text = " ".join(
        f"cycle {row['cycle']} operator {row['operator']} evidence {row['evidence']} valid {'yes' if row.get('valid', True) else 'no'}"
        for row in trace
    )
    # Put the answer first so a small Phase-1 model can be evaluated on the
    # task result before it has mastered long evidence serialization.  The
    # complete trace remains available in the JSON record and is still part
    # of the generation target.
    return f"answer {answer} {trace_text}"


def make_adversarial_rows(rows: list[dict[str, Any]], seed: int = 123, limit: int = 256) -> list[dict[str, Any]]:
    """Create labeled fake-evidence records without changing the task input."""
    rng = random.Random(seed)
    corruptions = [
        ("fabricated", "unsupported claim 999"),
        ("contradicted", "contradicted claim false"),
        ("irrelevant", "irrelevant evidence blue triangle"),
        ("copied_answer", "copied answer token"),
    ]
    result = []
    for index, source in enumerate(rows[:limit]):
        row = deepcopy(source)
        corruption, text = corruptions[index % len(corruptions)]
        target_index = rng.randrange(len(row["trace"]))
        row["trace"][target_index]["evidence"] = text
        row["trace"][target_index]["state_delta"] = text
        row["trace"][target_index]["valid"] = False
        row["cycle_valid_labels"][target_index] = 0
        row["target_text"] = _render_target(row["answer"], row["trace"])
        row["id"] = f"adversarial-{index:06d}"
        row["source_id"] = source["id"]
        row["corruption"] = corruption
        row["evidence_label"] = 0
        result.append(row)
    return result


def generate_dataset(num_examples: int, seed: int = 42, start_index: int = 0) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    generators = [_arithmetic, _binding, _graph, _constraints]
    examples = [generators[i % len(generators)](rng, start_index + i) for i in range(num_examples)]
    rng.shuffle(examples)
    return examples


def write_dataset(output_dir: str | Path, train_size: int = 4000, validation_size: int = 500, test_size: int = 500, seed: int = 42) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, size, offset in [("train", train_size, 0), ("validation", validation_size, train_size), ("test", test_size, train_size + validation_size)]:
        rows = generate_dataset(size, seed + offset, offset)
        with (output / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    train_rows = generate_dataset(train_size, seed, 0)
    adversarial = make_adversarial_rows(train_rows, seed + 999)
    with (output / "adversarial.jsonl").open("w", encoding="utf-8") as handle:
        for row in adversarial:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata = {"seed": seed, "train": train_size, "validation": validation_size, "test": test_size, "adversarial": len(adversarial), "operators": OPERATORS}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/synthetic")
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--validation-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    write_dataset(args.output_dir, args.train_size, args.validation_size, args.test_size, args.seed)
