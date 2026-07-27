"""Small independent checkers for the Phase-1 synthetic curriculum."""

from __future__ import annotations

import re
from typing import Any


def _arithmetic(text: str) -> str:
    start_match = re.search(r"(?:Start with|number is)\s+(-?\d+)", text)
    if not start_match:
        raise ValueError("arithmetic start not found")
    value = int(start_match.group(1))
    operation_text = text[start_match.end() :]
    operations = re.findall(r"\b(add|subtract|multiply)\s+(-?\d+)", operation_text)
    if len(operations) < 3:
        raise ValueError("arithmetic operations not found")
    for operation, operand_text in operations[:3]:
        operand = int(operand_text)
        if operation == "add":
            value += operand
        elif operation == "subtract":
            value -= operand
        else:
            value *= operand
    return str(value)


def _binding(text: str) -> str:
    assignments = re.findall(r"Let\s+([a-z])\s+be\s+(-?\d+)", text)
    if not assignments:
        raise ValueError("binding seed not found")
    values = {name: int(value) for name, value in assignments}
    relation = re.search(r"Let\s+([a-z])\s+equal\s+([a-z])\s+plus\s+(-?\d+)", text)
    if not relation:
        raise ValueError("binding relation not found")
    name, source, offset = relation.groups()
    values[name] = values[source] + int(offset)
    scale = re.search(r"Let\s+z\s+equal\s+([a-z])\s+times\s+(\d+)", text)
    if not scale:
        raise ValueError("binding scale not found")
    return str(values[scale.group(1)] * int(scale.group(2)))


def _graph(text: str) -> str:
    edges = re.findall(r"([A-E])\s+to\s+([A-E])", text)
    start_match = re.search(r"Starting at\s+([A-E])", text)
    steps_match = re.search(r"for\s+(\d+)\s+step", text)
    if not edges or not start_match or not steps_match:
        raise ValueError("graph fields not found")
    adjacency = {source: target for source, target in edges}
    node = start_match.group(1)
    for _ in range(int(steps_match.group(1))):
        node = adjacency[node]
    return node


def _constraints(text: str) -> str:
    relations = re.findall(r"([WXYZ])\s+must\s+come\s+before\s+([WXYZ])", text)
    if not relations:
        raise ValueError("constraints not found")
    successors = {right for _, right in relations}
    candidates = {left for left, _ in relations} - successors
    if len(candidates) != 1:
        raise ValueError("constraint answer is not unique")
    return next(iter(candidates))


def solve_problem(row_or_text: dict[str, Any] | str, task_type: str | None = None) -> str:
    text = row_or_text["input_text"] if isinstance(row_or_text, dict) else row_or_text
    task = task_type or (row_or_text.get("task_type") if isinstance(row_or_text, dict) else None)
    if task == "arithmetic" or "running number" in text or "final value" in text:
        return _arithmetic(text)
    if task == "binding" or "Let z equal" in text:
        return _binding(text)
    if task == "graph" or "directed edges" in text:
        return _graph(text)
    if task == "constraints" or "must come before" in text:
        return _constraints(text)
    raise ValueError(f"unknown task type: {task}")


def check_example(row: dict[str, Any]) -> dict[str, Any]:
    try:
        expected = solve_problem(row)
        observed = str(row["answer"])
        return {"valid": expected == observed, "expected": expected, "observed": observed, "reason": "match" if expected == observed else "answer_mismatch"}
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        return {"valid": False, "expected": None, "observed": row.get("answer"), "reason": str(exc)}


def check_trace(row: dict[str, Any]) -> dict[str, Any]:
    result = check_example(row)
    trace = row.get("trace", [])
    result["trace_present"] = bool(trace)
    result["trace_valid_flags"] = sum(bool(item.get("valid")) for item in trace)
    result["trace_count"] = len(trace)
    result["valid"] = bool(result["valid"] and trace)
    return result
