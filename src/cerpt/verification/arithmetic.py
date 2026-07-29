"""Deterministic Korean arithmetic verifier used by the CERPT CHECK cycle."""

from __future__ import annotations

import re
from typing import Any

START_RE = re.compile(r"(-?\d+)\s*(?:에서|부터|에|을\(를\)|을|를)")
OP_RE = re.compile(r"(-?\d+)\s*(?:을\(를\)|을|를|로)\s*(더|빼|곱|나누)")


def calculate_korean_arithmetic(text: str) -> dict[str, Any] | None:
    """Parse and execute a small integer arithmetic question.

    Returning ``None`` is intentional: unrecognized natural-language input
    must continue through the learned CERPT generator instead of being
    silently treated as arithmetic.
    """
    start_match = START_RE.search(text)
    if not start_match:
        return None
    operations = []
    for match in OP_RE.finditer(text, start_match.end()):
        operand = int(match.group(1))
        verb = match.group(2)
        if verb == "더":
            operation = "add"
        elif verb == "빼":
            operation = "subtract"
        elif verb == "곱":
            operation = "multiply"
        else:
            operation = "divide"
        operations.append((operation, operand))
    if not operations:
        return None

    value = int(start_match.group(1))
    trace = [f"initial value {value}"]
    for operation, operand in operations:
        if operation == "add":
            value += operand
        elif operation == "subtract":
            value -= operand
        elif operation == "multiply":
            value *= operand
        else:
            if operand == 0 or value % operand != 0:
                return None
            value //= operand
        trace.append(f"after {operation} {operand} value {value}")
    return {"answer": str(value), "trace": trace, "operations": operations}
