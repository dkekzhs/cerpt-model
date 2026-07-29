"""Input-evidence ablation utilities for the first causal audit."""

from __future__ import annotations

import re
from typing import Any


def ablate_input(row: dict[str, Any]) -> str:
    """Remove one task-relevant clause while preserving the prompt format."""
    text = row["input_text"]
    task = row.get("task_type")
    if task == "arithmetic":
        match = re.search(r"(?:add|subtract|multiply)\s+-?\d+[,;]?\s*", text)
        if match:
            return text[: match.start()] + text[match.end() :]
    elif task == "graph":
        match = re.search(r"[A-E]\s+to\s+[A-E],?\s*", text)
        if match:
            return text[: match.start()] + text[match.end() :]
    elif task == "constraints":
        match = re.search(r"[WXYZ]\s+must\s+come\s+before\s+[WXYZ],?\s*", text)
        if match:
            return text[: match.start()] + text[match.end() :]
    elif task == "binding":
        match = re.search(r"Let\s+[a-z]\s+equal\s+[a-z]\s+plus\s+-?\d+\.\s*", text)
        if match:
            return text[: match.start()] + text[match.end() :]
    return text + " Remove one piece of evidence before solving."


def normalize_answer(answer: str) -> str:
    tokens = answer.strip().split() if answer.strip() else []
    if tokens and tokens[0] == "-" and len(tokens) > 1:
        return "-" + tokens[1]
    return tokens[0] if tokens else ""


def extract_generated_answer(text: str) -> str:
    match = re.search(r"\banswer\s+(.+?)(?:\s+cycle\b|$)", text)
    return normalize_answer(match.group(1)) if match else ""
