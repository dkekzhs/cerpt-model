from __future__ import annotations

from contextlib import nullcontext

import torch


def select_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available; use a recent PyTorch on Apple Silicon")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_context(device: torch.device, precision: str = "auto"):
    precision = precision.lower()
    if precision == "auto":
        precision = "fp16" if device.type in {"cuda", "mps"} else "fp32"
    if precision == "fp32":
        return nullcontext()
    if precision == "fp16":
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    raise ValueError("precision must be auto, fp32, fp16, or bf16")
