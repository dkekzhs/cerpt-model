from __future__ import annotations

from pathlib import Path

_REQUIRED_STATE_FILES = ("trainer_state.json", "optimizer.pt", "scheduler.pt")


def _checkpoint_step(path: Path) -> int | None:
    prefix = "checkpoint-"
    if not path.is_dir() or not path.name.startswith(prefix):
        return None
    suffix = path.name.removeprefix(prefix)
    return int(suffix) if suffix.isdigit() else None


def _has_model_weights(path: Path) -> bool:
    return any(path.glob("model*.safetensors")) or any(path.glob("pytorch_model*.bin"))


def latest_resumable_checkpoint(output_dir: Path) -> str | None:
    candidates = sorted(
        ((step, path) for path in output_dir.glob("checkpoint-*") if (step := _checkpoint_step(path)) is not None),
        reverse=True,
    )
    for _, path in candidates:
        if _has_model_weights(path) and all((path / name).is_file() for name in _REQUIRED_STATE_FILES):
            return str(path)
    return None
