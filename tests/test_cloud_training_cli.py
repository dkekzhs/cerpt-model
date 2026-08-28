from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from cerpt.data.causal import add_workspace_tokens
from cerpt.data.tokenizer import build_tokenizer
from scripts.train_causal_cloud import (
    ArchitecturePreset,
    CloudProfile,
    _create_model,
    _profile_settings,
)

ROOT = Path(__file__).resolve().parents[1]


def _bash_executable() -> Path | None:
    if sys.platform != "win32":
        resolved = shutil.which("bash")
        return Path(resolved) if resolved is not None else None
    configured = os.environ.get("OMO_CODEX_GIT_BASH_PATH")
    candidates = [
        Path(configured) if configured is not None else None,
        Path("C:/Program Files/Git/bin/bash.exe"),
    ]
    return next((candidate for candidate in candidates if candidate is not None and candidate.is_file()), None)


def test_cloud_training_cli_exposes_resumable_lightning_profile() -> None:
    # Given: the repository's cloud training entrypoint.
    command = [sys.executable, str(ROOT / "scripts" / "train_causal_cloud.py"), "--help"]

    # When: a user asks for its command-line contract.
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)

    # Then: the Lightning profile and resume checkpoint controls are available.
    assert result.returncode == 0
    assert "--profile" in result.stdout
    assert "--save-steps" in result.stdout
    assert "lightning-t4" in result.stdout


def test_colab_t4_profile_accepts_the_runtime_reported_memory_floor() -> None:
    # Given: Colab's T4 runtime reports slightly less than the marketed 16 GB in GiB.
    profile = CloudProfile.COLAB_T4

    # When: the Colab profile is resolved.
    settings = _profile_settings(profile)

    # Then: its floor admits a normal T4 while retaining the memory preflight.
    assert settings.minimum_gpu_memory_gib == 14.5


def test_colab_model_is_initialized_directly_in_fp16() -> None:
    # Given: a tiny architecture using the same FP16 construction path as Colab 3B.
    tokenizer = build_tokenizer(["질문", "대답"])
    preset = ArchitecturePreset(
        vocab_size=len(tokenizer) + 4,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=32,
        workspace_slots=2,
        num_cycles=2,
        num_operators=8,
    )
    original_dtype = torch.get_default_dtype()

    # When: the model is created with Colab profile settings.
    model = _create_model(preset, tokenizer, _profile_settings(CloudProfile.COLAB_T4))

    # Then: no FP32 parameter copy remains and the process default is restored.
    assert {parameter.dtype for parameter in model.parameters()} == {torch.float16}
    assert torch.get_default_dtype() == original_dtype


def test_lightning_launcher_dispatches_prepare_train_and_upload() -> None:
    # Given: the one-command Lightning launcher.
    launcher = ROOT / "scripts" / "lightning_3b.sh"

    # When: its machine-consumed command dispatch is inspected.
    content = launcher.read_text(encoding="utf-8")

    # Then: every required pipeline phase has an explicit dispatch target.
    assert "prepare)" in content
    assert "train)" in content
    assert "upload)" in content
    assert "TRAINING_COMPLETE" in content
    assert "uv run --locked --project . hf upload" in content
    assert "scripts/upload_model.py" not in content


def test_lightning_launcher_reports_not_started_before_first_checkpoint(tmp_path: Path) -> None:
    # Given: a fresh cloud root without a model directory or checkpoint.
    bash = _bash_executable()
    if bash is None:
        pytest.skip("Git Bash is required to exercise the Lightning launcher")
    env = {**os.environ, "CERPT_CLOUD_ROOT": tmp_path.as_posix()}

    # When: status is requested before the first training run.
    result = subprocess.run(
        [str(bash), (ROOT / "scripts" / "lightning_3b.sh").as_posix(), "status"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    # Then: the command succeeds and explicitly reports the initial state.
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "not started"


def test_cloud_training_resumes_from_the_latest_step_checkpoint(tmp_path: Path) -> None:
    # Given: a tiny causal dataset, tokenizer, and architecture for a two-run CPU smoke test.
    data_dir = tmp_path / "data"
    tokenizer_dir = tmp_path / "tokenizer"
    output_dir = tmp_path / "model"
    data_dir.mkdir()
    rows = [
        {"input_text": "질문 하나", "target_text": "대답 하나"},
        {"input_text": "질문 둘", "target_text": "대답 둘"},
    ]
    for name in ("train", "validation"):
        (data_dir / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    tokenizer = build_tokenizer([text for row in rows for text in row.values()])
    expected_vocab_size = len(tokenizer) + len(add_workspace_tokens(tokenizer, 2, 2))
    tokenizer = build_tokenizer([text for row in rows for text in row.values()])
    tokenizer.save_pretrained(tokenizer_dir)
    architecture = tmp_path / "tiny.json"
    architecture.write_text(
        json.dumps(
            {
                "vocab_size": expected_vocab_size,
                "hidden_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "intermediate_size": 64,
                "max_position_embeddings": 32,
                "workspace_slots": 2,
                "num_cycles": 2,
                "num_operators": 8,
            }
        ),
        encoding="utf-8",
    )
    base_command = [
        sys.executable,
        str(ROOT / "scripts" / "train_causal_cloud.py"),
        "--data-dir",
        str(data_dir),
        "--tokenizer-dir",
        str(tokenizer_dir),
        "--output-dir",
        str(output_dir),
        "--architecture-config",
        str(architecture),
        "--profile",
        "cpu-smoke",
        "--batch-size",
        "1",
        "--gradient-accumulation-steps",
        "1",
        "--save-steps",
        "1",
    ]

    # When: one step is saved and the same command continues to step two.
    first = subprocess.run([*base_command, "--max-steps", "1"], cwd=ROOT, capture_output=True, text=True, check=False)
    second = subprocess.run([*base_command, "--max-steps", "2"], cwd=ROOT, capture_output=True, text=True, check=False)

    # Then: the second run reports an exact resume from checkpoint-1.
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    completion = json.loads((output_dir / "TRAINING_COMPLETE").read_text(encoding="utf-8"))
    assert completion["global_step"] == 2
    assert Path(completion["resumed_from"]).name == "checkpoint-1"
