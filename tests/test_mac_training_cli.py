from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_mac_cli_exposes_resumable_one_billion_mode() -> None:
    # Given: the Apple Silicon training entrypoint.
    command = [sys.executable, str(ROOT / "scripts" / "train_mac.py"), "--help"]

    # When: its machine-consumed CLI contract is requested.
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)

    # Then: the current 1B base mode replaces the obsolete 3B mode.
    assert result.returncode == 0, result.stderr
    assert "1b" in result.stdout
    assert "3b" not in result.stdout


def test_mac_double_click_launcher_dispatches_base_and_sft_inputs() -> None:
    # Given: the Finder double-click launcher.
    launcher = ROOT / "scripts" / "train_mac.command"

    # When: its executable dispatch contract is inspected.
    content = launcher.read_text(encoding="utf-8")

    # Then: it uses the project environment and passes every required mode input.
    assert ".venv/bin/python" in content
    assert "--mode 1b" in content
    assert "--resume-from" in content
    assert "--mode 3b" not in content
