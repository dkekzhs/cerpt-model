from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import TypedDict

ROOT = Path(__file__).resolve().parents[1]


class NotebookCell(TypedDict):
    id: str
    cell_type: str
    source: list[str]


class NotebookDocument(TypedDict):
    nbformat: int
    cells: list[NotebookCell]


def load_notebook() -> NotebookDocument:
    path = ROOT / "notebooks" / "CERPT_3B_Colab_Training.ipynb"
    return NotebookDocument(**json.loads(path.read_text(encoding="utf-8")))


def test_colab_notebook_covers_data_training_resume_and_storage() -> None:
    # Given: the checked-in Colab notebook.
    notebook = load_notebook()

    # When: its machine-executed cells are parsed.
    code_cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    code = "\n".join(code_cells)
    cell_ids = [cell["id"] for cell in notebook["cells"]]
    for index, source in enumerate(code_cells):
        ast.parse(source, filename=f"notebook-cell-{index}")

    # Then: one run includes persistent storage, all data stages, resumable training, and gated upload.
    assert notebook["nbformat"] == 4
    assert len(cell_ids) == len(set(cell_ids))
    assert "drive.mount(" in code
    assert "prepare_korean_conversations.py" in code
    assert "train_korean_tokenizer.py" in code
    assert "train_causal_cloud.py" in code
    assert '"colab-t4"' in code
    assert "TRAINING_COMPLETE" in code
    assert '"hf", "upload"' in code


def test_colab_notebook_checks_out_main_and_fails_fast_when_checkout_is_stale() -> None:
    # Given: a notebook that may reuse an existing Colab checkout.
    notebook = load_notebook()

    # When: the checkout cell is inspected as executable Python.
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    # Then: it selects the published branch and validates every required entry point.
    assert 'REPO_BRANCH = "main"' in code
    assert '"fetch", "origin", REPO_BRANCH' in code
    assert '"switch", REPO_BRANCH' in code
    assert 'PROJECT_ROOT / "scripts/prepare_korean_conversations.py"' in code
    assert 'PROJECT_ROOT / "scripts/train_korean_tokenizer.py"' in code
    assert 'PROJECT_ROOT / "scripts/train_causal_cloud.py"' in code


def test_colab_notebook_uses_the_active_interpreter_for_every_python_entry_point() -> None:
    # Given: dependencies installed into the active Colab interpreter.
    notebook = load_notebook()

    # When: executable notebook cells are inspected.
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    # Then: uv bootstrap and project scripts use the same interpreter environment.
    assert '[sys.executable, "-m", "pip", "install", "-q", "uv"]' in code
    assert 'shutil.which("uv")' in code
    assert code.count('sys.executable, "scripts/') == 3
    assert '"python", "scripts/' not in code
