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


def notebook_code(notebook: NotebookDocument) -> str:
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_colab_notebook_covers_data_training_resume_and_storage() -> None:
    # Given: the checked-in Colab notebook.
    notebook = load_notebook()

    # When: its machine-executed cells are parsed.
    code_cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    code = notebook_code(notebook)
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
    code = notebook_code(notebook)

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
    code = notebook_code(notebook)

    # Then: uv bootstrap and project scripts use the same interpreter environment.
    assert '[sys.executable, "-m", "pip", "install", "-q", "uv"]' in code
    assert 'shutil.which("uv")' in code
    assert code.count('sys.executable, "scripts/') == 2
    assert 'sys.executable, "-u", "scripts/train_causal_cloud.py"' in code
    assert '"python", "scripts/' not in code


def test_colab_dependency_install_targets_kernel_python_and_probes_training_arguments() -> None:
    # Given: Colab may expose more than one system Python installation.
    notebook = load_notebook()

    # When: the dependency-install cell is inspected.
    code = notebook_code(notebook)

    # Then: uv targets the kernel interpreter and validates the exact trainer constructor contract.
    assert '[uv, "pip", "install", "--system", "--python", sys.executable' in code
    assert "transformers.__version__" in code
    assert "TrainingArguments.__module__" in code
    assert "warmup_steps=0.03" in code
    assert "warmup_ratio=0.03" not in code
    assert "eval_strategy=" in code


def test_colab_training_streams_the_child_traceback_to_a_persistent_log() -> None:
    # Given: the cloud trainer can fail inside a separate Python process.
    notebook = load_notebook()

    # When: the resumable training cell is inspected.
    training_cell = next(cell for cell in notebook["cells"] if cell["id"] == "run-resumable-training")
    code = "".join(training_cell["source"])

    # Then: stdout and stderr are streamed together and retained on Drive.
    assert 'TRAINING_LOG = MODEL_DIR / "training.log"' in code
    assert 'sys.executable, "-u", "scripts/train_causal_cloud.py"' in code
    assert "subprocess.Popen(" in code
    assert "stdout=subprocess.PIPE" in code
    assert "stderr=subprocess.STDOUT" in code
    assert "training_process.wait()" in code
