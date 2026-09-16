from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from cerpt.data.korean_pretraining import (
    KoreanCorpusSpec,
    TextRow,
    prepare_korean_corpus,
)

ROOT = Path(__file__).resolve().parents[1]


def test_korean_corpus_preparation_filters_splits_and_deduplicates(tmp_path: Path) -> None:
    # Given: Korean documents mixed with duplicate, short, and non-Korean rows.
    korean = "한국어 모델을 학습하기 위한 충분히 긴 문장입니다. " * 12
    rows: list[TextRow] = [
        {"text": f"{korean} 문서 번호 {index}"}
        for index in range(20)
    ]
    rows.extend(
        [
            {"text": korean},
            {"text": korean},
            {"text": "짧은 글"},
            {"text": "English-only pretraining document. " * 20},
        ]
    )

    # When: the streaming rows are converted to CERPT causal JSONL.
    stats = prepare_korean_corpus(
        rows,
        tmp_path,
        KoreanCorpusSpec(
            max_documents=None,
            min_chars=100,
            max_chars=2_000,
            min_korean_fraction=0.5,
            validation_modulus=4,
            validation_buckets=1,
        ),
    )

    # Then: only unique Korean text is retained and both splits use the model schema.
    train_rows = _read_jsonl(tmp_path / "train.jsonl")
    validation_rows = _read_jsonl(tmp_path / "validation.jsonl")
    all_rows = [*train_rows, *validation_rows]
    assert stats.documents_seen == 24
    assert stats.documents_kept == 21
    assert len(all_rows) == 21
    assert train_rows
    assert validation_rows
    assert all(row["input_text"] == "" for row in all_rows)
    assert len({row["target_text"] for row in all_rows}) == 21
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["source"]["dataset"] == "HuggingFaceFW/fineweb-2"
    assert metadata["source"]["config"] == "kor_Hang"
    assert metadata["source"]["license"] == "ODC-By-1.0"


def test_korean_corpus_preparation_honors_source_document_limit(tmp_path: Path) -> None:
    # Given: more valid Korean documents than the requested source limit.
    rows = (TextRow(text=("한국어 사전학습 문서입니다. " * 20) + str(index)) for index in range(10))

    # When: preparation is limited to three source documents.
    stats = prepare_korean_corpus(
        rows,
        tmp_path,
        KoreanCorpusSpec(max_documents=3, min_chars=100),
    )

    # Then: the stream stops without reading or writing later documents.
    assert stats.documents_seen == 3
    assert stats.documents_kept == 3


def test_mac_launcher_uses_project_venv_and_exposes_automatic_korean_preparation() -> None:
    # Given: the Finder launcher and Python CLI on the main branch.
    launcher = (ROOT / "scripts" / "train_mac.command").read_text(encoding="utf-8")

    # When: the launcher's executable and CLI contract are inspected.
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train_mac.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    # Then: Finder uses the populated project venv and 3B can prepare missing assets.
    assert result.returncode == 0, result.stderr
    assert '.venv/bin/python' in launcher
    assert "--corpus-documents" in result.stdout
    assert "--tokenizer-dir" in result.stdout
    assert "--data-dir" in result.stdout


def _read_jsonl(path: Path) -> list[dict[str, str]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
