from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypedDict

DATASET_ID = "HuggingFaceFW/fineweb-2"
DATASET_CONFIG = "kor_Hang"
DATASET_LICENSE = "ODC-By-1.0"


class TextRow(TypedDict):
    text: str


@dataclass(frozen=True, slots=True)
class KoreanCorpusSpec:
    max_documents: int | None = 5_000_000
    min_chars: int = 200
    max_chars: int = 12_000
    min_korean_fraction: float = 0.5
    validation_modulus: int = 200
    validation_buckets: int = 1


@dataclass(frozen=True, slots=True)
class KoreanCorpusStats:
    documents_seen: int
    documents_kept: int
    duplicate_documents: int
    filtered_documents: int
    train_chunks: int
    validation_chunks: int


def prepare_korean_corpus(
    rows: Iterable[TextRow],
    output_dir: Path,
    spec: KoreanCorpusSpec,
) -> KoreanCorpusStats:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    seen_hashes: set[bytes] = set()
    documents_seen = 0
    documents_kept = 0
    duplicate_documents = 0
    filtered_documents = 0
    train_chunks = 0
    validation_chunks = 0

    with train_path.open("w", encoding="utf-8") as train_handle, validation_path.open(
        "w", encoding="utf-8"
    ) as validation_handle:
        for row in rows:
            if spec.max_documents is not None and documents_seen >= spec.max_documents:
                break
            documents_seen += 1
            text = normalize_korean_document(row["text"])
            if not _is_usable_korean(text, spec):
                filtered_documents += 1
                continue
            document_hash = hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()
            if document_hash in seen_hashes:
                duplicate_documents += 1
                continue
            seen_hashes.add(document_hash)
            documents_kept += 1
            for chunk in split_document(text, spec.max_chars, spec.min_chars):
                row_json = json.dumps(
                    {"input_text": "", "target_text": chunk},
                    ensure_ascii=False,
                )
                chunk_hash = hashlib.blake2b(chunk.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(chunk_hash, byteorder="big") % spec.validation_modulus
                use_validation = validation_chunks == 0 and train_chunks > 0
                use_validation = use_validation or bucket < spec.validation_buckets
                if use_validation:
                    validation_handle.write(row_json + "\n")
                    validation_chunks += 1
                else:
                    train_handle.write(row_json + "\n")
                    train_chunks += 1

    stats = KoreanCorpusStats(
        documents_seen=documents_seen,
        documents_kept=documents_kept,
        duplicate_documents=duplicate_documents,
        filtered_documents=filtered_documents,
        train_chunks=train_chunks,
        validation_chunks=validation_chunks,
    )
    metadata = {
        "source": {
            "dataset": DATASET_ID,
            "config": DATASET_CONFIG,
            "license": DATASET_LICENSE,
        },
        "spec": asdict(spec),
        "stats": asdict(stats),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return stats


def normalize_korean_document(text: str) -> str:
    paragraphs = [re.sub(r"\s+", " ", paragraph).strip() for paragraph in text.splitlines()]
    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph)


def split_document(text: str, max_chars: int, min_chars: int) -> Iterator[str]:
    for start in range(0, len(text), max_chars):
        chunk = text[start : start + max_chars].strip()
        if len(chunk) >= min_chars:
            yield chunk


def _is_usable_korean(text: str, spec: KoreanCorpusSpec) -> bool:
    if len(text) < spec.min_chars:
        return False
    letters = sum(character.isalpha() for character in text)
    if letters == 0:
        return False
    korean = sum(_is_hangul(character) for character in text)
    return korean / letters >= spec.min_korean_fraction


def _is_hangul(character: str) -> bool:
    codepoint = ord(character)
    return (
        0xAC00 <= codepoint <= 0xD7A3
        or 0x1100 <= codepoint <= 0x11FF
        or 0x3130 <= codepoint <= 0x318F
    )
