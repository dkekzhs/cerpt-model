from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import unicodedata
import zipfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TypedDict

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

SPLIT_NAMES = ("train", "validation", "test")
SONGYS_LICENSE = "MIT"
OFFICE_LICENSE = "user-acknowledged; redistribution terms not verified"


class CurrentSourceRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_text: str
    target_text: str
    source: str = "cerpt-korean-basic-v6"


class SongysSourceRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    question: str = Field(alias="Q")
    answer: str = Field(alias="A")
    label: int


class OfficeSourceRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int
    domain: str
    user_utterance: str
    system_utterance: str


class PreparedRowPayload(TypedDict):
    id: str
    task_type: str
    source: str
    source_license: str
    input_text: str
    target_text: str


@dataclass(frozen=True, slots=True)
class KoreanConversationSources:
    current_data_dir: Path
    songys_csv: Path
    office_zip: Path
    office_license_acknowledged: bool


@dataclass(frozen=True, slots=True)
class PreparedConversation:
    task_type: str
    source: str
    source_license: str
    input_text: str
    target_text: str

    @property
    def identity(self) -> str:
        return f"{self.input_text}\u241f{self.target_text}"

    def as_payload(self) -> PreparedRowPayload:
        row_id = hashlib.sha256(self.identity.encode("utf-8")).hexdigest()[:20]
        return {
            "id": f"korean-conversation-{row_id}",
            "task_type": self.task_type,
            "source": self.source,
            "source_license": self.source_license,
            "input_text": self.input_text,
            "target_text": self.target_text,
        }


@dataclass(frozen=True, slots=True)
class PreparationSummary:
    counts: Mapping[str, int]
    source_counts: Mapping[str, int]
    duplicates_removed: int


@dataclass(frozen=True, slots=True)
class OfficeLicenseAcknowledgementError(PermissionError):
    archive: Path

    def __str__(self) -> str:
        return f"office dataset license must be acknowledged before reading {self.archive}"


def _clean(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value)
    return re.sub(r"\s+", " ", normalized).strip()


def _read_current(root: Path) -> list[PreparedConversation]:
    rows: list[PreparedConversation] = []
    for split in SPLIT_NAMES:
        with (root / f"{split}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = CurrentSourceRow.model_validate_json(line)
                    rows.append(
                        PreparedConversation(
                            task_type="current",
                            source=row.source,
                            source_license="project-generated",
                            input_text=_clean(row.input_text),
                            target_text=_clean(row.target_text),
                        )
                    )
    return rows


def _read_songys(path: Path) -> list[PreparedConversation]:
    rows: list[PreparedConversation] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            row = SongysSourceRow.model_validate(raw)
            rows.append(
                PreparedConversation(
                    task_type=f"songys_label_{row.label}",
                    source="songys/Chatbot_data",
                    source_license=SONGYS_LICENSE,
                    input_text=f"[TASK_CHAT] {_clean(row.question)}",
                    target_text=f"answer {_clean(row.answer)}",
                )
            )
    return rows


def _read_office(path: Path) -> list[PreparedConversation]:
    rows: list[PreparedConversation] = []
    adapter = TypeAdapter(list[OfficeSourceRow])
    with zipfile.ZipFile(path) as archive:
        for member in sorted(name for name in archive.namelist() if name.lower().endswith(".json")):
            with archive.open(member) as handle:
                for row in adapter.validate_json(handle.read()):
                    rows.append(
                        PreparedConversation(
                            task_type=f"office_{_clean(row.domain).lower()}",
                            source="AI-Hub-derived Korean office dialogue archive",
                            source_license=OFFICE_LICENSE,
                            input_text=f"[TASK_OFFICE:{_clean(row.domain)}] {_clean(row.user_utterance)}",
                            target_text=f"answer {_clean(row.system_utterance)}",
                        )
                    )
    return rows


def _split(rows: list[PreparedConversation], seed: int) -> dict[str, list[PreparedConversation]]:
    groups: dict[str, list[PreparedConversation]] = {}
    for row in rows:
        groups.setdefault(row.task_type, []).append(row)
    result = {name: [] for name in SPLIT_NAMES}
    rng = random.Random(seed)
    for task_type in sorted(groups):
        group = groups[task_type]
        rng.shuffle(group)
        train_end = int(len(group) * 0.8)
        validation_end = train_end + int(len(group) * 0.1)
        result["train"].extend(group[:train_end])
        result["validation"].extend(group[train_end:validation_end])
        result["test"].extend(group[validation_end:])
    for name in SPLIT_NAMES:
        rng.shuffle(result[name])
    return result


def _write_jsonl(path: Path, rows: list[PreparedConversation]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.as_payload(), ensure_ascii=False) + "\n")


def prepare_korean_conversations(
    sources: KoreanConversationSources,
    output_dir: Path,
    seed: int,
) -> PreparationSummary:
    if not sources.office_license_acknowledged:
        raise OfficeLicenseAcknowledgementError(sources.office_zip)
    raw_rows = [
        *_read_current(sources.current_data_dir),
        *_read_songys(sources.songys_csv),
        *_read_office(sources.office_zip),
    ]
    unique_rows = list({row.identity: row for row in reversed(raw_rows)}.values())
    unique_rows.reverse()
    splits = _split(unique_rows, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_NAMES:
        _write_jsonl(output_dir / f"{name}.jsonl", splits[name])
    summary = PreparationSummary(
        counts=MappingProxyType({name: len(splits[name]) for name in SPLIT_NAMES}),
        source_counts=MappingProxyType(dict(Counter(row.source for row in unique_rows))),
        duplicates_removed=len(raw_rows) - len(unique_rows),
    )
    metadata = {
        "seed": seed,
        "split_ratio": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "counts": dict(summary.counts),
        "source_counts": dict(summary.source_counts),
        "duplicates_removed": summary.duplicates_removed,
        "office_license": OFFICE_LICENSE,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
