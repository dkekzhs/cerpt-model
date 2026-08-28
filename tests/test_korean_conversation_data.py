from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import cerpt.data as cerpt_data


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_prepare_korean_conversations_combines_deduplicates_and_splits(tmp_path: Path) -> None:
    # Given: one current row, two Songys rows with a duplicate, and one office row.
    current_dir = tmp_path / "current"
    current_dir.mkdir()
    current_row = {"input_text": "[TASK_CHAT] 안녕", "target_text": "answer 반가워"}
    for split in ("train", "validation", "test"):
        _write_jsonl(current_dir / f"{split}.jsonl", [current_row] if split == "train" else [])
    songys_csv = tmp_path / "ChatbotData.csv"
    with songys_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Q", "A", "label"])
        writer.writeheader()
        writer.writerows(
            [
                {"Q": "안녕", "A": "반가워", "label": "0"},
                {"Q": "오늘 어때", "A": "좋은 하루예요", "label": "0"},
            ]
        )
    office_zip = tmp_path / "office.zip"
    with zipfile.ZipFile(office_zip, "w") as archive:
        archive.writestr(
            "output_task.json",
            json.dumps(
                [
                    {
                        "index": 1,
                        "domain": "schedule",
                        "user_utterance": "회의 잡아줘",
                        "system_utterance": "내일 오전으로 잡았습니다",
                    }
                ],
                ensure_ascii=False,
            ),
        )
    output_dir = tmp_path / "prepared"
    sources = cerpt_data.KoreanConversationSources(
        current_data_dir=current_dir,
        songys_csv=songys_csv,
        office_zip=office_zip,
        office_license_acknowledged=True,
    )

    # When: the combined dataset is prepared twice with the same seed.
    first = cerpt_data.prepare_korean_conversations(sources, output_dir, seed=7)
    second = cerpt_data.prepare_korean_conversations(sources, output_dir, seed=7)

    # Then: the exact duplicate is removed and every split is deterministic.
    assert sum(first.counts.values()) == 3
    assert first == second
    assert first.duplicates_removed == 1
    assert all((output_dir / f"{name}.jsonl").exists() for name in first.counts)


def test_prepare_korean_conversations_requires_office_license_acknowledgement(tmp_path: Path) -> None:
    # Given: source paths whose office-data license has not been acknowledged.
    sources = cerpt_data.KoreanConversationSources(
        current_data_dir=tmp_path,
        songys_csv=tmp_path / "songys.csv",
        office_zip=tmp_path / "office.zip",
        office_license_acknowledged=False,
    )

    # When/Then: preparation stops before reading or redistributing the office archive.
    try:
        cerpt_data.prepare_korean_conversations(sources, tmp_path / "out", seed=7)
    except cerpt_data.OfficeLicenseAcknowledgementError as error:
        assert error.archive == sources.office_zip
    else:
        raise AssertionError("office-data preparation must require an explicit license acknowledgement")
