# /// script
# requires-python = ">=3.10"
# dependencies = ["pydantic>=2.7"]
# ///
# How to run: uv run --project . python scripts/prepare_korean_conversations.py --help

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cerpt.data.korean_conversations import (
    KoreanConversationSources,
    prepare_korean_conversations,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine Korean conversation sources for CERPT training")
    parser.add_argument("--current-data-dir", type=Path, default=Path("data/korean_basic_v6"))
    parser.add_argument("--songys-csv", type=Path, required=True)
    parser.add_argument("--office-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/korean_conversations_v7"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--acknowledge-office-license", action="store_true")
    args = parser.parse_args()
    summary = prepare_korean_conversations(
        KoreanConversationSources(
            current_data_dir=args.current_data_dir,
            songys_csv=args.songys_csv,
            office_zip=args.office_zip,
            office_license_acknowledged=args.acknowledge_office_license,
        ),
        args.output_dir,
        args.seed,
    )
    print(
        json.dumps(
            {
                "counts": dict(summary.counts),
                "source_counts": dict(summary.source_counts),
                "duplicates_removed": summary.duplicates_removed,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
