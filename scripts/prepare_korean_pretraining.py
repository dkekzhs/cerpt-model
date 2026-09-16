from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

from datasets import IterableDataset, load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cerpt.data.korean_pretraining import (
    DATASET_CONFIG,
    DATASET_ID,
    KoreanCorpusSpec,
    TextRow,
    prepare_korean_corpus,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream and prepare Korean-only causal pretraining data")
    parser.add_argument("--output-dir", type=Path, default=Path("data/pretraining_shards"))
    parser.add_argument("--max-documents", type=int, default=5_000_000, help="0 streams the full Korean corpus")
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--max-chars", type=int, default=12_000)
    args = parser.parse_args()
    if args.max_documents < 0:
        parser.error("--max-documents must be zero or greater")
    max_documents = None if args.max_documents == 0 else args.max_documents
    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split="train", streaming=True)
    stats = prepare_korean_corpus(
        _source_rows(dataset),
        args.output_dir,
        KoreanCorpusSpec(
            max_documents=max_documents,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
        ),
    )
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), **asdict(stats)}, ensure_ascii=False))


def _source_rows(dataset: IterableDataset) -> Iterator[TextRow]:
    for row in dataset:
        yield {"text": row["text"]}


if __name__ == "__main__":
    main()
