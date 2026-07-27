"""Small offline-friendly Hugging Face tokenizer for the synthetic curriculum."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]


def build_tokenizer(texts: Iterable[str]) -> PreTrainedTokenizerFast:
    vocab = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    pre_tokenizer = Whitespace()
    for text in texts:
        # Use exactly the same pre-tokenization rule for vocabulary creation
        # and encoding, so `4.` in a prompt and `4` in a trace share the `4`
        # token instead of becoming unrelated whole-word tokens.
        tokens = [token for token, _ in pre_tokenizer.pre_tokenize_str(text)]
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizer
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        model_max_length=256,
    )


def save_tokenizer(tokenizer: PreTrainedTokenizerFast, path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(path)


def load_jsonl_texts(path: str | Path) -> list[str]:
    texts: list[str] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            texts.extend([row["input_text"], row["target_text"]])
    return texts
