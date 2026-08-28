"""Small offline-friendly Hugging Face tokenizer for the synthetic curriculum."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer, decoders, normalizers, trainers
from tokenizers.models import BPE, WordLevel
from tokenizers.pre_tokenizers import ByteLevel, Whitespace
from transformers import PreTrainedTokenizerFast

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]


@dataclass(frozen=True, slots=True)
class TokenizerVocabularySizeError(ValueError):
    expected: int
    actual: int

    def __str__(self) -> str:
        return f"tokenizer training produced {self.actual} tokens; expected exactly {self.expected}"


@dataclass(frozen=True, slots=True)
class CausalTokenizerSpec:
    final_vocab_size: int
    workspace_tokens: int
    model_max_length: int


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


def build_causal_bpe_tokenizer(
    texts: Iterable[str],
    spec: CausalTokenizerSpec,
) -> PreTrainedTokenizerFast:
    learned_vocab_size = spec.final_vocab_size - spec.workspace_tokens
    backend = Tokenizer(BPE(unk_token="[UNK]", byte_fallback=True))
    backend.normalizer = normalizers.NFC()
    backend.pre_tokenizer = ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        texts,
        trainer=trainers.BpeTrainer(
            vocab_size=learned_vocab_size,
            min_frequency=1,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=ByteLevel.alphabet(),
            show_progress=False,
        ),
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        model_max_length=spec.model_max_length,
        clean_up_tokenization_spaces=False,
    )
    if len(tokenizer) != learned_vocab_size:
        raise TokenizerVocabularySizeError(learned_vocab_size, len(tokenizer))
    return tokenizer


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
