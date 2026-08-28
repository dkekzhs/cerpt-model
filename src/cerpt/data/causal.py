from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypedDict

import torch
from transformers import PreTrainedTokenizerBase


class ConversationRow(TypedDict):
    input_text: str
    target_text: str


class CausalBatch(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class PromptBatch(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class MissingSpecialTokenError(ValueError):
    token_name: str

    def __str__(self) -> str:
        return f"causal formatting requires tokenizer.{self.token_name}_token_id"


@dataclass(frozen=True, slots=True)
class SequenceBudgetError(ValueError):
    max_length: int
    workspace_tokens: int

    def __str__(self) -> str:
        return (
            f"max_length={self.max_length} cannot contain BOS, EOS, and "
            f"{self.workspace_tokens} workspace tokens"
        )


def workspace_token_names(num_cycles: int, workspace_slots: int) -> list[str]:
    return [
        f"<|cerpt_workspace_c{cycle:02d}_s{slot:02d}|>"
        for cycle in range(num_cycles)
        for slot in range(workspace_slots)
    ]


def add_workspace_tokens(
    tokenizer: PreTrainedTokenizerBase,
    num_cycles: int,
    workspace_slots: int,
) -> list[int]:
    names = workspace_token_names(num_cycles, workspace_slots)
    tokenizer.add_special_tokens({"additional_special_tokens": names})
    vocab = tokenizer.get_vocab()
    return [vocab[name] for name in names]


@dataclass(frozen=True, slots=True)
class CausalBatchFormatter:
    tokenizer: PreTrainedTokenizerBase
    workspace_token_ids: tuple[int, ...]
    max_length: int

    def __post_init__(self) -> None:
        if self.tokenizer.pad_token_id is None:
            raise MissingSpecialTokenError("pad")
        if self.tokenizer.bos_token_id is None:
            raise MissingSpecialTokenError("bos")
        if self.tokenizer.eos_token_id is None:
            raise MissingSpecialTokenError("eos")
        if self.max_length <= len(self.workspace_token_ids) + 1:
            raise SequenceBudgetError(self.max_length, len(self.workspace_token_ids))

    def _prompt_context(self, prompt: str, response_tokens: int) -> list[int]:
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        budget = self.max_length - 1 - len(self.workspace_token_ids) - response_tokens
        return prompt_ids[-budget:] if budget > 0 else []

    def _encode_pair(self, row: ConversationRow) -> tuple[list[int], list[int]]:
        eos_token_id = self.tokenizer.eos_token_id
        bos_token_id = self.tokenizer.bos_token_id
        if eos_token_id is None:
            raise MissingSpecialTokenError("eos")
        if bos_token_id is None:
            raise MissingSpecialTokenError("bos")
        response_budget = self.max_length - 1 - len(self.workspace_token_ids)
        response_ids = self.tokenizer.encode(row["target_text"], add_special_tokens=False)
        response_ids = response_ids[: response_budget - 1] + [eos_token_id]
        prompt_ids = self._prompt_context(row["input_text"], len(response_ids))
        context = [bos_token_id, *prompt_ids, *self.workspace_token_ids]
        return [*context, *response_ids], [-100] * len(context) + response_ids

    def collate(self, rows: Sequence[ConversationRow]) -> CausalBatch:
        examples = [self._encode_pair(row) for row in rows]
        width = max(len(input_ids) for input_ids, _ in examples)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise MissingSpecialTokenError("pad")
        input_ids = torch.full((len(examples), width), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(examples), width), dtype=torch.long)
        labels = torch.full((len(examples), width), -100, dtype=torch.long)
        for index, (tokens, targets) in enumerate(examples):
            length = len(tokens)
            input_ids[index, :length] = torch.tensor(tokens, dtype=torch.long)
            attention_mask[index, :length] = 1
            labels[index, :length] = torch.tensor(targets, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def encode_prompts(self, prompts: Sequence[str]) -> PromptBatch:
        bos_token_id = self.tokenizer.bos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        if bos_token_id is None:
            raise MissingSpecialTokenError("bos")
        if pad_token_id is None:
            raise MissingSpecialTokenError("pad")
        sequences = [
            [bos_token_id, *self._prompt_context(prompt, 0), *self.workspace_token_ids]
            for prompt in prompts
        ]
        width = max(len(sequence) for sequence in sequences)
        input_ids = torch.full((len(sequences), width), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(sequences), width), dtype=torch.long)
        for index, sequence in enumerate(sequences):
            start = width - len(sequence)
            input_ids[index, start:] = torch.tensor(sequence, dtype=torch.long)
            attention_mask[index, start:] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}
