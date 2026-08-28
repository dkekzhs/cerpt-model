from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils.generic import TransformersKwargs
from typing_extensions import Unpack


@dataclass(slots=True)  # noqa: MUTABLE_OK -- Transformers ModelOutput requires mutable fields.
class CERPTCausalOutput(CausalLMOutputWithPast):
    """Standard causal-LM output plus inspectable in-stream workspace states."""

    operator_logits: torch.Tensor | None = None
    cycle_valid_logits: torch.Tensor | None = None
    workspace: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class WorkspaceConfigurationError(ValueError):
    expected_tokens: int
    actual_tokens: int

    def __str__(self) -> str:
        return (
            "workspace_token_ids must contain exactly "
            f"{self.expected_tokens} unique ids; received {self.actual_tokens}"
        )


@dataclass(frozen=True, slots=True)
class WorkspaceTokensMissingError(RuntimeError):
    expected_tokens: int

    def __str__(self) -> str:
        return f"auxiliary CERPT labels require all {self.expected_tokens} workspace tokens"


class CERPTCausalConfig(LlamaConfig):
    model_type = "cerpt_causal"

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 256,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 4,
        num_key_value_heads: int | None = None,
        intermediate_size: int | None = None,
        max_position_embeddings: int = 1024,
        workspace_slots: int = 8,
        num_cycles: int = 4,
        num_operators: int = 8,
        workspace_token_ids: list[int] | None = None,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        attention_dropout = float(kwargs.pop("attention_dropout", dropout))
        hidden_act = str(kwargs.pop("hidden_act", "silu"))
        rms_norm_eps = float(kwargs.pop("rms_norm_eps", 1e-6))
        tie_word_embeddings = bool(kwargs.pop("tie_word_embeddings", False))
        use_cache = bool(kwargs.pop("use_cache", True))
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size or hidden_size * 4,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            attention_dropout=attention_dropout,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        token_ids = [] if workspace_token_ids is None else workspace_token_ids
        expected_tokens = workspace_slots * num_cycles
        if token_ids and (len(token_ids) != expected_tokens or len(set(token_ids)) != expected_tokens):
            raise WorkspaceConfigurationError(expected_tokens, len(set(token_ids)))
        self.workspace_slots = workspace_slots
        self.num_cycles = num_cycles
        self.num_operators = num_operators
        self.workspace_token_ids = token_ids
        self.dropout = dropout


class CERPTForCausalLM(LlamaForCausalLM):
    """Llama-compatible decoder with causal in-stream CERPT workspace tokens."""

    config_class = CERPTCausalConfig

    def __init__(self, config: CERPTCausalConfig) -> None:
        super().__init__(config)
        self.operator_head = nn.Linear(config.hidden_size, config.num_operators)
        self.verifier_head = nn.Linear(config.hidden_size, 1)
        self._init_weights(self.operator_head)
        self._init_weights(self.verifier_head)

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing_enable()

    def _extract_workspace(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if input_ids is None or not self.config.workspace_token_ids:
            return None
        workspace_ids = input_ids.new_tensor(self.config.workspace_token_ids)
        matches = input_ids.unsqueeze(-1).eq(workspace_ids)
        if not bool(matches.sum(dim=1).eq(1).all()):
            return None
        positions = matches.to(dtype=torch.int64).argmax(dim=1)
        indices = positions.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
        workspace = hidden_states.gather(dim=1, index=indices)
        return workspace.reshape(
            hidden_states.size(0),
            self.config.num_cycles,
            self.config.workspace_slots,
            hidden_states.size(-1),
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        operator_labels: torch.LongTensor | None = None,
        cycle_valid_labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CERPTCausalOutput:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        workspace = self._extract_workspace(hidden_states, input_ids)
        operator_logits = None
        cycle_valid_logits = None
        if workspace is not None:
            cycle_states = workspace.mean(dim=2)
            operator_logits = self.operator_head(cycle_states)
            cycle_valid_logits = self.verifier_head(cycle_states).squeeze(-1)
        if operator_labels is not None or cycle_valid_labels is not None:
            if workspace is None or loss is None or operator_logits is None or cycle_valid_logits is None:
                raise WorkspaceTokensMissingError(self.config.workspace_slots * self.config.num_cycles)
            if operator_labels is not None:
                operator_mask = operator_labels.ne(-100)
                if bool(operator_mask.any()):
                    loss = loss + 0.05 * nn.functional.cross_entropy(
                        operator_logits[operator_mask],
                        operator_labels[operator_mask],
                    )
            if cycle_valid_labels is not None:
                validity_mask = cycle_valid_labels.ne(-100)
                if bool(validity_mask.any()):
                    loss = loss + 0.05 * nn.functional.binary_cross_entropy_with_logits(
                        cycle_valid_logits[validity_mask],
                        cycle_valid_labels[validity_mask].float(),
                    )

        return CERPTCausalOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            operator_logits=operator_logits,
            cycle_valid_logits=cycle_valid_logits,
            workspace=workspace,
        )
