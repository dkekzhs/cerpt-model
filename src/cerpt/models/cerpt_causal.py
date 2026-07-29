from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


@dataclass
class CERPTCausalOutput(CausalLMOutputWithPast):
    """Causal-LM output with CERPT's inspectable workspace predictions."""

    operator_logits: Optional[torch.Tensor] = None
    cycle_valid_logits: Optional[torch.Tensor] = None
    workspace: Optional[torch.Tensor] = None


class CERPTCausalConfig(PretrainedConfig):
    model_type = "cerpt_causal"

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 256,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 4,
        intermediate_size: int | None = None,
        max_position_embeddings: int = 1024,
        workspace_slots: int = 8,
        num_cycles: int = 4,
        num_operators: int = 8,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size or hidden_size * 4
        self.max_position_embeddings = max_position_embeddings
        self.workspace_slots = workspace_slots
        self.num_cycles = num_cycles
        self.num_operators = num_operators
        self.dropout = dropout
        self.is_decoder = True
        self.use_cache = False


class CERPTForCausalLM(PreTrainedModel):
    """A from-scratch decoder-only CERPT Base scaffold.

    The language model remains causal, while every forward pass also exposes a
    typed workspace, operator logits, and verifier logits. This is intentionally
    an auditable research implementation; a real KV cache and vLLM adapter are
    separate follow-up work.
    """

    config_class = CERPTCausalConfig
    base_model_prefix = "cerpt"
    _supports_cache_class = False

    def __init__(self, config: CERPTCausalConfig):
        super().__init__(config)
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(layer, config.num_hidden_layers)
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.workspace_seed = nn.Parameter(torch.randn(config.workspace_slots, config.hidden_size) * 0.02)
        transition_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transition_core = nn.TransformerEncoder(transition_layer, 1)
        self.operator_head = nn.Linear(config.hidden_size, config.num_operators)
        self.verifier_head = nn.Linear(config.hidden_size, 1)
        self.gradient_checkpointing = False
        self.post_init()

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def _decode(self, hidden: torch.Tensor, causal_mask: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            for layer in self.decoder.layers:
                hidden = checkpoint(
                    lambda values: layer(values, src_mask=causal_mask, src_key_padding_mask=padding_mask),
                    hidden,
                    use_reentrant=False,
                )
            return hidden
        return self.decoder(hidden, mask=causal_mask, src_key_padding_mask=padding_mask)

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)

    def _workspace(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = hidden.size(0)
        workspace = self.workspace_seed.unsqueeze(0).expand(batch, -1, -1)
        cycles = []
        for _ in range(self.config.num_cycles):
            evidence = hidden.mean(dim=1, keepdim=True)
            workspace = self.transition_core(torch.cat([workspace, evidence], dim=1))[:, : self.config.workspace_slots]
            cycles.append(workspace)
        workspace = torch.stack(cycles, dim=1)
        summary = workspace.mean(dim=(1, 2))
        return workspace, workspace.mean(dim=2), summary

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        operator_labels: torch.Tensor | None = None,
        cycle_valid_labels: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> CERPTCausalOutput:
        del past_key_values, use_cache, kwargs
        batch, length = input_ids.shape
        if length > self.config.max_position_embeddings:
            raise ValueError("input sequence exceeds max_position_embeddings")
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0)
        hidden = self.token_embeddings(input_ids) + self.position_embeddings(positions)
        padding_mask = attention_mask.eq(0) if attention_mask is not None else None
        hidden = self._decode(hidden, self._causal_mask(length, input_ids.device), padding_mask)
        hidden = self.final_norm(hidden)
        workspace, cycle_summary, summary = self._workspace(hidden)
        hidden = hidden + summary[:, None, :]
        logits = self.lm_head(hidden)
        operator_logits = self.operator_head(cycle_summary)
        cycle_valid_logits = self.verifier_head(cycle_summary).squeeze(-1)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100
            )
            if operator_labels is not None:
                operator_labels = operator_labels[..., : operator_logits.size(1)]
                loss = loss + 0.05 * nn.functional.cross_entropy(
                    operator_logits.reshape(-1, operator_logits.size(-1)), operator_labels.reshape(-1), ignore_index=-100
                )
            if cycle_valid_labels is not None:
                cycle_valid_labels = cycle_valid_labels[..., : cycle_valid_logits.size(1)]
                loss = loss + 0.05 * nn.functional.binary_cross_entropy_with_logits(
                    cycle_valid_logits, cycle_valid_labels.float()
                )
        return CERPTCausalOutput(
            loss=loss,
            logits=logits,
            past_key_values=None,
            operator_logits=operator_logits,
            cycle_valid_logits=cycle_valid_logits,
            workspace=workspace,
        )

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, max_new_tokens: int = 64, eos_token_id: int | None = None, **kwargs):
        del kwargs
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        for _ in range(max_new_tokens):
            output = self(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
            if eos_token_id is not None and bool(next_token.eq(eos_token_id).all()):
                break
        return input_ids
