"""A compact CERPT Phase-1 model implemented with PyTorch and Transformers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import ModelOutput


@dataclass
class CERPTOutput(ModelOutput):
    loss: Optional[Tensor] = None
    logits: Optional[Tensor] = None
    cycle_valid_logits: Optional[Tensor] = None
    operator_logits: Optional[Tensor] = None
    commit_gates: Optional[Tensor] = None
    workspace: Optional[Tensor] = None


class CERPTConfig(PretrainedConfig):
    model_type = "cerpt"

    def __init__(
        self,
        vocab_size: int = 4096,
        hidden_size: int = 128,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        num_attention_heads: int = 4,
        intermediate_size: int = 256,
        max_position_embeddings: int = 256,
        workspace_slots: int = 16,
        num_cycles: int = 4,
        num_operators: int = 6,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        **kwargs,
    ):
        is_encoder_decoder = kwargs.pop("is_encoder_decoder", True)
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, is_encoder_decoder=is_encoder_decoder, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.workspace_slots = workspace_slots
        self.num_cycles = num_cycles
        self.num_operators = num_operators
        self.dropout = dropout
        self.decoder_start_token_id = bos_token_id


class CERPTForConditionalGeneration(PreTrainedModel):
    """Encoder-decoder generation with persistent typed workspace recursion.

    The workspace is not just a latent token: each cycle has a shared
    transition core, an operator prediction, and a learned verification gate.
    The gate controls how much of the proposed state is committed.
    """

    config_class = CERPTConfig
    base_model_prefix = "cerpt"

    def __init__(self, config: CERPTConfig):
        super().__init__(config)
        d = config.hidden_size
        self.token_embeddings = nn.Embedding(config.vocab_size, d, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, d)
        encoder_layer = nn.TransformerEncoderLayer(d, config.num_attention_heads, config.intermediate_size, config.dropout, activation="gelu", batch_first=True, norm_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d, config.num_attention_heads, config.intermediate_size, config.dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, config.num_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, config.num_decoder_layers)
        self.layer_norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(config.dropout)

        self.workspace_seed = nn.Parameter(torch.randn(config.workspace_slots, d) * 0.02)
        self.slot_type_embeddings = nn.Parameter(torch.randn(config.workspace_slots, d) * 0.02)
        self.cycle_embeddings = nn.Parameter(torch.randn(config.num_cycles, d) * 0.02)
        self.operator_embeddings = nn.Parameter(torch.randn(config.num_operators, d) * 0.02)
        self.transition_attention = nn.MultiheadAttention(d, config.num_attention_heads, dropout=config.dropout, batch_first=True)
        transition_layer = nn.TransformerEncoderLayer(d, config.num_attention_heads, config.intermediate_size, config.dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transition_core = nn.TransformerEncoder(transition_layer, 1)
        self.operator_head = nn.Linear(d, config.num_operators)
        self.verifier_head = nn.Sequential(nn.Linear(d * 2, d), nn.GELU(), nn.Linear(d, 1))
        # Keep the output projection independent in the first PoC.  Explicit
        # tying requires declaring `_tied_weights_keys` for Transformers' safe
        # serialization path; independent weights make save/load behavior
        # unambiguous while we compare architecture effects.
        self.lm_head = nn.Linear(d, config.vocab_size, bias=False)
        self.post_init()

    def _positions(self, length: int, device: torch.device) -> Tensor:
        return torch.arange(length, device=device).unsqueeze(0)

    def encode(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        vision_features: Optional[Tensor] = None,
        vision_attention_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).long()
        if input_ids.size(1) > self.config.max_position_embeddings:
            raise ValueError("input sequence exceeds max_position_embeddings")
        hidden = self.token_embeddings(input_ids) + self.position_embeddings(self._positions(input_ids.size(1), input_ids.device))
        hidden = self.dropout(hidden)
        hidden = self.layer_norm(self.encoder(hidden, src_key_padding_mask=attention_mask.eq(0)))
        if vision_features is not None:
            if vision_features.ndim != 3 or vision_features.size(-1) != self.config.hidden_size:
                raise ValueError(
                    "vision_features must have shape [batch, tokens, hidden_size] "
                    f"with hidden_size={self.config.hidden_size}"
                )
            if vision_features.size(0) != input_ids.size(0):
                raise ValueError("vision_features batch size must match input_ids")
            if vision_attention_mask is None:
                vision_attention_mask = input_ids.new_ones((input_ids.size(0), vision_features.size(1)))
            if vision_attention_mask.shape != vision_features.shape[:2]:
                raise ValueError("vision_attention_mask must match vision_features[:2]")
            hidden = torch.cat([hidden, vision_features.to(hidden.dtype)], dim=1)
            attention_mask = torch.cat([attention_mask, vision_attention_mask.to(attention_mask.dtype)], dim=1)
        return hidden, attention_mask

    def build_workspace(self, memory: Tensor, attention_mask: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch = memory.size(0)
        denom = attention_mask.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = (memory * attention_mask.unsqueeze(-1)).sum(dim=1) / denom
        workspace = self.workspace_seed.unsqueeze(0).expand(batch, -1, -1) + self.slot_type_embeddings.unsqueeze(0)
        workspace = workspace + pooled.unsqueeze(1)
        all_operator_logits = []
        all_valid_logits = []
        all_gates = []
        for cycle in range(self.config.num_cycles):
            current_summary = workspace.mean(dim=1)
            operator_logits = self.operator_head(current_summary)
            operator = operator_logits.argmax(dim=-1)
            op_embedding = self.operator_embeddings[operator]
            query = workspace + self.cycle_embeddings[cycle].view(1, 1, -1) + op_embedding.unsqueeze(1)
            attended, _ = self.transition_attention(query, memory, memory, key_padding_mask=attention_mask.eq(0), need_weights=False)
            proposal = self.transition_core(query + attended)
            proposed_summary = proposal.mean(dim=1)
            valid_logits = self.verifier_head(torch.cat([current_summary, proposed_summary], dim=-1)).squeeze(-1)
            gates = torch.sigmoid(valid_logits).unsqueeze(-1).unsqueeze(-1)
            workspace = workspace + gates * (proposal - workspace)
            all_operator_logits.append(operator_logits)
            all_valid_logits.append(valid_logits)
            all_gates.append(gates.squeeze(-1).squeeze(-1))
        return workspace, torch.stack(all_valid_logits, dim=1), torch.stack(all_operator_logits, dim=1), torch.stack(all_gates, dim=1)

    def _shift_right(self, labels: Tensor) -> Tensor:
        shifted = labels.new_full(labels.shape, self.config.pad_token_id)
        shifted[:, 0] = self.config.decoder_start_token_id
        shifted[:, 1:] = labels[:, :-1].masked_fill(labels[:, :-1].eq(-100), self.config.pad_token_id)
        return shifted

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        decoder_input_ids: Optional[Tensor] = None,
        decoder_attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        cycle_valid_labels: Optional[Tensor] = None,
        operator_labels: Optional[Tensor] = None,
        vision_features: Optional[Tensor] = None,
        vision_attention_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> CERPTOutput:
        memory, attention_mask = self.encode(input_ids, attention_mask, vision_features, vision_attention_mask)
        workspace, valid_logits, operator_logits, gates = self.build_workspace(memory, attention_mask)
        if labels is not None and decoder_input_ids is None:
            decoder_input_ids = self._shift_right(labels)
        if decoder_input_ids is None:
            decoder_input_ids = input_ids.new_full((input_ids.size(0), 1), self.config.decoder_start_token_id)
        if decoder_input_ids.size(1) > self.config.max_position_embeddings:
            raise ValueError("decoder sequence exceeds max_position_embeddings")
        decoder_hidden = self.token_embeddings(decoder_input_ids) + self.position_embeddings(self._positions(decoder_input_ids.size(1), input_ids.device))
        decoder_hidden = self.dropout(decoder_hidden)
        workspace_mask = input_ids.new_ones((input_ids.size(0), self.config.workspace_slots))
        decoder_memory = torch.cat([memory, workspace], dim=1)
        decoder_memory_mask = torch.cat([attention_mask, workspace_mask], dim=1)
        target_length = decoder_input_ids.size(1)
        causal_mask = torch.triu(torch.ones(target_length, target_length, device=input_ids.device, dtype=torch.bool), diagonal=1)
        if decoder_attention_mask is None:
            decoder_attention_mask = decoder_input_ids.ne(self.config.pad_token_id).long()
        decoded = self.decoder(
            decoder_hidden,
            decoder_memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=decoder_attention_mask.eq(0),
            memory_key_padding_mask=decoder_memory_mask.eq(0),
        )
        logits = self.lm_head(self.layer_norm(decoded))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
            if cycle_valid_labels is not None:
                loss = loss + 0.15 * nn.functional.binary_cross_entropy_with_logits(valid_logits, cycle_valid_labels.float())
            if operator_labels is not None:
                loss = loss + 0.15 * nn.functional.cross_entropy(operator_logits.reshape(-1, operator_logits.size(-1)), operator_labels.reshape(-1), ignore_index=-100)
        return CERPTOutput(loss=loss, logits=logits, cycle_valid_logits=valid_logits, operator_logits=operator_logits, commit_gates=gates, workspace=workspace)

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        max_new_tokens: int = 64,
        vision_features: Optional[Tensor] = None,
        vision_attention_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """Greedy generation kept local so the PoC works without a HF model hub."""
        self.eval()
        generated = input_ids.new_full((input_ids.size(0), 1), self.config.decoder_start_token_id)
        finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)
        for _ in range(max_new_tokens):
            output = self(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=generated,
                vision_features=vision_features,
                vision_attention_mask=vision_attention_mask,
            )
            next_token = output.logits[:, -1].argmax(dim=-1)
            next_token = torch.where(finished, input_ids.new_full(next_token.shape, self.config.eos_token_id), next_token)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            finished = finished | next_token.eq(self.config.eos_token_id)
            if bool(finished.all()):
                break
        return generated
