import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cerpt.models.cerpt_causal import CERPTCausalConfig, CERPTForCausalLM


def _config() -> CERPTCausalConfig:
    return CERPTCausalConfig(
        vocab_size=32,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=32,
        workspace_slots=2,
        num_cycles=2,
        num_operators=8,
        workspace_token_ids=[28, 29, 30, 31],
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def test_causal_forward_and_generation():
    config = _config()
    model = CERPTForCausalLM(config)
    input_ids = torch.tensor(
        [[1, 4, 5, 28, 29, 30, 31, 6], [1, 7, 8, 28, 29, 30, 31, 9]],
    )
    output = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=input_ids,
        operator_labels=torch.zeros((2, 2), dtype=torch.long),
        cycle_valid_labels=torch.ones((2, 2)),
    )
    assert output.loss is not None
    assert output.logits.shape == (2, 8, config.vocab_size)
    assert output.workspace.shape == (2, config.num_cycles, config.workspace_slots, config.hidden_size)
    generated = model.generate(input_ids[:, :3], max_new_tokens=2)
    assert generated.shape == (2, 5)


def test_prefix_logits_do_not_change_when_future_suffix_changes():
    # Given: two sequences with the same prefix and different future tokens.
    model = CERPTForCausalLM(_config()).eval()
    first = torch.tensor([[1, 4, 5, 6, 7, 8]])
    second = torch.tensor([[1, 4, 5, 6, 12, 13]])

    # When: both complete sequences are evaluated.
    with torch.no_grad():
        first_logits = model(input_ids=first).logits[:, :4]
        second_logits = model(input_ids=second).logits[:, :4]

    # Then: future suffixes cannot alter any prefix prediction.
    torch.testing.assert_close(first_logits, second_logits, rtol=0.0, atol=1e-6)


def test_padding_does_not_change_real_token_logits():
    # Given: the same real tokens with and without masked right padding.
    model = CERPTForCausalLM(_config()).eval()
    compact = torch.tensor([[1, 4, 5, 6]])
    padded = torch.tensor([[1, 4, 5, 6, 0, 0, 0, 0]])
    padding_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]])

    # When: logits for the real-token prefix are compared.
    with torch.no_grad():
        compact_logits = model(input_ids=compact, attention_mask=torch.ones_like(compact)).logits
        padded_logits = model(input_ids=padded, attention_mask=padding_mask).logits[:, :4]

    # Then: masked padding has no effect on real-token predictions.
    torch.testing.assert_close(compact_logits, padded_logits, rtol=0.0, atol=1e-6)


def test_cached_decode_matches_full_prefix_decode():
    # Given: a prefix whose first four tokens have already been decoded.
    model = CERPTForCausalLM(_config()).eval()
    input_ids = torch.tensor([[1, 4, 5, 6, 7]])

    # When: the fifth token is decoded from cache and from the full prefix.
    with torch.no_grad():
        prefix = model(input_ids=input_ids[:, :4], use_cache=True)
        assert prefix.past_key_values is not None
        cached = model(
            input_ids=input_ids[:, 4:],
            attention_mask=torch.ones((1, 5), dtype=torch.long),
            past_key_values=prefix.past_key_values,
            use_cache=True,
        )
        full = model(input_ids=input_ids, use_cache=True)

    # Then: KV caching preserves the exact next-token distribution.
    torch.testing.assert_close(cached.logits[:, -1], full.logits[:, -1], rtol=0.0, atol=1e-5)


def test_missing_auxiliary_labels_do_not_change_language_model_loss():
    # Given: a causal conversation with workspace tokens but no operator or verifier annotation.
    model = CERPTForCausalLM(_config()).eval()
    input_ids = torch.tensor([[1, 4, 5, 28, 29, 30, 31, 6]])
    ignored_operators = torch.full((1, 2), -100, dtype=torch.long)
    ignored_validity = torch.full((1, 2), -100.0)

    # When: missing auxiliary targets are represented by the common ignore index.
    with torch.no_grad():
        language_only = model(input_ids=input_ids, labels=input_ids).loss
        mixed_source = model(
            input_ids=input_ids,
            labels=input_ids,
            operator_labels=ignored_operators,
            cycle_valid_labels=ignored_validity,
        ).loss

    # Then: an unannotated Korean dialogue contributes only language-model loss.
    torch.testing.assert_close(mixed_source, language_only, rtol=0.0, atol=1e-7)
