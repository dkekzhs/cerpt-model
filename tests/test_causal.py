import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cerpt.models.cerpt_causal import CERPTCausalConfig, CERPTForCausalLM


def test_causal_forward_and_generation():
    config = CERPTCausalConfig(
        vocab_size=32,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=32,
        workspace_slots=4,
        num_cycles=3,
        num_operators=8,
    )
    model = CERPTForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    output = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=input_ids,
        operator_labels=torch.zeros((2, 3), dtype=torch.long),
        cycle_valid_labels=torch.ones((2, 3)),
    )
    assert output.loss is not None
    assert output.logits.shape == (2, 8, config.vocab_size)
    assert output.workspace.shape == (2, config.num_cycles, config.workspace_slots, config.hidden_size)
    generated = model.generate(input_ids[:, :3], max_new_tokens=2)
    assert generated.shape == (2, 5)
