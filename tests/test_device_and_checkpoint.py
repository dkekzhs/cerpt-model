import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cerpt.models.cerpt_causal import CERPTCausalConfig, CERPTForCausalLM
from cerpt.utils.device import select_device


def test_cpu_device_and_gradient_checkpointing_step():
    assert select_device("cpu") == torch.device("cpu")
    config = CERPTCausalConfig(vocab_size=24, hidden_size=24, num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=16)
    model = CERPTForCausalLM(config)
    model.enable_gradient_checkpointing()
    input_ids = torch.randint(0, config.vocab_size, (1, 6))
    output = model(input_ids=input_ids, labels=input_ids)
    output.loss.backward()
    assert output.loss is not None
