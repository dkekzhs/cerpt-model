import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cerpt.models.cerpt_causal import CERPTCausalConfig, CERPTForCausalLM
from scripts.estimate_causal_params import estimate


def test_estimator_matches_materialized_llama_core():
    # Given: a small GQA CERPT configuration that can be allocated in a unit test.
    architecture = {
        "vocab_size": 64,
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 64,
        "max_position_embeddings": 128,
        "workspace_slots": 2,
        "num_cycles": 2,
        "num_operators": 8,
        "tie_word_embeddings": False,
    }
    model = CERPTForCausalLM(CERPTCausalConfig(**architecture))

    # When: the allocation-free estimator evaluates the same configuration.
    estimated = estimate(architecture)

    # Then: it counts every trainable parameter exactly.
    assert estimated == sum(parameter.numel() for parameter in model.parameters())


def test_three_billion_preset_is_really_near_three_billion_parameters():
    # Given: the production-scale architecture preset.
    config_path = Path(__file__).resolve().parents[1] / "configs" / "cerpt-causal-3b.json"
    architecture = json.loads(config_path.read_text(encoding="utf-8"))

    # When: its parameter count is estimated without allocating weights.
    parameters = estimate(architecture)

    # Then: the model is a genuine approximately-3B core, not a small model with padded files.
    assert 2_900_000_000 <= parameters <= 3_100_000_000
