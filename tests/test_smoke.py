import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cerpt.data.synthetic import generate_dataset
from cerpt.data.tokenizer import build_tokenizer
from cerpt.models.cerpt import CERPTConfig, CERPTForConditionalGeneration
from cerpt.models.multimodal import CERPTMultimodalConfig, CERPTMultimodalForConditionalGeneration


def test_synthetic_examples_have_trace_and_answer():
    rows = generate_dataset(8, seed=7)
    assert len(rows) == 8
    assert all(row["trace"] and row["answer"] for row in rows)


def test_model_forward_and_generation():
    rows = generate_dataset(2, seed=9)
    tokenizer = build_tokenizer([x for row in rows for x in (row["input_text"], row["target_text"])])
    encoded = tokenizer([row["input_text"] for row in rows], padding=True, return_tensors="pt")
    model = CERPTForConditionalGeneration(CERPTConfig(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_attention_heads=4, workspace_slots=4, num_cycles=2, max_position_embeddings=64))
    labels = tokenizer([row["target_text"] for row in rows], padding=True, return_tensors="pt")["input_ids"]
    labels = labels.masked_fill(labels.eq(tokenizer.pad_token_id), -100)
    output = model(**encoded, labels=labels, cycle_valid_labels=torch.ones(2, 2), operator_labels=torch.zeros(2, 2, dtype=torch.long))
    assert output.loss is not None and output.logits.shape[0] == 2
    generated = model.generate(**encoded, max_new_tokens=4)
    assert generated.shape[0] == 2


def test_multimodal_bridge_accepts_image_tokens_without_network():
    rows = generate_dataset(1, seed=11)
    tokenizer = build_tokenizer([x for row in rows for x in (row["input_text"], row["target_text"])])
    encoded = tokenizer([rows[0]["input_text"]], return_tensors="pt")
    config = CERPTMultimodalConfig(
        cerpt_config=CERPTConfig(
            vocab_size=len(tokenizer),
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=4,
            workspace_slots=4,
            num_cycles=2,
            max_position_embeddings=64,
        ).to_dict(),
        vision_config={
            **{
                "hidden_size": 32,
                "intermediate_size": 64,
                "projection_dim": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "image_size": 8,
                "patch_size": 4,
            },
            "model_type": "clip_vision_model",
        },
    )
    model = CERPTMultimodalForConditionalGeneration(config)
    output = model(**encoded, pixel_values=torch.randn(1, 3, 8, 8))
    assert output.logits.shape[0] == 1
    assert output.workspace.shape == (1, 4, 32)
    video_output = model(**encoded, pixel_values=torch.randn(1, 2, 3, 8, 8))
    assert video_output.logits.shape[0] == 1
