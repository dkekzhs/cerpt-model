import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cerpt.data import CausalTokenizerSpec
from cerpt.data import tokenizer as tokenizer_module
from cerpt.data.tokenizer import build_tokenizer


def test_formatter_places_causal_workspace_between_prompt_and_response():
    # Given: a tokenizer and a two-cycle, two-slot CERPT workspace.
    from cerpt.data.causal import CausalBatchFormatter, add_workspace_tokens

    tokenizer = build_tokenizer(["질문 하나", "대답 둘"])
    workspace_ids = add_workspace_tokens(tokenizer, num_cycles=2, workspace_slots=2)
    formatter = CausalBatchFormatter(tokenizer, tuple(workspace_ids), max_length=32)

    # When: a conversation pair is formatted for causal training.
    batch = formatter.collate([{"input_text": "질문 하나", "target_text": "대답 둘"}])

    # Then: prompt and workspace are context-only while response tokens carry loss.
    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]
    workspace_positions = [int(input_ids.eq(token_id).nonzero()[0]) for token_id in workspace_ids]
    first_target = int(labels.ne(-100).nonzero()[0])
    assert workspace_positions == list(range(workspace_positions[0], workspace_positions[0] + 4))
    assert workspace_positions[-1] + 1 == first_target
    assert labels[:first_target].eq(-100).all()
    assert batch["attention_mask"].dtype == torch.long


def test_prompt_formatting_keeps_workspace_tokens_at_the_causal_boundary():
    # Given: a prompt formatter whose context must be truncated.
    from cerpt.data.causal import CausalBatchFormatter, add_workspace_tokens

    tokenizer = build_tokenizer(["가 나 다 라 마 바 사 아 자 차 카 타 파 하"])
    workspace_ids = add_workspace_tokens(tokenizer, num_cycles=2, workspace_slots=2)
    formatter = CausalBatchFormatter(tokenizer, tuple(workspace_ids), max_length=8)

    # When: an overlong inference prompt is encoded.
    batch = formatter.encode_prompts(["가 나 다 라 마 바 사 아 자 차 카 타 파 하", "가"])

    # Then: the newest prompt context and every workspace token remain present.
    input_ids = batch["input_ids"]
    assert input_ids.shape == (2, 8)
    assert input_ids[:, -4:].tolist() == [workspace_ids, workspace_ids]
    assert batch["attention_mask"][1, :2].eq(0).all()


def test_unannotated_conversation_rows_receive_ignored_auxiliary_targets():
    # Given: an external Korean dialogue row without CERPT trace annotations.
    from cerpt.data.causal import CausalBatchFormatter, add_workspace_tokens
    from scripts.train_causal import make_collator

    tokenizer = build_tokenizer(["질문", "응답"])
    workspace_ids = add_workspace_tokens(tokenizer, num_cycles=2, workspace_slots=2)
    formatter = CausalBatchFormatter(tokenizer, tuple(workspace_ids), max_length=16)

    # When: the base-training collator receives that external row.
    batch = make_collator(formatter, num_cycles=2)([{"input_text": "질문", "target_text": "응답"}])

    # Then: language targets remain active and unavailable auxiliary labels are ignored.
    assert batch["labels"].ne(-100).any()
    assert batch["operator_labels"].eq(-100).all()
    assert batch["cycle_valid_labels"].eq(-100).all()


def test_causal_bpe_tokenizer_reserves_exact_workspace_vocabulary() -> None:
    # Given: a small multilingual corpus and four future workspace tokens.
    texts = [f"한국어 대화 문장 {index} English sample {index}" for index in range(300)]

    # When: a tokenizer is trained for a fixed final vocabulary size.
    tokenizer = tokenizer_module.build_causal_bpe_tokenizer(
        texts,
        CausalTokenizerSpec(final_vocab_size=512, workspace_tokens=4, model_max_length=256),
    )
    from cerpt.data.causal import add_workspace_tokens

    add_workspace_tokens(tokenizer, num_cycles=2, workspace_slots=2)

    # Then: learned and reserved tokens together match the architecture exactly.
    assert len(tokenizer) == 512
    assert tokenizer.decode(tokenizer.encode("한국어 대화", add_special_tokens=False)) == "한국어 대화"


def test_causal_bpe_tokenizer_reaches_fixed_vocabulary_with_single_occurrence_merges() -> None:
    # Given: a corpus whose rare Korean syllable merges each occur only once.
    texts = ["".join(chr(0xAC00 + index) for index in range(400))]

    # When: a tokenizer is trained against the architecture's fixed vocabulary contract.
    tokenizer = tokenizer_module.build_causal_bpe_tokenizer(
        texts,
        CausalTokenizerSpec(final_vocab_size=512, workspace_tokens=4, model_max_length=256),
    )

    # Then: rare but valid merges still fill the learned portion exactly.
    assert len(tokenizer) == 508
