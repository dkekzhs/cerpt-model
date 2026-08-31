# /// script
# requires-python = ">=3.10"
# dependencies = ["accelerate==1.14.0", "pydantic==2.11.4", "torch==2.11.0", "transformers==5.7.0"]
# ///
# How to run: uv run --project . python scripts/train_causal_cloud.py --help

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import assert_never

import torch
from pydantic import BaseModel, ConfigDict
from transformers import PreTrainedTokenizerFast, Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from cerpt.data.causal import CausalBatchFormatter, add_workspace_tokens
from cerpt.models.cerpt_causal import CERPTCausalConfig, CERPTForCausalLM
from cerpt.training.checkpoint import latest_resumable_checkpoint
from cerpt.utils.device import select_device
from scripts.train_causal import JsonlDataset, make_collator


class CloudProfile(StrEnum):
    COLAB_T4 = "colab-t4"
    LIGHTNING_T4 = "lightning-t4"
    MAC_MPS = "mac-mps"
    CPU_SMOKE = "cpu-smoke"


class ArchitecturePreset(BaseModel):
    model_config = ConfigDict(frozen=True)

    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    max_position_embeddings: int
    workspace_slots: int
    num_cycles: int
    num_operators: int
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False


@dataclass(frozen=True, slots=True)
class ProfileSettings:
    parameter_dtype: torch.dtype
    fp16: bool
    gradient_checkpointing: bool
    optimizer: str
    minimum_gpu_memory_gib: float


@dataclass(frozen=True, slots=True)
class CloudTrainingRequest:
    data_dir: Path
    tokenizer_dir: Path
    output_dir: Path
    architecture_config: Path
    profile: CloudProfile
    epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    save_steps: int
    max_steps: int
    seed: int


@dataclass(frozen=True, slots=True)
class CudaRequiredError(RuntimeError):
    profile: CloudProfile

    def __str__(self) -> str:
        return f"profile {self.profile.value} requires an NVIDIA CUDA GPU"


@dataclass(frozen=True, slots=True)
class InsufficientGpuMemoryError(RuntimeError):
    required_gib: float
    actual_gib: float

    def __str__(self) -> str:
        return f"GPU has {self.actual_gib:.2f} GiB; this profile requires at least {self.required_gib:.2f} GiB"


@dataclass(frozen=True, slots=True)
class TokenizerArchitectureMismatchError(ValueError):
    expected: int
    actual: int

    def __str__(self) -> str:
        return f"architecture expects {self.expected} tokens after workspace reservation; found {self.actual}"


def _profile_settings(profile: CloudProfile) -> ProfileSettings:
    match profile:
        case CloudProfile.COLAB_T4:
            return ProfileSettings(torch.float32, True, True, "adafactor", 14.5)
        case CloudProfile.LIGHTNING_T4:
            return ProfileSettings(torch.float32, True, True, "adafactor", 14.5)
        case CloudProfile.MAC_MPS:
            return ProfileSettings(torch.float32, False, True, "adafactor", 0.0)
        case CloudProfile.CPU_SMOKE:
            return ProfileSettings(torch.float32, False, False, "adamw_torch", 0.0)
        case unreachable:
            assert_never(unreachable)


def _preflight(profile: CloudProfile, settings: ProfileSettings) -> None:
    match profile:
        case CloudProfile.CPU_SMOKE:
            return
        case CloudProfile.MAC_MPS:
            select_device("mps")
        case CloudProfile.COLAB_T4 | CloudProfile.LIGHTNING_T4:
            if not torch.cuda.is_available():
                raise CudaRequiredError(profile)
            actual_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if actual_gib < settings.minimum_gpu_memory_gib:
                raise InsufficientGpuMemoryError(settings.minimum_gpu_memory_gib, actual_gib)
        case unreachable:
            assert_never(unreachable)


def _create_model(
    preset: ArchitecturePreset,
    tokenizer: PreTrainedTokenizerFast,
    settings: ProfileSettings,
) -> CERPTForCausalLM:
    workspace_token_ids = add_workspace_tokens(tokenizer, preset.num_cycles, preset.workspace_slots)
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(settings.parameter_dtype)
    try:
        config = CERPTCausalConfig(
            vocab_size=len(tokenizer),
            hidden_size=preset.hidden_size,
            num_hidden_layers=preset.num_hidden_layers,
            num_attention_heads=preset.num_attention_heads,
            num_key_value_heads=preset.num_key_value_heads,
            intermediate_size=preset.intermediate_size,
            max_position_embeddings=preset.max_position_embeddings,
            workspace_slots=preset.workspace_slots,
            num_cycles=preset.num_cycles,
            num_operators=preset.num_operators,
            workspace_token_ids=workspace_token_ids,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            hidden_act=preset.hidden_act,
            rms_norm_eps=preset.rms_norm_eps,
            tie_word_embeddings=preset.tie_word_embeddings,
            use_cache=False,
        )
        return CERPTForCausalLM(config)
    finally:
        torch.set_default_dtype(original_dtype)


def _training_arguments(
    request: CloudTrainingRequest,
    settings: ProfileSettings,
) -> TrainingArguments:
    match request.profile:
        case CloudProfile.COLAB_T4:
            save_total_limit = 1
        case CloudProfile.CPU_SMOKE | CloudProfile.LIGHTNING_T4 | CloudProfile.MAC_MPS:
            save_total_limit = 2
        case unreachable:
            assert_never(unreachable)
    return TrainingArguments(
        output_dir=str(request.output_dir),
        num_train_epochs=request.epochs,
        max_steps=request.max_steps,
        per_device_train_batch_size=request.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=request.gradient_accumulation_steps,
        learning_rate=request.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,
        optim=settings.optimizer,
        fp16=settings.fp16,
        gradient_checkpointing=settings.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=request.save_steps,
        save_total_limit=save_total_limit,
        logging_strategy="steps",
        logging_steps=10,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=0,
        seed=request.seed,
        data_seed=request.seed,
    )


def _parse_request() -> CloudTrainingRequest:
    parser = argparse.ArgumentParser(description="Run resumable causal CERPT training on free cloud GPUs")
    parser.add_argument("--data-dir", type=Path, default=Path("data/korean_conversations_v7"))
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("artifacts/tokenizers/cerpt-korean-32k"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/cerpt-causal-korean-v7-1b"))
    parser.add_argument("--architecture-config", type=Path, default=Path("configs/cerpt-causal-1b.json"))
    parser.add_argument("--profile", type=CloudProfile, choices=list(CloudProfile), default=CloudProfile.LIGHTNING_T4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return CloudTrainingRequest(
        data_dir=args.data_dir,
        tokenizer_dir=args.tokenizer_dir,
        output_dir=args.output_dir,
        architecture_config=args.architecture_config,
        profile=args.profile,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        max_steps=args.max_steps,
        seed=args.seed,
    )


def main() -> None:
    request = _parse_request()
    settings = _profile_settings(request.profile)
    _preflight(request.profile, settings)
    preset = ArchitecturePreset.model_validate_json(request.architecture_config.read_text(encoding="utf-8"))
    tokenizer = PreTrainedTokenizerFast.from_pretrained(request.tokenizer_dir)
    workspace_ids = add_workspace_tokens(tokenizer, preset.num_cycles, preset.workspace_slots)
    if len(tokenizer) != preset.vocab_size:
        raise TokenizerArchitectureMismatchError(preset.vocab_size, len(tokenizer))
    request.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(request.output_dir / "tokenizer")
    checkpoint = latest_resumable_checkpoint(request.output_dir)
    model = (
        CERPTForCausalLM.from_pretrained(checkpoint, dtype=settings.parameter_dtype)
        if checkpoint is not None
        else _create_model(preset, tokenizer, settings)
    )
    model.config.use_cache = False
    formatter = CausalBatchFormatter(tokenizer, tuple(workspace_ids), preset.max_position_embeddings)
    trainer = Trainer(
        model=model,
        args=_training_arguments(request, settings),
        train_dataset=JsonlDataset(request.data_dir / "train.jsonl"),
        eval_dataset=JsonlDataset(request.data_dir / "validation.jsonl"),
        data_collator=make_collator(formatter, preset.num_cycles),
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=checkpoint)
    model.config.use_cache = True
    final_dir = request.output_dir / "final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    trainer.save_state()
    completion = {
        "epochs_requested": request.epochs,
        "epoch": trainer.state.epoch,
        "global_step": trainer.state.global_step,
        "resumed_from": checkpoint,
    }
    (request.output_dir / "TRAINING_COMPLETE").write_text(
        json.dumps(completion, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(completion, ensure_ascii=False))


if __name__ == "__main__":
    main()
