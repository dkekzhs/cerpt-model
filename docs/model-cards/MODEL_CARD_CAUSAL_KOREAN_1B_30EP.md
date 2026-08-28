---
language:
- ko
license: other
library_name: transformers
tags:
- cerpt
- causal-lm
- korean
---

# CERPT Causal Korean 1B 30-Epoch

This is the release-card template for the checkpoint produced by `scripts/lightning_1b.sh` or the Colab 1B notebook. The source repository does not claim that the model exists until a completed training run produces `TRAINING_COMPLETE`. The cloud uploader refuses to publish without that marker.

## Architecture

- decoder: Llama-compatible causal decoder
- parameters: 1,020,366,857
- hidden size: 2,048
- layers: 20
- query heads / KV heads: 16 / 4
- FFN: 5,504 with SwiGLU
- context limit: 2,048
- vocabulary: 32,768, including 96 CERPT workspace tokens
- auxiliary outputs: six-cycle operator and verifier predictions

The workspace tokens occur after the prompt and before the response. Prompt and workspace positions are masked from the language-model loss; the response is trained with next-token loss.

## Training data

- project-generated `korean_basic_v6`
- `songys/Chatbot_data` at commit `4cf20d13fc46f5037fd1c531cd566e2dd9f72974` (MIT)
- a user-provided Korean office-dialogue ZIP described as AI-Hub-derived

Exact duplicate prompt/response pairs are removed before a seeded source/task-stratified 80/10/10 split. The prepared corpus contains 62,095 pairs: 49,673 train, 6,207 validation, and 6,215 test. The raw and normalized datasets are not uploaded with the model.

The office archive's redistribution and derived-model terms are not established by this repository. A publisher must verify those terms and explicitly acknowledge them before preparing the data. This unresolved source condition is why the aggregate model license is marked `other`.

## Training recipe

- from scratch
- target epochs: 30
- per-device batch: 1
- gradient accumulation: 32
- optimizer: Adafactor
- precision: FP32 model parameters with FP16 AMP autocast
- gradient checkpointing: enabled
- checkpoint interval: 100 optimizer steps
- automatic resume: latest Hugging Face Trainer `checkpoint-*`

The published repository should include `TRAINING_COMPLETE` and `trainer_state.json`. Those files, not this template, are the source of the actual completed epoch, global step, loss, and runtime records.

## Intended use and limitations

This is a research checkpoint for Korean conversational adaptation and CERPT workspace experiments. Repeating 62,095 prompt/response pairs for 30 epochs is not sufficient general-language pretraining for a 1B model. Memorization, source imbalance, synthetic phrasing, unsafe answers, and severe topic gaps are expected. It must not be described as unbiased, generally knowledgeable, or production-safe without separate held-out evaluations.

The model is not validated for medical, legal, financial, safety-critical, or autonomous decision-making. vLLM registration and Ollama GGUF conversion remain separate engineering work.
