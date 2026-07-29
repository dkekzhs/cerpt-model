---
library_name: transformers
pipeline_tag: text2text-generation
tags:
- pytorch
- transformers
- recursive-transformer
- causal-evidence
- synthetic-reasoning
---

# CERPT — Causal Evidence Recursive Program Transformer

## What is CERPT?

CERPT is a recursive reasoning architecture. It is designed to preserve
intermediate reasoning state in a persistent typed workspace instead of
leaving all intermediate computation in an undifferentiated residual stream.

At each reasoning cycle CERPT:

1. selects an operator program,
2. reads typed state and input evidence,
3. proposes a state transition,
4. verifies the proposed evidence and expected effect, and
5. commits, rolls back, or branches the transition.

The central research question is whether persistent, verifiable intermediate
state produces useful additional computation rather than merely polishing
confidence.

CERPT is intended to be a modality-agnostic framework rather than a single
fixed general-purpose model. A shared CERPT core can be paired with task
adapters, NPC-specific persona/memory, and image/video evidence encoders.
This makes it possible to build many small specialized agents without
duplicating the entire base model for every agent.

## Current checkpoint

The current repository contains a small text-only Phase-1 proof of concept.
It was trained on synthetic algorithmic tasks including arithmetic chains,
variable binding, graph traversal, and ordering constraints. The latest
locally confirmed training checkpoint was stopped at 71 epochs.

## What this model is not

The currently uploaded checkpoint is not a general-purpose pretrained LLM,
Korean word-chain expert, or multimodal image/video model. It has no
large-scale web or code pretraining and should not be evaluated as an
open-domain assistant. The repository describes the extensible CERPT
architecture; individual checkpoints may support different tasks and
modalities depending on their training data and adapters.

## Multimodal status

The repository now includes a Hugging Face CLIP vision front-end and a
temporal video front-end. They convert image/video evidence into tokens that
are merged into the CERPT typed workspace. The current text checkpoint has not
been trained on image/video question-answer data, so connecting the encoders
does not by itself provide reliable visual answers.

## Planned extensions

- large-scale natural-language and code pretraining
- Korean end-to-end word-chain data with a deterministic dictionary checker
- typed visual-object and temporal-event workspace slots
- image/video evidence training and multimodal fusion evaluation
- instruction tuning and adversarial causal-evidence evaluation
- shared-base plus per-agent adapters for large populations of small NPCs

## Intended use

Architecture research, synthetic reasoning experiments, state-transition
ablation studies, and development of evidence verification methods.
