---
library_name: transformers
pipeline_tag: text-generation
tags:
- pytorch
- transformers
- causal-lm
- korean
- cerpt
---

# CERPT Korean Causal Base Smoke Checkpoint

This is a small decoder-only CERPT causal-language-model checkpoint trained
from scratch on the synthetic Korean arithmetic and daily-chat curriculum.

## Architecture

- CERPT decoder-only causal Transformer
- hidden size: 64
- decoder layers: 1
- CERPT workspace, operator head, and verifier head included

## Training

- data: `data/korean_basic_v5`
- 10 pretraining epochs
- validation loss: 1.6539
- arithmetic questions are additionally checked by the deterministic CERPT verifier

## Limitations

This is a research smoke checkpoint, not a general-purpose LLM, not a 3B
model, and not a multimodal checkpoint. It has limited Korean chat quality and
has not received large-scale web, code, image, or video pretraining.
