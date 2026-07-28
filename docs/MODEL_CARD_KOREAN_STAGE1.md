---
language:
- ko
library_name: transformers
pipeline_tag: text-generation
tags:
- pytorch
- cerpt
- recursive-reasoning
- korean
- evidence-verification
---

# CERPT Korean Stage 1

CERPT (Causal Evidence Recursive Program Transformer) is a research
architecture that keeps typed intermediate state in a workspace and updates
that state through explicit operators and evidence checks.

This checkpoint is a small Korean Stage 1 experiment. It is trained from
scratch on Korean math, logic, and multi-hop reasoning examples converted to
four cycles:

`EXTRACT → SIMULATE/BIND → CHECK → WRITE_RESULT`

## Status

This is not a general-purpose LLM. It is a research checkpoint for testing
whether a small CERPT model can learn Korean task solving and structured
reasoning traces. It should not be used as an open-domain assistant or for
high-stakes decisions.

## Training data

- [OpenMathReasoning-mini-ko](https://huggingface.co/datasets/neuralfoundry-coder/OpenMathReasoning-mini-ko), CC-BY-4.0
- [HRMCR](https://huggingface.co/datasets/HAERAE-HUB/HRMCR), Apache-2.0
- [Ko-StrategyQA](https://huggingface.co/datasets/NomaDamas/Ko-StrategyQA), Apache-2.0

The local conversion and split details are documented in
`docs/KOREAN_STAGE1_DATA.md`. Dataset providers and licenses must be credited
when redistributing derived artifacts.

## Architecture

- PyTorch and Hugging Face compatible `CERPTForConditionalGeneration`
- 4 recursive workspace cycles
- 6 typed operators: `EXTRACT`, `BIND`, `COMPARE`, `SIMULATE`, `CHECK`,
  `WRITE_RESULT`
- Korean tokenizer trained from the Stage 1 text

## Limitations

- No general web or code pretraining
- No reliable image/video understanding training in this checkpoint
- Structured output quality and Korean open-domain ability are not established
- Reported loss is a training diagnostic, not a general capability score
