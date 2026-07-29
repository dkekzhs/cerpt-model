---
language:
- ko
license: other
task_categories:
- question-answering
- text-generation
tags:
- korean
- reasoning
- cerpt
size_categories:
- 10K<n<100K
---

# CERPT Korean Stage 1 Dataset

This is a CERPT-compatible Korean reasoning mixture containing math, logic,
and multi-hop reasoning examples. Each record contains `input_text`,
`target_text`, `answer`, operator labels, validity labels, and a four-cycle
trace.

The data was converted from these sources; their original licenses remain
applicable and attribution is required:

- [OpenMathReasoning-mini-ko](https://huggingface.co/datasets/neuralfoundry-coder/OpenMathReasoning-mini-ko) — CC-BY-4.0
- [HRMCR](https://huggingface.co/datasets/HAERAE-HUB/HRMCR) — Apache-2.0
- [Ko-StrategyQA](https://huggingface.co/datasets/NomaDamas/Ko-StrategyQA) — Apache-2.0

The `raw/` directory is intentionally not redistributed here. The published
files are the CERPT conversion and deterministic train/validation/test split.
This is a research dataset, not a general-purpose Korean language corpus.
