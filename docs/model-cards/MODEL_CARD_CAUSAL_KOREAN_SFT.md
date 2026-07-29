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

# CERPT Korean Causal SFT

This checkpoint is a small decoder-only CERPT research model. It was continued
from the Korean CERPT causal checkpoint and supervised fine-tuned on synthetic
Korean daily-chat responses.

## Architecture

- CERPT decoder-only causal Transformer
- hidden size: 64
- decoder layers: 1
- CERPT workspace, operator head, and verifier head included
- PyTorch and Transformers `save_pretrained` format

## Training

- Base checkpoint: `cerpt-causal-korean-v5-10`
- SFT data: `data/korean_basic_v6`
- SFT epochs: 5
- SFT validation loss: 0.1099
- Korean arithmetic is handled by the deterministic CERPT verifier in the CLI

## Limitations

This is not a general-purpose LLM and is not a 3B model. It can produce short
Korean responses but may mix intents because of its very small capacity and
synthetic training data. It has not been trained for image, video, code, or
open-domain factual QA.

Use the custom CERPT class rather than assuming vLLM/Ollama compatibility:

```python
from cerpt.models.cerpt_causal import CERPTForCausalLM
model = CERPTForCausalLM.from_pretrained("<repo-id>")
```
