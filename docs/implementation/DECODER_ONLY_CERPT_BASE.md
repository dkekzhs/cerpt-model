# Decoder-only CERPT Base

## 현재 구현

`CERPTForCausalLM`은 이제 Hugging Face `LlamaForCausalLM`과 같은 decoder core를 사용한다.

- RoPE positional encoding
- RMSNorm
- SwiGLU MLP
- grouped-query attention(GQA)
- 표준 autoregressive causal mask
- Hugging Face KV cache와 generation
- response-only causal LM loss
- 인과적 in-stream CERPT workspace token
- cycle별 operator/verifier 보조 head
- `save_pretrained`/`from_pretrained`

핵심 코드는 [src/cerpt/models/cerpt_causal.py](../../src/cerpt/models/cerpt_causal.py), sequence formatter는 [src/cerpt/data/causal.py](../../src/cerpt/data/causal.py), 학습기는 [scripts/train_causal.py](../../scripts/train_causal.py), 실행기는 [scripts/chat_causal.py](../../scripts/chat_causal.py)다.

## 왜 구조를 바꿨나

이전 구현은 causal decoder 뒤에서 전체 sequence hidden을 평균내 workspace를 만들고 그 결과를 모든 token 위치에 더했다. 그 때문에 미래 response와 masked padding이 과거 logit을 바꿨다. 누수 전 decoder hidden은 정상적이었으므로 문제는 workspace 결합에 있었다.

수정 전 재현값:

```text
decoder before workspace: max prefix hidden difference 0.000000
random initialization: max prefix-logit difference 0.001315
trained v6 SFT checkpoint: max prefix-logit difference 3.535420
same real tokens, padding length 4 → 8: max real-token logit difference 6.008883
```

따라서 dense global workspace와 shared transition core를 causal 모델에서 제거했다. 수정 전 base/SFT checkpoint와 기록 loss는 폐기 대상이며 새 구조로 처음부터 학습해야 한다.

## 새 causal workspace

학습 sequence는 다음처럼 구성한다.

```text
[BOS]
prompt tokens
workspace cycle 0 / slot 0 ... slot N
workspace cycle 1 / slot 0 ... slot N
...
response tokens
[EOS]
```

workspace는 별도 full-sequence tensor를 만들지 않는다. 각각의 workspace token은 같은 causal decoder를 통과하면서 앞선 prompt와 workspace token만 볼 수 있다. response token은 모든 workspace token을 볼 수 있으므로 workspace가 생성에 실제 memory로 참여한다. 학습 loss는 prompt와 workspace 위치를 `-100`으로 mask하고 response와 EOS에만 적용한다.

모델은 workspace token의 최종 hidden을 `[batch, cycle, slot, hidden]`으로 추출한다. cycle 안의 slot 평균에서 operator logits와 validity logits를 계산한다. 이 head들은 현재 보조 감독과 관찰용이다. operator가 다른 함수를 실행하거나 verifier가 state를 commit/rollback하는 단계는 아직 구현되지 않았다.

## 회귀 계약

[tests/test_causal.py](../../tests/test_causal.py)와 [tests/test_causal_data.py](../../tests/test_causal_data.py)가 다음 경계를 고정한다.

- 미래 suffix를 바꿔도 동일 prefix logits는 변하지 않는다.
- masked right padding을 추가해도 real-token logits는 변하지 않는다.
- KV cache로 한 token씩 계산한 logit과 full-prefix logit이 일치한다.
- workspace token은 prompt 뒤, response 앞에 빠짐없이 순서대로 배치된다.
- prompt와 workspace에는 LM loss를 주지 않고 response만 학습한다.
- 긴 prompt는 앞부분을 버리되 모든 workspace token을 보존한다.

## 무료 T4용 1B preset

[configs/cerpt-causal-1b.json](../../configs/cerpt-causal-1b.json)은 hidden 2,048, 20 layers, 16 query / 4 KV heads, SwiGLU 5,504, context 2,048로 정확히 1,020,366,857 parameters다. T4 profile은 모델 파라미터와 gradient를 FP32로 유지하고 FP16 autocast, Adafactor, gradient checkpointing, batch 1을 사용한다. FP32 weight와 gradient만 약 7.60GiB다. 이 경로는 [Colab 1B 노트북](../../notebooks/CERPT_1B_Colab_Training.ipynb)과 [Lightning 1B 런처](../../scripts/lightning_1b.sh)에 연결되어 있다.

기존 3B 모델을 직접 FP16 파라미터로 만들고 `TrainingArguments(fp16=True)`를 함께 사용한 경로는 GradScaler가 FP16 gradient를 unscale할 수 없어 첫 optimizer step에서 실패했다. 반대로 3B 파라미터를 정상 FP32로 유지하면 weight와 gradient만 약 22.50GiB라 14.5GiB T4에 들어가지 않는다.

## 3B preset

[configs/cerpt-causal-3b.json](../../configs/cerpt-causal-3b.json)은 다음 구조다.

| 항목 | 값 |
|---|---:|
| vocabulary | 32,768, workspace token 96개 포함 |
| hidden | 3,072 |
| decoder layers | 28 |
| query heads | 24 |
| KV heads | 8 |
| head dimension | 128 |
| SwiGLU intermediate | 8,192 |
| context | 4,096 |
| workspace | 6 cycles × 16 slots |
| embeddings | input/output untied |
| parameters | 3,020,101,641 |
| FP16 weights | 약 5.63 GiB |

```powershell
python scripts/estimate_causal_params.py --config configs/cerpt-causal-3b.json
```

이 수치는 실제 Llama/GQA projection shape, SwiGLU 3개 projection, RMSNorm, LM head, CERPT auxiliary head를 센 값이다. 3B weight가 이미 학습되었다는 뜻은 아니다.

## 새 학습 데이터 경계

현재 확인한 후보 원천은 다음과 같다.

| 원천 | 규모 | 비고 |
|---|---:|---|
| `data/korean_basic_v6` | 11,000 | 산수 8,000 + 일상 대화 3,000 |
| `songys/Chatbot_data` | 11,876 | 일상/이별/사랑 인공 문답, MIT 저장소 |
| 로컬 오피스 대화 ZIP | 46,414 | daily/email/schedule/meeting, 4개 JSON |
| 합계 | 69,290 | 중복 제거 전 |

이 데이터는 대화 특화 학습과 pipeline 검증에는 쓸 수 있지만 3B 범용 base를 충분히 학습시키는 corpus는 아니다. 특히 Chatbot_data는 감정 도메인 분포가 명시되어 있고 오피스 데이터는 업무 도메인에 집중되어 있어, “편향이 없다”고 가정하면 안 된다. source별 sampling weight, 중복/near-duplicate 제거, train/validation/test의 prompt-group 분리, 개인정보 검사, 독성·성별·지역·연령 평가가 먼저 필요하다. 로컬 ZIP은 재배포 전에 원출처와 라이선스를 확정한다.

## 학습과 서빙

소형 구조 smoke 학습:

```powershell
python scripts/train_causal.py `
  --data-dir data/korean_basic_v6 `
  --output-dir artifacts/cerpt-causal-korean-v7-base `
  --epochs 1 `
  --batch-size 16 `
  --max-length 128
```

3B 학습 서버용 구조 연결 예시:

```powershell
python scripts/train_causal.py `
  --architecture-config configs/cerpt-causal-3b.json `
  --tokenizer-dir artifacts/tokenizer-32k `
  --data-dir data/pretraining_shards `
  --output-dir artifacts/cerpt-causal-3b `
  --batch-size 1 `
  --epochs 1 `
  --gradient-checkpointing
```

32k tokenizer에는 96개 workspace special token이 최종 vocabulary 안에 예약되어 있어야 한다. `train_causal.py`가 token을 등록한 뒤 tokenizer 크기와 preset의 `vocab_size`가 다르면 학습을 중단한다.

KV cache는 구현되었고 core parameter naming은 Llama 계열을 따른다. 남은 production serving 작업은 `cerpt_causal` config 등록/패키징, vLLM 실제 load·continuous batching 검증, GGUF converter와 Ollama/llama.cpp template, FlashAttention, FSDP/DeepSpeed다.
