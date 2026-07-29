# Decoder-only CERPT Base

## 목적

기존 `CERPTForConditionalGeneration`은 입력과 정답 trace를 나눠 학습하는 Encoder–Decoder PoC다. 따라서 일반적인 Base LLM처럼 다음 토큰을 계속 예측하는 모델과는 다르다.

`CERPTForCausalLM`은 이 문제를 분리하기 위한 새 골격이다.

- decoder-only causal Transformer
- next-token language-model loss
- decoder hidden state에서 CERPT typed workspace 생성
- cycle별 operator prediction
- cycle별 verifier prediction
- workspace summary를 decoder 출력에 다시 주입
- `save_pretrained`/`from_pretrained` 형식 저장

핵심 코드는 [src/cerpt/models/cerpt_causal.py](../../src/cerpt/models/cerpt_causal.py)이고, 학습기는 [scripts/train_causal.py](../../scripts/train_causal.py), 실행기는 [scripts/chat_causal.py](../../scripts/chat_causal.py)다.

## 현재 상태

현재 구현은 “CERPT 구조를 가진 작은 Base 모델을 처음부터 학습할 수 있는 골격”이다. `data/korean_basic_v2`로는 한국어 산수·일상 대화 중심의 smoke pretraining을 할 수 있지만, 이것만으로 범용 LLM이 되었다고 주장할 수 없다.

3B급 확장 목표는 [configs/cerpt-causal-3b.json](../../configs/cerpt-causal-3b.json)에 정의했다. 현재 구조 기준으로 hidden 3072, 32 layers, FFN 8192, 24 heads, 32k vocabulary를 사용하며 약 3.12B parameters다. 파라미터 수만 확인하려면 다음 명령을 실행한다.

```powershell
python scripts/estimate_causal_params.py --config configs/cerpt-causal-3b.json
```

3B 학습은 현재 8GB GPU에서 실행하지 않는다. 실제 학습에는 32k tokenizer, 대규모 corpus, distributed mixed precision, activation checkpointing, sharded optimizer가 필요하다.

대형 학습 서버에서는 다음처럼 preset을 적용한다. 아래 명령은 구조 연결 예시이며, 현재 PC에서 실행하지 않는다.

```powershell
python scripts/train_causal.py `
  --architecture-config configs/cerpt-causal-3b.json `
  --tokenizer-dir artifacts/tokenizer-32k `
  --data-dir data/pretraining_shards `
  --output-dir artifacts/cerpt-causal-3b `
  --batch-size 1 `
  --epochs 1
```

특히 아래 항목은 아직 후속 단계다.

- 실제 KV cache
- distributed/mixed-precision 대규모 사전학습
- packed web/code/multilingual corpus
- vLLM custom model adapter와 attention backend
- 검증기 negative sample 및 causal utility ablation

따라서 지금은 PyTorch에서 구조와 손실을 검증하는 단계이며, vLLM 호환 모델로 공개하는 단계는 아니다.

## 실행

```powershell
python scripts/train_causal.py `
  --data-dir data/korean_basic_v2 `
  --output-dir artifacts/cerpt-causal-korean-base `
  --epochs 3 `
  --batch-size 16 `
  --max-length 128
```

```powershell
python scripts/chat_causal.py `
  --model-dir artifacts/cerpt-causal-korean-base `
  --question "안녕"
```

한국어 산수 입력은 현재 결정론적 verifier가 먼저 계산해 정답을 확인한다. 이는 작은 모델의 생성 품질을 과장하지 않고 CERPT의 검증 경로를 따로 측정하기 위한 것이다.
