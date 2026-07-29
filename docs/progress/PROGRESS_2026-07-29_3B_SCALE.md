# 진행 기록 — 2026-07-29 3B 규모 설계

## 추가한 내용

- `configs/cerpt-causal-3b.json` 추가
- `scripts/estimate_causal_params.py` 추가
- `scripts/train_causal.py --architecture-config` 지원
- 기존 tokenizer 대신 `--tokenizer-dir`로 대형 tokenizer를 연결할 수 있도록 수정

## 목표 구조

| 항목 | 값 |
|---|---:|
| hidden size | 3,072 |
| decoder layers | 32 |
| attention heads | 24 |
| FFN intermediate | 8,192 |
| vocabulary | 32,768 |
| context | 4,096 |
| parameter estimate | 3,121,827,849 |
| FP16 weight only | 약 5.81 GiB |

파라미터 산정은 다음으로 재현할 수 있다.

```powershell
python scripts/estimate_causal_params.py --config configs/cerpt-causal-3b.json
```

## 현재 판단

이 구조는 3B급 설정으로 확장 가능하지만, 현재 8GB GPU에서 실제 학습을 시작하지 않았다. FP16 weight만 약 5.81GiB이며 optimizer, gradient, activation, CUDA workspace가 추가된다. 실제 학습에는 멀티 GPU, mixed precision, activation checkpointing, sharded optimizer/checkpoint가 필요하다.

또한 현재 CERPT causal 구현은 KV cache와 vLLM backend가 없으므로, 3B weight를 만든 뒤 바로 vLLM/Ollama 모델로 사용할 수 있는 상태도 아니다.
