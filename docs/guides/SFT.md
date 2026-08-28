# CERPT Decoder-only SFT

새 `train_causal.py`와 `sft_causal.py`는 모두 `[BOS] + prompt + workspace tokens + response + [EOS]` 형식을 사용한다. 질문과 workspace 위치는 `-100`으로 마스킹하고 응답만 next-token loss로 학습한다. 차이는 전자는 새 model/tokenizer를 처음 만들고 operator/verifier 보조 label도 사용하며, 후자는 이미 학습한 새 구조 checkpoint를 이어 학습한다는 점이다.

```powershell
python scripts/sft_causal.py `
  --resume-from artifacts/cerpt-causal-korean-v7-base `
  --data-dir data/korean_basic_v6 `
  --output-dir artifacts/cerpt-causal-korean-v7-sft `
  --epochs 5 `
  --batch-size 64 `
  --max-length 96
```

수정 전 `cerpt-causal-korean-v5-10`과 `v6-sft`는 future-token leakage가 있던 구조로 학습되어 resume 대상으로 사용할 수 없다. 위 경로의 v7 이름은 새 학습 산출물 예시이며 아직 결과가 존재한다는 뜻이 아니다.
