# CERPT Decoder-only SFT

`train_causal.py`는 입력과 정답을 모두 next-token pretraining loss로 학습한다. 그 다음 단계인 `sft_causal.py`는 질문 prompt 부분을 `-100`으로 마스킹하고 응답 부분만 supervised fine-tuning한다.

```powershell
python scripts/sft_causal.py `
  --resume-from artifacts/cerpt-causal-korean-v5-10 `
  --data-dir data/korean_basic_v6 `
  --output-dir artifacts/cerpt-causal-korean-v6-sft `
  --epochs 5 `
  --batch-size 64 `
  --max-length 96
```

현재 실행 결과는 `artifacts/cerpt-causal-korean-v6-sft`에 있다. SFT loss는 감소했지만, hidden 64·1 layer 모델은 여러 채팅 intent를 안정적으로 분리하지 못한다. 따라서 다음 단계에서는 모델 크기와 대화 데이터 다양성을 함께 늘려야 한다.
