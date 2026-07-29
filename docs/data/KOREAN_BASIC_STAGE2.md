# CERPT 한국어 Basic Stage 2

Stage 2는 긴 evidence trace보다 먼저 한국어 입력을 읽고 짧은 답을 만드는 능력을 확보하는 단계다.

## 포함 데이터

- 한국어 단순 산수: 덧셈, 뺄셈, 곱셈, 나눗셈, 3단계 연산
- 한국어 일상 대화: 인사, 감사, 사과, 도움 요청, 응원, 취침 인사 등
- 총 9,000개: train 7,200 / validation 900 / test 900
- 각 target은 현재 `answer 93`처럼 짧게 유지

```powershell
python scripts/generate_korean_basic.py
```

생성 위치:

- `data/korean_basic/train.jsonl`
- `data/korean_basic/validation.jsonl`
- `data/korean_basic/test.jsonl`
- `data/korean_basic/metadata.json`

## 학습 방향

기존 Stage 1 checkpoint를 그대로 이어 학습하지 않고, 새 tokenizer와 짧은 target을 사용하는 별도 Stage 2 모델로 시작한다.

```powershell
python scripts/train.py `
  --data-dir data/korean_basic `
  --output-dir artifacts/cerpt-korean-basic `
  --epochs 10 `
  --batch-size 32 `
  --max-length 96
```

현재 단계에서 의도적으로 뒤로 미룬 것:

- 긴 evidence trace
- 복잡한 cycle 출력
- 고급 EOS 직렬화
- 답변 파싱기 개선

짧은 산수와 일상 대화가 안정된 뒤, 이 모델에 evidence trace와 verifier 학습을 다시 추가한다.

## 실행 기록

- 10 epoch 완료
- Train loss: `0.3574`
- Validation loss: `0.3771`
- 일상 대화 smoke test는 응답을 생성했지만, 산수 test sample에서 정답 `93` 대신 `51`을 생성했다.
- 따라서 loss는 개선됐지만 산수 정확도가 확보된 상태는 아니며, 다음 단계에서 exact-match 평가와 산수 전용 curriculum 보강이 필요하다.
