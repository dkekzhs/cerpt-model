# 한국어 Stage 1 진행 기록

최종 갱신: 2026-07-28

## 단계별 상태

| 단계 | 상태 | 기록 |
| --- | --- | --- |
| 1. 소형 데이터셋 선정 | 완료 | OpenMathReasoning-mini-ko, HRMCR, Ko-StrategyQA를 라이선스와 구조 확인 후 선정 |
| 2. 원본 다운로드 | 완료 | `data/korean_stage1/raw/`에 Hugging Face 원본 파일 보관 |
| 3. CERPT 형식 변환 | 완료 | 총 21,642개를 80/10/10으로 분할하고 4-cycle trace 생성 |
| 4. 형식·모델 호환성 검사 | 완료 | JSONL 필드, cycle 길이, CERPT forward loss 계산 확인 |
| 5. 실제 전체 학습 | 대기 | GPU가 있는 PC에서 실행할 단계. 현재 작업 PC는 CUDA를 사용할 수 없음 |

## 현재 결과

- Train: 17,313개
- Validation: 2,164개
- Test: 2,165개
- Task: 수학 19,252개, 다중 홉 논리 2,290개, 알고리즘 논리 100개
- Operator: `EXTRACT`, `SIMULATE` 또는 `BIND`, `CHECK`, `WRITE_RESULT`
- 고정 seed: `42`

## 검증 결과

두 개의 한국어 샘플을 tokenizer와 CERPT forward에 넣었고, 출력은 다음 형태로 정상 생성되었다.

```text
rows=2, vocab=107, logits=[2, 128, 107], loss 계산 성공
```

이 검사는 학습이 끝났다는 뜻이 아니라, 데이터셋과 모델의 입출력 계약이 맞는다는 뜻이다.

## 다음 실행

GPU PC에서 다음 명령으로 1 epoch smoke run을 먼저 실행한다.

```powershell
pip install -e .
python scripts/train.py `
  --data-dir data/korean_stage1 `
  --output-dir artifacts/cerpt-korean-stage1 `
  --epochs 1 `
  --batch-size 16 `
  --max-length 256
```

그 후 validation loss와 `scripts/chat.py`의 한국어 수학·논리 질문 결과를 확인하고, 필요하면 epoch와 hidden size를 조정한다.

## 실제 학습 기록

- 2026-07-28: CPU에서 1~3 epoch 완료
- Epoch 3 validation loss: `2.1737`
- 실행 도구의 foreground 제한으로 중단된 뒤, Epoch 3 checkpoint에서 `--resume-from`으로 4~30 epoch를 백그라운드 재개
- 백그라운드 로그: `artifacts/cerpt-korean-stage1/training_resume.log`
