# 진행 기록 — 2026-07-29 데이터 품질·동작 검증

## 데이터 감사

검사 명령:

```powershell
python scripts/audit_korean_basic.py --data-dir data/korean_basic_v5
```

검사 대상은 `data/korean_basic_v5`다.

| 항목 | 결과 |
|---|---:|
| 전체 샘플 | 11,000 |
| train / validation / test | 8,800 / 1,100 / 1,100 |
| 산수 / 채팅 | 8,000 / 3,000 |
| 필수 필드 오류 | 0 |
| 중복 ID | 0 |
| split 간 동일 입력 누출 | 0 |
| split 내부 중복 | 0 |
| 산수 verifier 일치 | 8,000 / 8,000 |

이전 `korean_basic_v2`에는 split 간 입력 누출 725건이 있었다. 원인은 같은 채팅 템플릿을 행 단위로 복제한 뒤 랜덤 분할한 것이었다. v5에서는 고유 prompt 그룹 단위로 분할하고, 산수 질문도 중복 생성하지 않도록 수정했다.

## 모델 동작

사용한 모델:

```text
artifacts/cerpt-causal-korean-v5-10
hidden size 64 / 1 decoder layer / 10 epochs / CPU
```

학습 loss는 `6.1304 → 1.7204`, validation loss는 `5.5383 → 1.6539`로 감소했다. 저장된 checkpoint 재로딩도 성공했다.

### 확인된 동작

- 새로운 산수 입력 `37에서 8을 빼고 4를 곱하면 얼마야?` → `116`
- 산수 답은 CERPT의 결정론적 verifier가 중간 계산 trace와 함께 정확히 확인
- 채팅 생성은 아직 자연스럽지 않고 `answer 질문에 답하세요` 같은 학습 템플릿을 반복

## 판정

- 데이터 파이프라인: 통과
- 산수 verifier: 통과
- Decoder-only 모델 forward/save/load: 통과
- 자연어 채팅 품질: 미달

현재 결과는 CERPT 구조와 학습 경로가 작동한다는 증거이지, 범용 LLM 또는 완성된 한국어 대화 모델이라는 증거는 아니다. 채팅 품질을 높이려면 더 큰 decoder, 더 다양한 실제·합성 대화, 더 긴 학습, 별도 대화 SFT가 필요하다.
