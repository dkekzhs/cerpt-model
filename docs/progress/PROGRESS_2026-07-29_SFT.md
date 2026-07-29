# 진행 기록 — 2026-07-29 한국어 채팅 SFT

## 실행

기준 모델은 `artifacts/cerpt-causal-korean-v5-10`이고, SFT 데이터는 `data/korean_basic_v6`다. SFT는 `korean_daily_chat` 샘플만 사용하고 prompt 토큰은 loss에서 제외했다.

결과 checkpoint:

```text
artifacts/cerpt-causal-korean-v6-sft
```

## 결과

| epoch | train loss | validation loss |
|---:|---:|---:|
| 1 | 1.5886 | 1.2333 |
| 2 | 1.0520 | 0.8855 |
| 3 | 0.6852 | 0.4643 |
| 4 | 0.3105 | 0.2032 |
| 5 | 0.1522 | 0.1099 |

새 산수 질문 `37에서 8을 빼고 4를 곱하면 얼마야?`는 verifier를 통해 `116`으로 정확히 처리했다.

일상 채팅은 `answer` 접두어 문제는 수정했지만, `안녕`과 `이름이 뭐야?`에서 다른 intent의 답변을 섞는 현상이 남아 있다. 이는 SFT 코드 오류라기보다 hidden 64·1 layer 모델의 용량과 데이터 복잡도 한계로 판정한다.

## 다음 단계

- 최소 hidden 256 이상, decoder layer 3~6으로 확대
- 채팅 intent별 holdout 평가 자동화
- 실제 한국어 대화·질문 응답 데이터 추가
- SFT 이후 산수 성능 보존 여부와 catastrophic forgetting 측정
