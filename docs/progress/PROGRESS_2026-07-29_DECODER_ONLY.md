# 진행 기록 — 2026-07-29 Decoder-only 전환

## 완료

- `data/korean_basic_v2` 생성: 한국어 산수 8,000건, 일상 채팅 3,000건
- 산수 입력에 `[TASK_ARITHMETIC]`, 채팅 입력에 `[TASK_CHAT]` 태그 추가
- 산수와 채팅의 operator label을 분리
- 한국어 산수 결정론적 parser/verifier 추가
- 기존 `scripts/chat.py`에서 산수 verifier와 전체 답변 추출 지원
- `CERPTCausalConfig`, `CERPTForCausalLM` 추가
- causal model forward/generate smoke test 통과
- `scripts/train_causal.py`, `scripts/chat_causal.py` 추가
- 기존 V2 Encoder–Decoder 학습 종료: 총 10 epoch, 마지막 validation loss `0.349498`

## 해석

이 단계는 범용 LLM 완성이 아니다. 기존 PoC와 다른 Decoder-only 학습 경로가 제대로 실행되는지 확인하는 단계다. 범용성을 주장하려면 훨씬 큰 한국어·다국어·코드·문서 데이터로 사전학습하고, 별도의 held-out 평가를 통과해야 한다.

## 다음 단계

1. causal 모델로 작은 한국어 smoke 학습
2. 산수·채팅·끝말잇기·코드 trace를 통합한 curriculum 작성
3. workspace/operator/verifier ablation과 일반 causal baseline 비교
4. KV cache 및 vLLM custom backend 설계
