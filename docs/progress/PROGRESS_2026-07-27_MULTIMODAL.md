# 구현 진행 기록: 멀티모달 CERPT 연결

## 완료

1. 텍스트 CERPT core에 `vision_features` 입력을 추가했다.
2. 이미지 evidence token을 텍스트 memory와 합쳐 typed workspace로 전달하도록 했다.
3. Hugging Face `CLIPVisionModel` 기반 멀티모달 wrapper를 추가했다.
4. vision encoder는 기본 frozen이며 projector와 CERPT core만 먼저 학습할 수 있게 했다.
5. 비디오 `[B, T, 3, H, W]` 입력을 추가했다.
6. 비디오 프레임 feature에 temporal Transformer를 적용하고 시간 요약 token을 CERPT에 전달한다.
7. 기존 CERPT checkpoint에 vision encoder를 붙이는 준비 스크립트와 이미지 질문 스크립트를 추가했다.
8. 이미지/비디오 forward smoke test, 문법 검사, 멀티모달 save/load를 통과했다.

## 검증 결과

- `pytest -q`: 6 passed
- 멀티모달 이미지 forward: 통과
- 멀티모달 비디오 forward: 통과
- 멀티모달 checkpoint save/load 후 forward: 통과

## 아직 남은 단계

현재 checkpoint는 텍스트 합성 데이터로만 학습되었다. 따라서 vision encoder를 연결한 직후에는 이미지 내용을 잘 설명한다고 볼 수 없다. 다음 학습 단계에서 이미지-질문-답변 데이터와 CERPT evidence/verification label을 추가해야 실제 시각 응답 성능이 생긴다.
