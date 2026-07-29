# CERPT 문서 안내

이 저장소에는 성격이 다른 문서가 함께 있다. 아래 기준으로 보면 된다.

## 어디부터 읽나

처음 보는 사람은 다음 순서가 가장 빠르다.

1. 루트 [README.md](../README.md): 프로젝트 목적과 전체 문서 지도
2. [Decoder-only CERPT Base](implementation/DECODER_ONLY_CERPT_BASE.md): 현재 구현 방향과 제한
3. [구현 진행 기록](progress/IMPLEMENTATION_STATUS.md): 실제로 완료된 코드와 성능
4. [공개·실행 가이드](guides/PUBLISHING.md): 다른 PC, GitHub, Hugging Face 사용법

SFT를 추가하려면 [SFT 가이드](guides/SFT.md)를 참고한다.

Mac에서 더블클릭으로 학습하려면 [Apple Silicon 학습 가이드](guides/MAC_TRAINING.md)를 참고한다.

가장 최근의 데이터와 모델 판정은 [2026-07-29 데이터 감사 기록](progress/PROGRESS_2026-07-29_DATA_AUDIT.md)에서 확인할 수 있다.

## 폴더별 의미

| 폴더 | 질문 | 포함 내용 |
|---|---|---|
| `research/` | 무엇을 연구할 것인가? | 목표, 가설, 아키텍처, 검증, 로드맵 |
| `implementation/` | 지금 코드는 어떤 구조인가? | Decoder-only Base 설계와 vLLM 전 단계 |
| `guides/` | 어떻게 실행·학습·공개하는가? | 실행 명령, 멀티모달, GitHub/Hugging Face |
| `data/` | 어떤 데이터를 쓰는가? | 데이터셋 카드, 한국어 데이터 준비 |
| `model-cards/` | 공개할 때 무엇이라고 설명하는가? | 모델의 용도, 제한, 학습 정보 |
| `progress/` | 어디까지 했는가? | 날짜별 진행 및 실험 결과 |

## 현재 모델을 정확히 구분하기

- `src/cerpt/models/cerpt.py`: 기존 CERPT Encoder–Decoder 구조 검증용 PoC
- `src/cerpt/models/multimodal.py`: 기존 텍스트 core에 vision/video 입력 경로를 연결한 PoC
- `src/cerpt/models/cerpt_causal.py`: 새로 시작한 자체 Decoder-only CERPT Base 골격

마지막 모델은 아직 대규모 사전학습 모델이 아니다. 현재는 causal next-token 학습, CERPT workspace, operator head, verifier head를 한 모델 안에 넣고 학습·저장·생성할 수 있는 연구용 기반이다. 실제 vLLM 배포에는 KV cache, custom model registration, attention backend를 추가해야 한다.
