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

CPU-only PC 대신 무료 GPU 크레딧에서 1B 학습을 시작하고 중단 뒤 재개하려면 [Lightning AI 1B 학습 가이드](guides/LIGHTNING_1B_TRAINING.md)를 참고한다.

Google Colab에서 데이터 업로드부터 Drive checkpoint 재개와 최종 모델 저장까지 셀 순서대로 실행하려면 [Colab 1B 학습 가이드](guides/COLAB_1B_TRAINING.md)와 [실행 노트북](../notebooks/CERPT_1B_Colab_Training.ipynb)을 사용한다.

가장 최근의 코드·문서·serving 판정은 [2026-08-27 현재 상태 감사](progress/PROGRESS_2026-08-27_CURRENT_AUDIT.md)에서 확인할 수 있다. 데이터 자체의 상세 감사는 [2026-07-29 데이터 감사 기록](progress/PROGRESS_2026-07-29_DATA_AUDIT.md)에 있다.

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
- `src/cerpt/models/cerpt_causal.py`: Llama 계열 Decoder-only CERPT Base와 KV cache
- `src/cerpt/data/causal.py`: prompt→workspace token→response causal sequence formatter

마지막 모델은 아직 대규모 사전학습 모델이 아니다. 현재는 RoPE/RMSNorm/SwiGLU/GQA causal decoder, response-only next-token loss, in-stream workspace token, operator/verifier head, KV cache를 한 모델 안에 넣고 학습·저장·생성할 수 있는 연구용 기반이다. head는 아직 operator 실행이나 commit/rollback을 제어하지 않는다. 실제 vLLM 배포에는 custom config 등록과 serving 검증이 필요하고, Ollama에는 별도의 GGUF 변환과 prompt template이 필요하다. 수정 전 v5/v6 causal checkpoint는 누수 구조로 학습되어 재사용하지 않는다.
