# CERPT 구현 진행 기록

이 문서는 `README.md` 및 `docs/research/` 설계 문서를 실제 코드로 옮긴 진행 기록이다.

## 단계 상태

| 단계 | 상태 | 산출물 |
|---|---|---|
| 0. 요구사항 및 환경 확인 | 완료 | Phase 0~1 PoC 범위 확정, PyTorch/Transformers 설치 확인 |
| 1. 합성 학습 데이터 | 완료 | `data/synthetic/*.jsonl`, `scripts/generate_data.py` |
| 2. CERPT 모델 | 완료 | `src/cerpt/models/cerpt.py` |
| 3. 학습/평가 실행기 | 완료 | `scripts/train.py`, `scripts/evaluate.py` |
| 4. Hugging Face 업로드 준비 | 완료 | `scripts/upload_model.py` |
| 5. 실제 학습 및 성능 검증 | 1차 완료 | 3 epoch 로컬 학습, checkpoint reload, test subset 평가 완료 |
| 6. 검증/감사 기반 추가 | 완료 | deterministic checker, adversarial data, input-evidence ablation audit |
| 7. Decoder-only CERPT Base 골격 | 완료 | `src/cerpt/models/cerpt_causal.py`, `scripts/train_causal.py`, `scripts/chat_causal.py` |
| 8. 문서 구조 정리 | 완료 | `docs/research`, `docs/guides`, `docs/data`, `docs/model-cards`, `docs/progress` |
| 9. 한국어 데이터 품질 감사 | 완료 | `data/korean_basic_v5`, `scripts/audit_korean_basic.py` |

## 현재 구현 범위

- 네 가지 deterministic synthetic task: arithmetic, binding, graph, constraints
- 각 샘플에 자연어 문제, 구조화된 trace, operator labels, cycle validity labels, answer 저장
- offline-friendly Hugging Face `PreTrainedTokenizerFast`
- PyTorch encoder-decoder backbone
- typed workspace seed와 slot type embedding
- 공유 transition core를 `num_cycles`번 재사용
- cycle별 operator prediction 및 learned verification/commit gate
- causal decoder와 `save_pretrained`/`from_pretrained` 호환 checkpoint
- generation, exact-match 평가, commit gate 평균 기록
- 대화형 추론 CLI: `scripts/chat.py`

## 1차 실행 결과

- 데이터: train 4,000 / validation 500 / test 500
- 모델: hidden size 64, workspace 8 slots, shared cycles 4, encoder/decoder 2 layers
- 학습: seed 42, batch 32, 3 epochs, CPU
- loss: train `0.733`, validation `0.514`
- checkpoint: `artifacts/cerpt-small`에 `save_pretrained` 형식으로 저장 및 `from_pretrained` 재로딩 성공
- 평가: test 앞 32개 greedy generation에서 exact-match `2/32 (6.25%)`
- 관찰: 평균 commit gate가 약 `0.9998`로 포화. 현재 모든 학습 transition이 positive라 verifier가 reject를 학습할 negative signal이 부족하다.
- deterministic checker: test 500/500 정답 검증 성공
- adversarial data: fabricated/contradicted/irrelevant/copied-answer 256건 생성
- causal audit(test 32): full accuracy `6.25%`, ablated accuracy `6.25%`, prediction sensitivity `0%`
- 해석: 현재 모델은 입력 evidence 제거에 반응하지 않아 shortcut/underfitting 기준선을 확인했다. 따라서 CERPT의 causal utility 주장을 아직 할 수 없다.

## 1000 epoch 학습 실행 상태

- 기존 3 epoch checkpoint에서 이어 학습 시작
- Windows 파일 잠금 문제로 출력 경로를 `artifacts/cerpt-small-1000`으로 분리
- 목표: 기존 3 epoch + 추가 997 epoch = 총 1000 epoch
- 현재 확인 상태: 학습 중지, 총 71 epoch까지 완료. 마지막 validation loss `0.02578`
- 백그라운드 로그: `artifacts/cerpt-small-1000/training-1000.log`
- 최신 복구 checkpoint: `artifacts/cerpt-small-1000/latest`
- 재개 지원: `scripts/train.py --resume-from ... --start-epoch 71 --epochs 1000`

## 실행 방법

```powershell
python scripts/generate_data.py --output-dir data/synthetic
python scripts/train.py --data-dir data/synthetic --output-dir artifacts/cerpt-small
python scripts/evaluate.py --model-dir artifacts/cerpt-small --data data/synthetic/test.jsonl
python scripts/chat.py --model-dir artifacts/cerpt-small
```

Hub 업로드는 인증 후 명시적으로 실행한다.

```powershell
python scripts/upload_model.py --model-dir artifacts/cerpt-small --repo-id <HF_USER>/cerpt-small
```

## 연구상 주의점

현재 모델은 문서의 100M~300M 목표 모델이 아니라 구조 검증용 소형 PoC다. 데이터도 합성 알고리즘 과제에 한정되어 있으므로, 이 단계의 성능은 일반 언어 능력이나 CERPT의 최종 연구 주장을 의미하지 않는다. 다음 구현 단계는 identity perturbation, verifier negative-sample 학습, Dense/recursive baseline 비교다.

## Decoder-only smoke 결과

- 데이터: `data/korean_basic_v2`
- 설정: hidden 64, 1 layer, batch 512, 1 epoch, CPU
- train loss: `6.07056`
- validation loss: `5.50206`
- 저장·재로딩·생성 CLI 확인 완료
- 산수 질문은 deterministic verifier가 `30`을 검증해 반환
- 일상 채팅은 생성되지만 아직 짧은 smoke 학습이라 자연스러운 대화 품질을 보장하지 않음

최신 10 epoch 검증 결과는 [데이터 감사 기록](PROGRESS_2026-07-29_DATA_AUDIT.md)에 별도로 기록했다.
