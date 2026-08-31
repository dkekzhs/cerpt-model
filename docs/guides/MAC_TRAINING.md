# Apple Silicon 1B 학습과 재개

Mac 경로는 Colab과 같은 `configs/cerpt-causal-1b.json`, 한국어 v7 데이터, 32k tokenizer, 30 epoch 설정과 Hugging Face Trainer 체크포인트 형식을 사용한다. 차이는 T4의 CUDA AMP 대신 Apple MPS에서 FP32와 Adafactor를 쓴다는 점이다.

## 브랜치와 환경 준비

```bash
git switch mac-mps-sft
git pull --ff-only
uv venv --python 3.12
uv sync --locked
.venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"
chmod +x scripts/train_mac.command
```

마지막 명령이 `True`여야 한다. MPS를 사용할 수 없으면 CPU로 조용히 전환하지 않고 학습 시작 전에 중단한다.

## Colab 결과 가져오기

Google Drive의 `cerpt-cloud` 폴더를 프로젝트의 `artifacts/cerpt-cloud`로 복사한다. 기본 구조는 다음과 같다.

```text
artifacts/cerpt-cloud/
├── data/korean_conversations_v7/
├── tokenizers/cerpt-korean-32k/
└── models/cerpt-causal-korean-v7-1b-30ep/
```

모델 폴더까지 복사하면 Mac이 기존 체크포인트에서 이어서 학습한다. 다른 위치를 쓸 때는 실행 창에서 절대 경로를 입력하면 된다.

## Finder에서 실행

Finder에서 `scripts/train_mac.command`를 더블클릭한다.

- `1`: 1B 한국어 base 학습을 시작하거나 재개한다. 기본값은 위 `artifacts/cerpt-cloud` 구조다.
- `2`: 완전한 base checkpoint를 직접 지정해 한국어 chat SFT를 실행한다.

실행기는 프로젝트의 `.venv/bin/python`만 사용한다. 가상환경이 없으면 설치 명령을 표시하고 중단한다.

## 체크포인트 재개 규칙

재개 대상은 `checkpoint-숫자` 중 모델 weight, `trainer_state.json`, `optimizer.pt`, `scheduler.pt`가 모두 있는 가장 최신 폴더다. 저장 중 끊겨 `trainer_state.json` 등이 없는 `checkpoint-1800` 같은 폴더는 건드리거나 삭제하지 않고 직전의 완전한 체크포인트를 선택한다.

동일한 Trainer 형식이므로 Colab에서 Mac으로 옮겨도 model, optimizer, scheduler, global step과 데이터 진행 위치를 이어받는다. CUDA와 MPS의 난수 구현은 다르므로 중단 없이 같은 장치에서 학습한 결과와 비트 단위까지 동일하다고 보장하지는 않는다.

MPS fallback은 MPS 미지원 연산에만 CPU를 사용하도록 켜며, gradient accumulation 32와 gradient checkpointing도 유지한다. 이 경로의 CLI와 CPU 대체 재개 테스트는 검증했지만, 현재 개발 호스트에는 Apple Silicon이 없어 실제 MPS 장시간 학습은 Mac에서 처음 확인해야 한다.
