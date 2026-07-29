# Apple Silicon 원클릭 학습

## 준비

터미널에서 프로젝트 최초 1회만 실행한다.

```bash
cd /path/to/cerpt-planning
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

`MPS available` 여부를 확인한다.

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

## 딸깍 실행

Finder에서 `scripts/train_mac.command`를 더블클릭한다.

- `1`: 현재 한국어 채팅 checkpoint에 SFT
- `2`: 3B 사전학습 preset 실행

3B 모드는 32k tokenizer와 대규모 pretraining shard가 이미 준비되어 있어야 한다. 현재 `data/korean_basic_v6`는 3B 사전학습용 corpus가 아니므로 3B 모드에 사용하지 않는다.

## 안전장치

- MPS를 강제로 선택하고 CUDA/CPU로 조용히 내려가지 않는다.
- MPS 미지원이면 시작 전에 중단한다.
- MPS fallback은 지원되지 않는 연산에만 CPU를 사용하도록 설정한다.
- gradient accumulation과 gradient checkpointing을 켠다.
- 기본 SFT는 현재 검증된 checkpoint를 별도 output directory에 저장한다.
- 로컬에 Base checkpoint가 없으면 Hugging Face `qweqwqw113/cerpt-causal-korean-v5-10`에서 자동으로 불러온다.
- 256GB 통합 메모리를 활용해 호환성 우선으로 기본 precision은 fp32다.

현재 구현은 Mac에서 MPS 동작을 코드 수준으로 검증했지만, 실제 M4 Pro 하드웨어 테스트는 해당 Mac에서 처음 실행해야 한다.
