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

- `1`: `--resume-from`으로 지정한 새 causal base checkpoint에 SFT
- `2`: 3B 사전학습 preset 실행

3B 모드는 32k tokenizer와 대규모 pretraining shard가 이미 준비되어 있어야 한다. 현재 `data/korean_basic_v6`는 3B 사전학습용 corpus가 아니므로 3B 모드에 사용하지 않는다.

## 안전장치

- MPS를 강제로 선택하고 CUDA/CPU로 조용히 내려가지 않는다.
- MPS 미지원이면 시작 전에 중단한다.
- MPS fallback은 지원되지 않는 연산에만 CPU를 사용하도록 설정한다.
- gradient accumulation과 gradient checkpointing을 켠다.
- SFT 모드는 새 구조로 처음부터 학습한 base checkpoint를 `--resume-from`으로 반드시 지정한다.
- 누수 구조로 학습된 v5/v6 checkpoint를 자동 다운로드하거나 이어 학습하지 않는다.
- 256GB 통합 메모리를 활용해 호환성 우선으로 기본 precision은 fp32다.

현재 구현은 Mac에서 MPS 동작을 코드 수준으로 검증했지만, 실제 M4 Pro 하드웨어 테스트는 해당 Mac에서 처음 실행해야 한다.
