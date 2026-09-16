# Apple Silicon 원클릭 학습

## 준비

터미널에서 프로젝트 최초 1회만 실행한다.

```bash
cd /path/to/cerpt-planning
brew install uv
uv venv --python 3.12
uv sync --locked
```

`MPS available` 여부를 확인한다.

```bash
.venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"
```

## 딸깍 실행

Finder에서 `scripts/train_mac.command`를 더블클릭한다.

- `1`: 한국어 전용 3B 사전학습. 데이터와 tokenizer가 없으면 자동 준비
- `2`: `--resume-from`으로 지정한 새 causal base checkpoint에 SFT

3B 모드는 `HuggingFaceFW/fineweb-2`의 한국어 전용 `kor_Hang` subset을 streaming으로 받는다. 기본값은 500만 원문 문서이며 `0`을 입력하면 약 6,087만 문서, 486억 token인 전체 한국어 corpus를 사용한다. 전체 source 크기는 약 98.5GB이고 생성한 JSONL과 checkpoint 공간이 별도로 필요하다. 라이선스는 ODC-By 1.0이며 Common Crawl 이용 조건도 적용된다.

준비 결과는 다음 위치에 저장되며, 세 파일이 이미 있으면 다시 다운로드하지 않는다.

```text
data/pretraining_shards/{train.jsonl,validation.jsonl,metadata.json}
artifacts/tokenizers/cerpt-korean-32k/
```

기본 500만 문서는 실행 가능성을 위한 starter corpus다. 범용 한국어 3B 품질을 목표로 한다면 전체 corpus와 충분한 학습 token budget이 필요하며, 단순히 1 epoch를 완료하는 것만으로 품질이 보장되지는 않는다. `data/korean_basic_v6`는 SFT용 소규모 데이터이므로 3B 사전학습 corpus로 사용하지 않는다.

## 안전장치

- MPS를 강제로 선택하고 CUDA/CPU로 조용히 내려가지 않는다.
- MPS 미지원이면 시작 전에 중단한다.
- MPS fallback은 지원되지 않는 연산에만 CPU를 사용하도록 설정한다.
- gradient accumulation과 gradient checkpointing을 켠다.
- SFT 모드는 새 구조로 처음부터 학습한 base checkpoint를 `--resume-from`으로 반드시 지정한다.
- 누수 구조로 학습된 v5/v6 checkpoint를 자동 다운로드하거나 이어 학습하지 않는다.
- 256GB 통합 메모리를 활용해 호환성 우선으로 기본 precision은 fp32다.
- 실행기는 system `python3`가 아니라 프로젝트의 `.venv/bin/python`만 사용한다.

현재 구현은 Mac에서 MPS 동작을 코드 수준으로 검증했지만, 실제 M4 Pro 하드웨어 테스트는 해당 Mac에서 처음 실행해야 한다.
