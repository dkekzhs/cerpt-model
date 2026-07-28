# CERPT 한국어 Stage 1 데이터

이 단계의 목표는 CERPT를 범용 LLM으로 만드는 것이 아니라, 작은 모델이 한국어 문제를 읽고 `EXTRACT → SIMULATE/BIND → CHECK → WRITE_RESULT`라는 typed workspace 추론 흐름을 학습하는 것이다.

## 선택한 소형 공개 데이터

| 데이터셋 | 용도 | 라이선스 | 원본 |
| --- | --- | --- | --- |
| OpenMathReasoning-mini-ko | 한국어 수학 풀이 | CC-BY-4.0 | [Hugging Face 카드](https://huggingface.co/datasets/neuralfoundry-coder/OpenMathReasoning-mini-ko) |
| HRMCR | 한국어 알고리즘 논리 추론 | Apache-2.0 | [Hugging Face 카드](https://huggingface.co/datasets/HAERAE-HUB/HRMCR) |
| Ko-StrategyQA | 한국어 다중 홉 질문·근거 결합 | Apache-2.0 | [Hugging Face 카드](https://huggingface.co/datasets/NomaDamas/Ko-StrategyQA) |

원본은 `data/korean_stage1/raw/`에 보관한다. 원본 데이터셋의 라이선스와 제공자 표시를 유지해야 하며, 원본을 GitHub나 Hugging Face 모델 저장소에 자동으로 재배포하지 않는다.

## 변환 실행

프로젝트 루트에서 다음을 실행한다.

```powershell
pip install -e .
python scripts/prepare_korean_stage1.py
```

생성 파일:

- `data/korean_stage1/train.jsonl`
- `data/korean_stage1/validation.jsonl`
- `data/korean_stage1/test.jsonl`
- `data/korean_stage1/metadata.json`

각 레코드는 CERPT 학습기가 사용하는 `input_text`, `target_text`, `answer`, `operator_labels`, `cycle_valid_labels`, `trace`를 포함한다. 수학·HRMCR은 `SIMULATE`, StrategyQA는 근거 결합을 뜻하는 `BIND`를 두 번째 연산자로 사용한다. 긴 원본 풀이 전체는 target에 넣지 않고 짧은 증거로 요약해 작은 모델의 출력 길이를 제한한다.

## Stage 1 학습

데이터 변환이 끝난 뒤, 현재 CERPT 학습 코드로 작은 실험을 실행할 수 있다.

```powershell
python scripts/train.py `
  --data-dir data/korean_stage1 `
  --output-dir artifacts/cerpt-korean-stage1 `
  --epochs 1 `
  --batch-size 16 `
  --max-length 256
```

`--epochs 1`은 먼저 파이프라인을 확인하는 smoke run이다. 실제 성능 비교는 seed를 고정한 뒤 여러 epoch로 학습하고 `test.jsonl`을 별도로 평가해야 한다. 이 데이터만으로 범용 LLM이 되지는 않는다.

## 포함하지 않은 것

이번 소형 Stage 1에는 실행 가능한 코드 디버깅 데이터와 이미지·비디오 데이터는 포함하지 않았다. 코드 실행 검증은 별도의 샌드박스와 테스트 케이스가 필요하고, 멀티모달 학습은 vision/video encoder와 projector를 붙이는 별도 단계이기 때문이다. 다음 단계에서 라이선스가 명확한 코드·실행 검증 세트를 추가하고 CERPT workspace에 관찰 결과를 기록한다.

## 재현성 기록

- 분할 seed: `42`
- 분할: task type별 80% train / 10% validation / 10% test
- 생성 스크립트: `scripts/prepare_korean_stage1.py`
- 원본 다운로드 위치: `data/korean_stage1/raw/`
