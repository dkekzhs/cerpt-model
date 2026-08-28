# Google Colab에서 CERPT 1B 학습하기

[Colab에서 노트북 열기](https://colab.research.google.com/github/dkekzhs/cerpt-model/blob/main/notebooks/CERPT_1B_Colab_Training.ipynb)

실행 노트북은 `notebooks/CERPT_1B_Colab_Training.ipynb`다. 런타임을 GPU로 바꾼 뒤 위에서 아래로 실행하면 데이터 입력부터 최종 모델 저장까지 진행한다.

기존 3.02B 경로는 모델 파라미터만 FP16으로 만든 상태에서 Transformers AMP를 함께 켜 `Attempting to unscale FP16 gradients`로 중단됐고, 정상적인 FP32 파라미터와 gradient만으로도 T4 메모리를 초과한다. 이 노트북은 무료 T4에서 성립하도록 1,020,366,857 parameter preset으로 교체한 경로다.

## 노트북이 처리하는 내용

1. CUDA GPU와 14.5GiB 이상의 GPU 메모리를 확인한다.
2. Google Drive를 마운트하고 영구 저장 경로를 만든다.
3. 오피스 ZIP이 Drive에 없으면 업로드 창으로 받아 Drive에 저장한다.
4. GitHub `main`에서 CERPT 코드를 clone하거나 fast-forward update하고, 세 학습 entry point가 실제 checkout에 있는지 확인한다.
5. Colab에 설치된 CUDA PyTorch는 유지하고, 현재 notebook kernel의 Python에 검증된 나머지 의존성을 설치한 뒤 `TrainingArguments`의 실제 module과 signature를 검사한다.
6. `korean_basic_v6`, 고정 commit의 Songys CSV, 오피스 ZIP을 합치고 중복을 제거한다.
7. base 32,672개와 workspace 96개가 합쳐져 최종 32,768개가 되는 tokenizer를 만든다.
8. 약 1.02B 모델 파라미터는 FP32로 유지하고 Transformers AMP의 FP16 autocast를 사용해 GradScaler가 FP32 gradient를 정상적으로 처리하게 한다.
9. batch 1, gradient accumulation 32, Adafactor, gradient checkpointing으로 30 epoch를 실행한다.
10. 100 optimizer step마다 Drive에 checkpoint를 저장하고 최근 한 개만 유지한다.
11. 완료되면 `final/`, `trainer_state.json`, `TRAINING_COMPLETE`를 Drive에 저장한다.
12. 선택적으로 Colab Secret의 `HF_TOKEN`으로 모델 파일을 Hugging Face에 스트리밍 업로드한다.

학습 셀은 자식 Python 프로세스의 stdout과 stderr를 실시간으로 함께 표시하고 같은 내용을 `MODEL_DIR/training.log`에 누적 저장한다. 프로세스가 실패하면 셀 마지막 오류에 최근 80줄과 전체 로그 경로가 함께 나오므로 바깥쪽 `CalledProcessError`만 보고 원인을 추측하지 않아도 된다.

## 처음 실행할 때 바꿀 값

설정 셀에서 다음 값을 확인한다.

```python
OFFICE_ZIP = Path("/content/drive/MyDrive/대화데이터_오피스(JSON).zip")
HF_REPO_ID = "qweqwqw113/cerpt-causal-korean-v7-1b-30ep"
ACKNOWLEDGE_OFFICE_LICENSE = True
```

`ACKNOWLEDGE_OFFICE_LICENSE`는 오피스 데이터의 학습 및 파생 모델 공개 조건을 직접 확인했을 때만 `True`로 바꾼다. 원본 ZIP과 변환 JSONL은 Hugging Face에 올리지 않는다.

## 런타임이 끊어진 뒤 재개

무료 Colab 세션은 중간에 종료될 수 있다. 새 런타임에서 GPU를 다시 선택하고 노트북을 위에서부터 실행한다. 데이터와 tokenizer가 이미 있으면 해당 단계는 건너뛰며, 학습기는 Drive의 최신 `checkpoint-*`를 찾아 optimizer, scheduler, RNG 상태까지 이어간다.

Drive 기본 저장 위치는 다음과 같다.

```text
/content/drive/MyDrive/cerpt-cloud/
├── data/korean_conversations_v7/
├── tokenizers/cerpt-korean-32k/
└── models/cerpt-causal-korean-v7-1b-30ep/
    ├── checkpoint-*/
    ├── final/
    ├── training.log
    ├── trainer_state.json
    └── TRAINING_COMPLETE
```

FP32 1B weight, 최근 checkpoint 한 개, 최종 모델을 동시에 저장하므로 Drive에는 최소 약 12GiB의 빈 공간을 확보하는 것이 안전하다. 기존 `...-3b-30ep` 폴더는 구조가 달라 재개하지 않으며, 새 1B 경로가 별도 폴더를 사용한다. `TRAINING_COMPLETE`가 생기기 전에는 최종 모델 업로드 셀을 실행할 수 없다.

## 예상 실패와 대응

- `GPU 런타임이 아닙니다`: Colab의 런타임 유형을 GPU로 바꾼다.
- `최소 14.5 GiB`: 배정된 GPU 메모리가 부족하므로 새 GPU 런타임을 요청한다.
- `Git checkout에 필수 학습 파일이 없습니다`: 이전 노트북 셀을 사용 중인 상태다. GitHub의 `main` 노트북을 다시 열고 코드 준비 셀부터 실행한다.
- `unexpected keyword argument 'warmup_ratio'`: dependency 셀을 새 버전으로 다시 실행한다. 이 셀은 `uv` 설치 대상을 `sys.executable`로 고정하고, 학습 전에 transformers 버전·module·signature와 실제 constructor 호출을 검사한다.
- 학습 프로세스가 non-zero exit code로 종료됨: 오류 마지막에 출력되는 최근 80줄을 먼저 확인한다. 전체 traceback은 Drive의 `MODEL_DIR/training.log`에 남아 있으므로 런타임이 종료된 뒤에도 확인할 수 있다.
- `Attempting to unscale FP16 gradients`: 이전 3B 코드 checkout을 실행한 것이다. GitHub `main`의 1B 노트북을 다시 열고 코드 준비 셀부터 실행한다.
- CUDA OOM: 런타임을 재시작해 다른 GPU 프로세스를 비운 뒤 같은 1B 명령을 재개한다. 계속되면 `training.log`의 전체 traceback과 `nvidia-smi` 결과를 확인한다.
- Drive 용량 부족: 이전의 불필요한 모델 파일을 정리하되 최신 `checkpoint-*`는 재개 전까지 보존한다.
- 세션 종료: 오류가 아니라 예상 동작이다. 위에서부터 다시 실행하면 자동 재개한다.

무료 GPU만으로 30 epoch 완료 시간은 보장되지 않는다. 처음 100 step의 소요 시간을 확인하고 `측정 시간 × 약 466`으로 전체 시간을 추정한 뒤 계속할지 판단한다.
