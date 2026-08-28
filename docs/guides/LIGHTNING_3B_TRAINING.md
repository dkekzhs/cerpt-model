# Lightning AI 무료 크레딧으로 CERPT 3B 학습하기

## 결론

현재 기본 경로는 Lightning AI Studio다. 무료 등급은 GPU 크레딧, 영구 Studio 저장소, 관리형 secret, interruptible GPU를 제공한다. 무료 Studio는 4시간마다 재시작될 수 있고 GPU 자체도 회수될 수 있으므로 이 저장소의 클라우드 학습기는 100 optimizer step마다 모델·optimizer·scheduler·RNG 상태를 저장하고, 같은 명령을 다시 실행하면 가장 최신 `checkpoint-*`에서 자동으로 이어간다.

무료 GPU나 총 학습 시간은 보장되지 않는다. Lightning의 가격·크레딧·GPU 재고는 바뀔 수 있으므로 시작 직전에 [공식 가격표](https://lightning.ai/pricing)를 확인해야 한다. 현재 공개 안내는 무료 크레딧과 GPU 시간을 표시하고, 무료 Studio에 4시간 재시작 제한을 둔다. [환경 영속성 문서](https://lightning.ai/docs/overview/ai-studio/environment-persistence)에 따르면 `/teamspace/studios/this_studio` 아래 파일과 설치 환경은 sleep·machine switch 뒤에도 유지된다.

## 다른 무료 후보보다 Lightning을 기본으로 삼은 이유

| 서비스 | 현재 공식 제약 | CERPT 3B 판정 |
|---|---|---|
| Lightning AI | 무료 크레딧, 영구 저장소, T4부터 H200까지 선택, 4시간 재시작 | 기본 경로. 중간 checkpoint와 자동 resume로 재시작을 견딤 |
| Kaggle | 무료 P100, GPU quota는 보통 주 30시간이며 수요에 따라 변동 | 16GB와 구형 FP16 성능 때문에 3B 30 epoch 기본 경로로는 느림 |
| Google Colab Free | GPU 종류·사용 한도·VM 수명이 동적으로 변하고 무료 분산 worker가 제한됨 | 실험용 대안. 장시간 3B 학습의 재현 가능한 기본값으로 삼기 어려움 |
| Modal Starter | 매월 무료 compute credit과 A100/H100 선택 가능 | 빠른 대안이지만 무료 크레딧만으로 30 epoch 완료를 보장할 수 없음 |

Kaggle의 현재 GPU 안내는 [공식 GPU 사용 문서](https://www.kaggle.com/docs/efficient-gpu-usage), Colab 제약은 [공식 FAQ](https://research.google.com/colaboratory/faq.html), Modal 크레딧은 [공식 가격표](https://modal.com/pricing)에서 확인한다.

## 먼저 알아야 할 한계

- 이 경로는 `configs/cerpt-causal-3b.json`의 3,020,101,641 parameter 모델을 작은 모델로 바꾸지 않는다.
- T4 profile은 FP16 parameter, Adafactor, batch 1, gradient accumulation 32, gradient checkpointing을 사용한다. GPU가 15GiB 미만이면 학습 전에 중단한다.
- 실제 T4 peak memory와 처리량은 Lightning 이미지와 PyTorch 버전에 따라 달라진다. T4에서 OOM이면 같은 명령으로 L40S/A100/H100 Studio를 선택한다.
- 약 69k 대화쌍을 30번 보는 것은 범용 3B pretraining이 아니다. 과적합과 데이터 편향을 측정하는 한국어 대화 특화 실험이다.
- 무료 크레딧 한 번으로 끝난다고 가정하지 않는다. 크레딧이 다시 생기거나 다른 제공자로 checkpoint를 옮긴 뒤 같은 출력 폴더에서 계속할 수 있다.

## 1. Studio와 secret 준비

1. [Lightning AI](https://lightning.ai/)에서 계정을 만들고 빈 Studio를 CPU로 시작한다.
2. Teamspace 또는 사용자 Secrets에 `HF_TOKEN`을 추가한다. 토큰은 Hugging Face write 권한이 있어야 한다. Lightning은 secret을 암호화해 환경변수로 제공한다. 자세한 절차는 [Managed secrets 문서](https://lightning.ai/docs/platform/build/ai-studio/managed-secrets)에 있다.
3. 로컬의 `대화데이터_오피스(JSON).zip`을 Studio의 `/teamspace/studios/this_studio/` 아래로 업로드한다.
4. Studio 터미널에서 저장소를 받는다.

```bash
cd /teamspace/studios/this_studio
git clone https://github.com/dkekzhs/cerpt-model.git
cd cerpt-model
```

## 2. CPU에서 데이터와 32k tokenizer 준비

오피스 ZIP의 이용·학습·파생 모델 공개 조건을 직접 확인한 경우에만 acknowledgement를 켠다. 변환본과 원본은 Git에 들어가지 않는다.

```bash
export CERPT_ACK_OFFICE_LICENSE=1
bash scripts/lightning_3b.sh prepare \
  "/teamspace/studios/this_studio/대화데이터_오피스(JSON).zip"
```

이 명령은 다음을 수행한다.

1. `uv.lock`에 고정된 검증 버전으로 Python 환경을 준비한다.
2. `songys/Chatbot_data`의 고정 commit `4cf20d13fc46f5037fd1c531cd566e2dd9f72974`에서 CSV와 MIT license를 받는다.
3. `korean_basic_v6`, Chatbot_data, 오피스 ZIP을 정규화하고 완전 중복을 제거한다.
4. source/task별로 seed 42의 80/10/10 split을 새로 만든다.
5. learned/base token 32,672개를 학습한다. 학습기가 workspace token 96개를 더하면 최종 vocabulary가 정확히 32,768개가 된다.

2026-08-28에 같은 세 입력으로 실제 준비 명령을 검증했을 때 완전 중복 7,142쌍이 제거됐고, 최종 62,095쌍이 train 49,673 / validation 6,207 / test 6,215로 분할됐다. source별 최종 건수는 기존 합성 11,000 / Chatbot_data 11,750 / 오피스 39,345였으며, 빈 필드·JSON 오류·split 간 중복은 0건이었다. 이 데이터로 tokenizer를 저장한 뒤 확인한 결과도 base 32,672 + workspace 96 = 최종 32,768이었다.

## 3. GPU에서 30 epoch 실행

Studio machine을 interruptible T4로 바꾸고 다음 명령을 실행한다. Interruptible 설정은 비용을 줄이지만 회수될 수 있으므로 checkpoint가 필수다. Lightning도 [interruptible machine 문서](https://lightning.ai/docs/overview/ai-studio/interruptible-machines)에서 정기 checkpoint와 fault-tolerant resume를 권장한다.

```bash
bash scripts/lightning_3b.sh train
```

브라우저를 닫아도 실행 중인 프로세스는 계속된다. 4시간 재시작, interruptible 회수, 크레딧 소진으로 멈추면 Studio를 다시 GPU로 바꾸고 같은 명령을 다시 실행한다. 마지막 두 checkpoint만 유지하므로 기본 50GB 저장소 안에서 회전한다.

현재 진행 상태는 CPU Studio에서도 확인할 수 있다.

```bash
bash scripts/lightning_3b.sh status
```

30 epoch가 모두 끝나면 아래 파일이 생긴다.

```text
/teamspace/studios/this_studio/cerpt-cloud/models/cerpt-causal-korean-v7-3b-30ep/TRAINING_COMPLETE
```

이 파일이 없으면 최종 모델로 업로드하지 않는다.

## 4. Hugging Face 업로드

`HF_TOKEN` secret이 새 터미널에 보이는지 확인하고 실행한다. 기본 저장소는 `qweqwqw113/cerpt-causal-korean-v7-3b-30ep`이며 `HF_REPO_ID`로 바꿀 수 있다.

```bash
export HF_REPO_ID=qweqwqw113/cerpt-causal-korean-v7-3b-30ep
bash scripts/lightning_3b.sh upload
```

업로더는 `TRAINING_COMPLETE`가 없으면 실행되지 않는다. 3B 모델을 CPU 메모리에 다시 올리지 않고 저장된 shard·tokenizer·model card를 그대로 스트리밍하며, 완료 요약과 Trainer state도 함께 올린다. 원본 대화 데이터와 변환 JSONL은 업로드하지 않는다.

## 비용 사고 방지

- 데이터 준비와 tokenizer 학습은 CPU Studio에서 끝낸다.
- Teamspace에서 interruptible machine을 기본으로 켜고 credit 사용량을 확인한다.
- GPU terminal에서 학습 외 작업을 하지 않는다.
- 학습이 끝나면 프로세스 종료 후 auto-sleep이 작동하는지 확인한다.
- 무료 혜택을 넘겨 자동 결제가 가능한 계정이라면 Lightning의 spending limit을 먼저 설정한다.
