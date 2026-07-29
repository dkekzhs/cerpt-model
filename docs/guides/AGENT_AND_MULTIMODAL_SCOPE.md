# CERPT 공개 연구 프레임워크 범위

## 핵심 목표

CERPT는 하나의 완성된 범용 LLM을 제공하는 프로젝트가 아니라, 같은
recursive reasoning core를 공유하면서 task별로 작은 전문 모델을 만들고
재현·비교할 수 있는 공개 프레임워크를 목표로 한다.

```text
Shared CERPT Core
  ├─ task adapter: word-chain
  ├─ task adapter: combat planning
  ├─ NPC adapter: merchant-001
  ├─ NPC adapter: guard-017
  └─ modality adapters: text / image / video
```

1,000명의 NPC를 만들 때 1,000개의 전체 모델을 복제하지 않는다. 하나의
shared base를 메모리에 올리고, NPC별로 다음만 분리한다.

- 작은 LoRA 또는 task adapter
- persona/configuration
- private memory/state
- 허용된 tool과 행동 정책

## 멀티모달 입력 설계

CERPT core는 raw pixel이나 raw video frame을 직접 처리하지 않는다. 각
modality encoder가 evidence token을 만들고, 같은 typed workspace에 기록한다.

```text
Text  ── Text Encoder ───┐
Image ── Vision Encoder ─┼─ Evidence Projector ── Typed Workspace
Video ── Temporal Encoder┘                              │
                                                       ▼
                                      Operator → Propose → Verify → Commit
                                                       │
                                                       ▼
                                             Text / Action / Tool output
```

권장 workspace slot 타입:

- `text_fact`
- `visual_object`
- `visual_relation`
- `temporal_event`
- `audio_evidence`
- `hypothesis`
- `counterexample`
- `answer_or_action`

초기 멀티모달 구현에서는 vision/video encoder를 frozen하고 projector,
workspace, verifier, task adapter만 학습한다. 이후 필요한 경우에만 encoder
일부를 LoRA로 미세조정한다. 이 방식이 작은 GPU와 여러 NPC adapter에 적합하다.

## 끝말잇기 전문 에이전트

끝말잇기는 범용 사전학습의 대체가 아니라 CERPT의 전문-task 검증으로 사용한다.

workspace 예시:

```text
previous_word
required_syllable
candidate_word
dictionary_evidence
used_word_history
opponent_state
```

필수 checker:

- 실제 사전에 존재하는 단어인지
- 요구 음절로 시작하는지
- 이미 사용한 단어가 아닌지
- 게임 규칙에 맞는지

평가에서는 train에 없는 단어, 희귀 단어, 긴 게임, adversarial opponent를
사용해 단순 transition 암기와 실제 규칙 추론을 구분한다.

## 공개 저장소의 완료 조건

- text/image/video 입력 인터페이스
- modality별 evidence projector
- task adapter와 NPC adapter 예제
- deterministic checker와 benchmark
- seed/config/checkpoint가 포함된 재현 명령
- 각 checkpoint의 데이터·학습 단계·제한을 설명하는 model card
- 범용 모델이라고 과장하지 않는 명확한 capability report
