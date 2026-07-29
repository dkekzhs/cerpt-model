# 06. 학습 전략

## 1. 학습 단계 개요

CERPT는 next-token prediction만으로 충분히 학습되기 어렵다. 단계별 curriculum을 사용한다.

```text
Stage 0: Backbone pretraining
Stage 1: State extraction
Stage 2: State transition
Stage 3: Program imitation
Stage 4: Certification/adversarial training
Stage 5: Joint end-to-end training
Stage 6: Budget and depth generalization
Stage 7: Instruction/reasoning alignment
```

## 2. Stage 0: Backbone

선택지:

- 처음부터 소형 LM 사전학습
- 기존 공개 checkpoint에서 시작
- parameter tying을 적용한 recursive backbone으로 변환

초기 연구에서는 구조 비교를 위해 동일 tokenizer, 동일 데이터, 동일 training FLOPs를 사용한다.

## 3. Stage 1: State extraction

입력과 teacher trace에서 다음 구조를 추출하도록 학습한다.

```text
Goal
Fact
Constraint
Hypothesis
Evidence
Result
Answer
```

Teacher 데이터는 자연어 JSON 형태로 시작하고, 점차 latent state로 증류한다.

### 데이터 소스

- 수학 풀이
- 코드 디버깅 trace
- 논리 문제
- 다중 문서 QA
- 실행 가능한 synthetic task

## 4. Stage 2: State transition

입력:

```text
current state + operator
```

출력:

```text
state delta + evidence + verification plan
```

positive와 negative transition을 함께 학습한다.

## 5. Stage 3: Program imitation

Teacher 또는 search가 생성한 operator program을 따라 하게 한다.

초기에는 정해진 template을 사용한다.

- extract-compare-write
- hypothesize-check-invalidate
- branch-verify-select
- retrieve-compose-ground

그 후 latent controller로 증류한다.

## 6. Stage 4: Adversarial certification

가짜 evidence 생성기를 별도로 둔다.

Negative 유형:

- irrelevant
- circular
- fabricated
- contradicted
- copied answer
- task identity leakage
- premature conclusion
- verifier-targeted pattern

Generator와 adversary를 교대로 업데이트한다.

## 7. Stage 5: Joint training

전체 손실 예시:

```text
L_total =
  L_lm
+ λ_state L_state
+ λ_program L_program
+ λ_evidence L_evidence
+ λ_causal L_causal
+ λ_energy L_energy
+ λ_budget L_budget
+ λ_identity L_identity
+ λ_diversity L_diversity
```

### L_lm

일반 언어 모델링 품질 유지.

### L_state

typed state reconstruction 및 transition 정확도.

### L_program

operator sequence 및 template 선택.

### L_evidence

evidence validity 및 provenance.

### L_causal

state delta 제거 시 성능 하락 유도.

### L_energy

모순·미해결 상태 감소.

### L_budget

평균 compute budget 제약.

### L_identity

동형 변환 문제에서 동일한 transition 유도.

### L_diversity

operator와 branch가 기능적으로 붕괴하지 않게 함.

## 8. Stage 6: Budget randomization

동일 문제를 서로 다른 계산 예산으로 학습한다.

- 0 cycle
- 1 cycle
- 2 cycle
- 4 cycle
- 8 cycle

학습 목표:

- 작은 budget: 빠른 근사 답
- 큰 budget: 더 많은 evidence와 높은 정확도
- 추가 cycle이 해롭지 않도록 calibration

## 9. Depth extrapolation

훈련 최대 cycle보다 긴 평가를 준비한다.

방법:

- cycle count randomization
- skipped-cycle training
- step embedding interpolation/extrapolation
- repeated-state detection
- cycle-specific state probes

## 10. Scaling curriculum

### 100M~300M

- synthetic algorithmic tasks 비중 높음
- 명시적인 state/operator supervision
- deterministic checker 활용

### 1B

- 자연어·코드·수학 혼합
- latent program 비중 확대
- segment-level reasoning

### 3B~7B

- tool use
- long-context memory
- open-domain evidence
- verifier ensemble
- harder adversarial data

## 11. 데이터 품질 원칙

- 정답뿐 아니라 중간 상태를 검증 가능하게 생성
- task ID와 템플릿 leakage 제거
- train/test 구조적 분리
- 동일 문제의 표면 변환 버전 포함
- negative evidence 비율 관리
- teacher trace를 그대로 신뢰하지 않고 실행/검증
