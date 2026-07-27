# CERPT 연구·개발 기획서

> **CERPT (Causal Evidence Recursive Program Transformer)**는 중간 추론
> 결과를 typed workspace에 보존하고, operator program으로 상태를 전이하며,
> 새 evidence가 실제 예측 개선에 기여하는지 검증한 뒤에만 commit하는
> recursive reasoning architecture다.

이 저장소의 현재 구현은 CERPT 구조를 검증하기 위한 **text-only synthetic-task
PoC**다. 범용 사전학습 LLM이나 완성된 멀티모달 모델이 아니며, 다음 단계에서
자연어·코드 사전학습, 한국어 끝말잇기 평가, 이미지·비디오 evidence encoder를
추가해 CERPT 기반 범용 모델로 확장한다.

CERPT의 핵심 목표는 하나의 거대한 모델만 만드는 것이 아니다. 하나의
공유 CERPT core 위에 task adapter, NPC별 persona/memory, image/video evidence
encoder를 조합해 낮은 용량의 전문 에이전트를 반복해서 만들고 비교할 수 있는
공개 연구 프레임워크를 제공하는 것이다. 예를 들어 1,000명의 게임 NPC는
같은 base core를 공유하고, 각 NPC는 별도의 작은 adapter와 상태 메모리만
가질 수 있다.

## 1. 문서 목적

이 문서 묶음은 작은 저장 파라미터로 높은 추론 성능을 달성하고, 동일한 구조를 더 큰 모델로 확장했을 때 기존 Dense Transformer보다 높은 파라미터·추론 계산 효율을 얻기 위한 연구 및 개발 계획을 정의한다.

최종 목표는 단순히 작은 모델을 반복 실행하는 것이 아니다. 모델이 중간 계산 결과를 구조화된 상태로 보존하고, 매 계산 단계에서 새로운 증거를 생성하며, 해당 증거가 실제 예측 개선에 기여하는지 검증한 뒤에만 상태를 갱신하도록 만드는 것이다.

가칭 모델명은 다음과 같다.

> **CERPT — Causal Evidence Recursive Program Transformer**

## 2. 핵심 문제 정의

기존 Transformer 계열 모델은 다음 한계를 가진다.

1. 지식 저장과 사고 알고리즘이 동일한 파라미터에 혼재한다.
2. 모델의 깊이와 저장 파라미터 수가 강하게 결합되어 있다.
3. 반복 계산이 실제 정보 이득인지, 단순 confidence polishing인지 구분하기 어렵다.
4. 중간 추론 상태가 residual stream에 섞여 이전 정보가 덮어써질 수 있다.
5. learned verifier와 reasoning trace가 서로 결탁해 가짜 progress를 만들 수 있다.
6. 작은 모델의 반복 성능이 task ID, 템플릿, 다수결 또는 과도한 test-time compute에 의존할 수 있다.

CERPT는 이 문제를 다음 원칙으로 해결한다.

- **Persistent State**: 중간 상태를 역할별 workspace에 보존한다.
- **Programmed Transition**: expert 가중합이 아니라 순서 있는 상태 전이 프로그램을 실행한다.
- **Evidence-Certified Update**: 새 상태가 검증 가능한 증거를 포함할 때만 commit한다.
- **Causal Utility**: 새 evidence를 제거했을 때 성능이 떨어지는지 측정한다.
- **Adaptive Resource Allocation**: depth, memory, branch를 공동 배분한다.
- **Anti-Shortcut Evaluation**: task ID, 표면 패턴, verifier 해킹을 제거한 상태에서도 성능을 검증한다.

## 3. 문서 목록

| 파일 | 내용 |
|---|---|
| [01-goals-and-success-criteria.md](./01-goals-and-success-criteria.md) | 연구 목표, 성공 기준, 비목표 |
| [02-research-hypotheses.md](./02-research-hypotheses.md) | 핵심 연구 가설과 검증 가능한 주장 |
| [03-system-architecture.md](./03-system-architecture.md) | 전체 모델 구조와 데이터 흐름 |
| [04-state-and-program-design.md](./04-state-and-program-design.md) | workspace, operator, controller 상세 설계 |
| [05-certification-and-verification.md](./05-certification-and-verification.md) | evidence 검증, commit/rollback, shortcut 방지 |
| [06-training-strategy.md](./06-training-strategy.md) | 사전학습, curriculum, 손실 함수, 데이터 구성 |
| [07-experiment-and-evaluation.md](./07-experiment-and-evaluation.md) | 비교군, 실험, ablation, 평가 지표 |
| [08-implementation-roadmap.md](./08-implementation-roadmap.md) | PoC부터 7B 이상 확장까지 개발 계획 |
| [09-risks-and-mitigation.md](./09-risks-and-mitigation.md) | 기술적·연구적·운영적 리스크와 대응 |
| [10-research-positioning.md](./10-research-positioning.md) | 기존 연구와의 관계, 차별화 포인트, 논문 주장 |
| [11-repository-and-engineering-plan.md](./11-repository-and-engineering-plan.md) | 저장소 구조, 모듈, 실험 관리, 코드 품질 기준 |

## 4. 개발 우선순위

### 우선순위 1: 반복이 진짜 정보를 만드는지 검증

모델 크기를 늘리기 전에 다음을 증명해야 한다.

- cycle 2 이후에도 새로운 evidence가 생성되는가?
- 해당 evidence 제거 시 정답 확률이 실제로 하락하는가?
- task ID와 표면 패턴을 제거해도 성능이 유지되는가?
- 동일 추론 FLOPs 기준으로 Dense/Recursive baseline보다 나은가?

### 우선순위 2: 구조적 이득 검증

- persistent typed workspace가 단순 latent token보다 나은가?
- operator sequence가 Mixture-of-LoRAs 가중합보다 나은가?
- propose–verify–commit이 직접 상태 갱신보다 나은가?
- branch search가 단순 best-of-N 또는 Tree of Thoughts보다 효율적인가?

### 우선순위 3: Scaling advantage 검증

- 100M → 300M → 1B → 3B → 7B에서 Dense 대비 격차가 커지는가?
- 모델이 커질수록 추가 reasoning cycle의 한계효용이 증가하는가?
- memory·depth·breadth를 독립적으로 확장할 수 있는가?

## 5. 최종 산출물

1. 100M~300M급 CERPT 연구 프로토타입
2. 재현 가능한 baseline 및 ablation suite
3. 알고리즘·코드·언어 추론 평가 세트
4. evidence causal audit 도구
5. resource allocation profiler
6. 1B급 범용 사전학습 모델
7. 3B~7B scaling 결과
8. 논문 및 기술 보고서
