# 03. 시스템 아키텍처

README에서 바로 볼 수 있는 아키텍처 그림은 다음 두 개로 나뉜다.

- [CERPT 연구 목표 구조](../architecture/cerpt-target-architecture.svg)
- [현재 causal PyTorch 구현](../architecture/cerpt-current-implementation.svg)

두 그림을 분리한 이유는 중요하다. 아래의 일곱 모듈은 최종 연구 목표이고, 현재 `CERPTForCausalLM`은 decoder, workspace seed, shared transition core, operator/verifier heads까지 구현한 scaffold다. 동적 operator 실행, typed slot read/write, certification에 따른 commit/rollback, resource scheduler는 추가 구현과 실험이 필요하다.

## 1. 전체 구성

CERPT는 다음 일곱 모듈로 구성한다.

1. Token Backbone
2. Persistent Typed Workspace
3. Program Controller
4. Shared Transition Core
5. Operator Bank
6. Certification Layer
7. Resource Scheduler

```text
Input tokens
    │
    ▼
Token Backbone ──────────────┐
    │                        │
    ▼                        │
Persistent Typed Workspace   │
    │                        │
    ▼                        │
Program Controller           │
    │                        │
    ▼                        │
Shared Transition Core ◀─ Operator Bank
    │
    ▼
Proposed State + Evidence + Verification Plan
    │
    ▼
Certification Layer
    ├─ accept  → commit
    ├─ reject  → rollback
    └─ uncertain → branch/retry/tool
    │
    ▼
Answer Decoder
```

## 2. Token Backbone

일반적인 decoder-only Transformer 또는 encoder-decoder 구조를 사용할 수 있다. 초기 PoC에서는 decoder-only보다 상태 분석이 쉬운 encoder-style 또는 prefix-LM 구조도 고려한다.

역할:

- 입력 token representation 생성
- workspace가 원문을 조회할 key/value 제공
- 최종 answer token 생성
- 일반 언어 능력 유지

## 3. Persistent Typed Workspace

상태 구성:

```text
Goal
Fact
Constraint
Hypothesis
Counterexample
Evidence
Result
Answer Draft
Invalidated State
```

각 type은 다음을 가진다.

- 고정 또는 동적 slot 수
- 전용 read projection
- 전용 write projection
- validity mask
- provenance metadata
- 생성 cycle index
- verification status

상태 예시:

```yaml
type: evidence
content: latent-vector
source: input-span-14:29
created_by: operator-VERIFY_CONSTRAINT
cycle: 3
status: certified
confidence: 0.84
```

실제 학습 구현에서는 metadata 일부는 discrete tensor 또는 auxiliary label로 표현한다.

## 4. Program Controller

Controller 출력:

```text
operator sequence
read slot mask
write slot mask
memory query
branch count
cycle budget
verification method
halt probability
```

초기 구현에서는 자유로운 프로그램 대신 정해진 template 중 하나를 선택한다.

```text
T0: READ → UPDATE
T1: READ → COMPARE → UPDATE
T2: READ → CHECK → INVALIDATE
T3: BRANCH → VERIFY → SELECT
T4: MEMORY_READ → UPDATE → VERIFY
```

추후 template 내부 operator 조합을 자유화한다.

## 5. Shared Transition Core

동일한 core parameters를 reasoning cycle마다 재사용한다.

입력:

- current workspace
- selected token states
- operator embedding
- cycle embedding
- memory result

출력:

- proposed workspace delta
- proposed evidence
- answer draft delta
- expected utility

Core 구조 예시:

```text
Cross-attention to input tokens
→ Workspace self-attention
→ Shared SwiGLU FFN
→ Active operator low-rank delta
→ Typed write heads
```

## 6. Operator Bank

Operator는 독립적인 full expert가 아니라 저랭크 상태 전이 모듈이다.

```text
O_j(x) = x + B_j(A_jx)
```

초기 operator vocabulary:

- EXTRACT
- BIND
- COMPARE
- COMPOSE
- SUBSTITUTE
- SIMULATE
- CHECK
- SEARCH_COUNTEREXAMPLE
- BRANCH
- MERGE
- INVALIDATE
- WRITE_RESULT
- READ_MEMORY
- REQUEST_TOOL
- HALT

의미를 완전히 하드코딩하지 않고, operator별 mask와 supervision을 통해 기능적 분화를 유도한다.

## 7. Certification Layer

Certification 입력:

- previous state
- proposed state
- evidence delta
- verification plan
- original input
- external checker result

출력:

```text
validity
novelty
causal utility estimate
constraint satisfaction
contradiction risk
grounding score
commit decision
```

검증 방식은 여러 계층으로 구성한다.

1. deterministic checker
2. independent neural verifier
3. causal ablation probe
4. energy/constraint score
5. adversarial evidence discriminator

## 8. Resource Scheduler

각 reasoning segment에 대해 다음을 결정한다.

```text
depth budget
workspace budget
branch budget
tool budget
verification strength
```

초기에는 4개 compute class로 제한한다.

| Class | Depth | Branch | Workspace | 용도 |
|---|---:|---:|---:|---|
| C0 | 0 | 1 | 8 | 일반 생성 |
| C1 | 2 | 1 | 16 | 단순 추론 |
| C2 | 4 | 2 | 32 | 중간 추론 |
| C3 | 8 | 4 | 64 | 복잡한 추론 |

## 9. Autoregressive 실행 방식

매 token마다 recursion을 수행하지 않는다.

```text
질문 입력
→ reasoning segment 시작
→ workspace cycle 수행
→ answer plan 생성
→ 일반 token decoding
→ 검증이 필요한 경계에서 reasoning 재실행
```

경계 예시:

- 답안 시작 전
- 코드 함수 생성 전
- 도구 결과 수신 후
- 모순 감지 시
- 최종 답 확정 전
