# 04. 상태 및 프로그램 설계

## 1. 상태 모델

CERPT의 상태는 다음과 같이 표현한다.

```text
S_t = (H_t, W_t, M_t, A_t)
```

- `H_t`: token hidden states
- `W_t`: persistent typed workspace
- `M_t`: episode/long-term memory interface
- `A_t`: answer draft state

## 2. Workspace slot 유형

### Goal

현재 해결해야 할 목표와 하위 목표를 저장한다.

### Fact

입력 또는 도구에서 직접 확인된 사실을 저장한다.

### Constraint

답이 만족해야 하는 규칙, 조건, 타입, 테스트를 저장한다.

### Hypothesis

아직 검증되지 않은 후보 답, 원인, 계획을 저장한다.

### Counterexample

가설을 반박하는 사례와 실패 조건을 저장한다.

### Evidence

가설 또는 상태 변화를 지지하는 검증 가능한 정보를 저장한다.

### Result

실행기, 계산기, 검색, 테스트 결과를 저장한다.

### Answer Draft

현재 출력 후보를 저장한다.

### Invalidated

거부되었거나 반박된 상태를 보존한다.

## 3. 상태 쓰기 연산

### APPEND

새 slot 또는 새 정보를 추가한다.

### REVISE

기존 slot의 일부를 교체한다. revision history는 보존한다.

### INVALIDATE

상태를 삭제하지 않고 사용할 수 없게 표시한다.

### MERGE

호환되는 여러 hypothesis 또는 evidence를 하나로 결합한다.

### SPLIT

하나의 hypothesis를 여러 branch로 분리한다.

## 4. Provenance

각 상태는 어디서 생성되었는지 추적해야 한다.

```text
source type:
- input span
- previous state
- memory retrieval
- tool result
- operator inference
```

필수 필드:

- source IDs
- cycle index
- operator ID
- branch ID
- certification result

이 정보는 추론 감사와 evidence removal에 사용한다.

## 5. Operator 프로그램

Program은 길이 1~4의 operator sequence로 시작한다.

예시:

```text
READ_FACT
READ_CONSTRAINT
COMPARE
WRITE_COUNTEREXAMPLE
```

또는:

```text
READ_HYPOTHESIS
SIMULATE
CHECK_RESULT
INVALIDATE_OR_COMMIT
```

## 6. Non-commutative 설계

같은 operator 집합이라도 순서가 달라지면 결과가 달라야 한다.

```text
EXTRACT → BIND → CHECK
CHECK → EXTRACT → BIND
```

이를 강제하기 위해:

- cycle/position embedding
- operator별 read/write mask
- 순서 permutation negative samples
- intermediate state supervision

을 사용한다.

## 7. Branch 상태

Branch는 workspace 전체 복사가 아니라 copy-on-write 방식으로 관리한다.

```text
Base state
├─ Branch A delta
├─ Branch B delta
└─ Branch C delta
```

각 branch는:

- hypothesis
- program
- evidence
- cost
- verifier score

를 가진다.

## 8. State compaction

workspace가 계속 증가하면 비용이 커지므로 다음 정책을 사용한다.

- invalidated states 압축
- 중복 fact 병합
- 오래된 evidence summary
- low-utility slot eviction
- external episode memory로 이동

단, state compaction 전후의 정답 분포가 지나치게 변하지 않도록 distillation loss를 둔다.

## 9. 초기 구현 제약

PoC에서는 자유도를 제한한다.

- slot type 6개부터 시작
- operator 8~12개
- program length 최대 3
- branch 최대 2
- workspace 16~32 slots
- cycle 최대 4 또는 8

이후 ablation에서 필요한 요소만 확대한다.
