# 05. Certification 및 검증 설계

## 1. 목적

Certification Layer의 목적은 모델이 생성한 상태 변화를 믿는 것이 아니라, 해당 변화가 실제로 유효하고 새로운 정보를 제공하는지 확인하는 것이다.

핵심 흐름:

```text
PROPOSE
→ VERIFY
→ COMMIT / ROLLBACK / BRANCH
```

## 2. Proposal 출력 형식

각 proposal은 다음을 포함한다.

```text
state delta
claimed evidence
source/provenance
verification method
expected effect
estimated cost
```

예:

```yaml
operation: INVALIDATE_HYPOTHESIS
hypothesis_id: H2
evidence: "실행 결과에서 H2가 예측한 값이 나타나지 않음"
verification: tool_result_match
expected_effect: reduce_candidate_count
```

## 3. 검증 계층

### 3.1 Deterministic verification

가능한 경우 최우선 사용.

- 코드 실행 및 테스트
- 수식 계산
- SAT/constraint solver
- SQL parser/execution
- 타입 검사
- JSON/XML schema
- 정규식 및 형식 검증

### 3.2 Independent neural verifier

Generator와 완전히 동일한 head를 쓰지 않는다.

권장:

- 일부 독립 파라미터
- 다른 데이터 분포
- frozen backbone probe와 독립 verifier 비교
- 여러 verifier disagreement 측정

### 3.3 Causal removal test

새 evidence 또는 state delta를 제거한 뒤 예측을 다시 측정한다.

```text
full state prediction
vs
ablated state prediction
```

단일 샘플 제거뿐 아니라 다음을 수행한다.

- evidence 제거
- evidence 교체
- provenance 제거
- 순서 교체
- 관련 없는 evidence 삽입

### 3.4 Energy/constraint verification

상태의 미해결 조건, 모순, 근거 부족을 energy로 표현한다.

```text
E = unresolved + contradiction + unsupported + invalid
```

좋은 transition은 일반적으로 energy를 낮춰야 한다.

### 3.5 Adversarial evidence verification

별도 adversary가 다음 가짜 evidence를 생성한다.

- 결론을 반복한 문장
- 관련 있지만 인과적이지 않은 사실
- 사실과 미세하게 충돌하는 근거
- 출처가 없는 주장
- 잘못된 계산 결과
- task ID 기반 shortcut

Certification은 이를 거부하도록 학습한다.

## 4. Commit 정책

proposal은 다음 조건 중 설정된 조합을 만족해야 한다.

- deterministic checker pass
- verifier threshold pass
- causal utility positive
- energy reduction
- novelty threshold
- no new contradiction

초기에는 hard rule과 learned score를 혼합한다.

```text
commit = hard_constraints_pass
         AND verifier_score > τ1
         AND causal_utility > τ2
```

## 5. Rollback 정책

다음 경우 rollback한다.

- evidence invalid
- contradiction 증가
- candidate space가 잘못된 방향으로 축소
- verifier disagreement 큼
- tool result와 불일치
- 같은 상태 반복

rollback된 proposal은 Invalidated slot에 요약하여 같은 실패를 반복하지 않게 한다.

## 6. Uncertain 처리

즉시 accept/reject가 어려운 경우:

- stronger verifier 호출
- 다른 branch 생성
- tool 요청
- memory 추가 조회
- compute budget 확대

## 7. Verifier 해킹 방지

### 방법 A. Holdout verifier

훈련에 사용하지 않은 verifier로 audit한다.

### 방법 B. Cross-model audit

다른 architecture/seed의 verifier와 비교한다.

### 방법 C. Hidden-state mediation

표면 evidence text가 아니라 내부 state delta가 실제 answer head에 미치는 영향을 분석한다.

### 방법 D. Randomized certification

훈련 중 일부 검증 규칙과 마스킹 패턴을 무작위화해 고정된 검사기 공략을 어렵게 한다.

### 방법 E. External truth anchors

실행 가능한 task에서는 외부 결과를 최종 기준으로 사용한다.

## 8. Certification 지표

- proposal acceptance precision
- invalid proposal rejection recall
- causal utility calibration
- evidence faithfulness
- verifier disagreement rate
- rollback recovery rate
- false evidence rejection rate
- correct path premature pruning rate
