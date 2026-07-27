# 02. 핵심 연구 가설

## 1. 중심 가설

> 범용 추론 성능은 저장 파라미터 수뿐 아니라, 중간 계산 결과를 보존하고 검증 가능한 상태 전이로 재사용하는 능력에 의해 결정된다.

기존 Transformer는 중간 결과를 residual stream 내부에 암묵적으로 저장한다. CERPT는 이를 지속 가능한 typed workspace와 evidence-certified transition으로 분리한다.

## 2. 가설 H1: Persistent state 가설

구조화된 workspace가 단일 latent vector 또는 단순 scratch token보다 반복 추론에서 정보 손실을 줄일 것이다.

### 검증

- typed workspace vs untyped latent tokens
- append/revise/invalidate vs overwrite-only
- state slot 제거 실험
- 긴 cycle에서 정보 회수율 측정

### 반증 조건

- typed workspace의 성능 향상이 parameter increase로 설명됨
- slot type을 무작위화해도 성능 차이가 없음
- 모든 정보가 answer slot 하나에 집중

## 3. 가설 H2: Sequential operator 가설

순서 있는 non-commutative operator program이 expert 가중합이나 단일 FFN update보다 깊은 추론을 더 잘 표현할 것이다.

### 검증

- operator sequence vs top-k LoRA mixture
- 동일 operator 집합의 순서 permutation
- program length별 성능
- program causality probing

### 반증 조건

- operator 순서를 섞어도 성능이 동일
- controller가 항상 동일한 프로그램 출력
- operator bank 제거 후에도 성능 유지

## 4. 가설 H3: Evidence-certified update 가설

직접 상태 갱신보다 propose–verify–commit/rollback 구조가 가짜 progress와 잘못된 가설 누적을 줄일 것이다.

### 검증

- direct update vs transactional update
- 잘못된 evidence 주입
- verifier가 거부한 transition의 실제 오류율
- rollback 후 재탐색 성공률

### 반증 조건

- verifier가 거의 모든 proposal을 통과
- reject가 많지만 성능은 개선되지 않음
- commit 조건을 만족하는 형식적 evidence만 생성

## 5. 가설 H4: Causal evidence 가설

실제 추론에 유용한 evidence는 제거했을 때 정답 분포를 악화시킨다.

### 검증

- cycle delta removal
- evidence replacement
- random evidence insertion
- causal mediation 및 activation patching

### 주요 위험

모델이 evidence에 의존하는 척하도록 학습할 수 있다. 따라서 causal loss는 학습 데이터와 독립된 감사 모델 및 deterministic checker로 교차 검증해야 한다.

## 6. 가설 H5: Resource allocation 가설

문제 유형에 따라 memory, depth, breadth의 최적 조합이 다르며, 이를 공동 제어하면 고정 compute 정책보다 높은 Pareto 효율을 얻을 수 있다.

### 검증

- fixed depth vs adaptive depth
- fixed beam vs adaptive branch
- fixed workspace vs dynamic memory
- 동일 FLOPs 내 자원 조합 비교

## 7. 가설 H6: Scaling complementarity 가설

모델 파라미터가 증가할수록 추가 reasoning cycle의 효용도 증가할 수 있다.

```text
∂²Q / ∂P∂C > 0
```

### 검증

100M, 300M, 1B, 3B, 7B에서 다음을 측정한다.

- cycle 1→2 개선
- cycle 2→4 개선
- cycle 4→8 개선
- 추가 cycle당 정확도 증가/FLOP

### 반증 조건

- 모델 크기 증가와 함께 cycle 효율 감소
- 큰 모델이 첫 pass에서 답을 결정해 recursion이 무의미
- branch와 verifier 비용만 증가

## 8. 가장 중요한 미검증 가정

1. 범용 언어 문제에서 검증 가능한 evidence를 정의할 수 있는가?
2. typed workspace가 hidden representation의 자유도를 지나치게 제한하지 않는가?
3. operator program을 학습할 충분한 supervision을 만들 수 있는가?
4. causal utility가 새로운 shortcut loss가 되지 않는가?
5. 실제 GPU 효율을 유지하면서 동적 computation을 구현할 수 있는가?

이 다섯 질문이 프로젝트의 핵심 연구 리스크다.
