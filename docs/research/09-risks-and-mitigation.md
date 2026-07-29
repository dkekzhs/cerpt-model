# 09. 리스크 및 대응 전략

## 1. 가짜 Progress

### 위험

첫 cycle에서 답이 결정되고 이후 cycle은 confidence만 높일 수 있다.

### 대응

- cycle별 정확도 공개
- evidence removal
- state mediation
- unresolved constraint 감소 측정
- outcome-independent verifier 사용

## 2. Verifier 해킹

### 위험

모델이 실제 reasoning 대신 verifier가 좋아하는 evidence 형식을 학습한다.

### 대응

- holdout verifier
- adversarial evidence generator
- randomized certification
- deterministic truth anchor
- cross-model audit

## 3. Task Identity Leakage

### 위험

문제 ID나 템플릿을 암기한다.

### 대응

- ID randomization
- isomorphic remapping
- cross-family holdout
- representation probe
- identity contrastive loss

## 4. Workspace 붕괴

### 위험

모든 정보가 특정 slot에 몰리거나 type 구분이 무의미해진다.

### 대응

- type-specific masks
- slot permutation test
- capacity constraints
- causal slot ablation
- slot diversity loss

## 5. Operator 붕괴

### 위험

controller가 항상 동일한 operator를 사용한다.

### 대응

- functional diversity
- negative program training
- task-conditioned routing audit
- operator CKA 분석
- 강제 균등 사용 대신 causal specialization 평가

## 6. Premature Pruning

### 위험

verifier가 정답 branch를 일찍 제거한다.

### 대응

- uncertainty-aware pruning
- minimum branch retention
- delayed commit
- rollback cache
- oracle-path analysis

## 7. Compute 폭증

### 위험

branch, verifier, memory가 추가되어 작은 모델의 장점을 잃는다.

### 대응

- compute class 제한
- segment-level reasoning
- verifier cascade
- cheap probe first, expensive checker later
- copy-on-write state

## 8. GPU 비효율

### 위험

동적 program과 branch로 batching이 깨진다.

### 대응

- fixed execution template
- program compilation
- 동일 class batching
- fused operator kernel
- branch state sharing

## 9. 일반 LM 품질 하락

### 위험

reasoning 구조에 집중해 자연어 생성 품질이 저하된다.

### 대응

- LM loss 유지
- reasoning trigger 분리
- 일반 token path bypass
- backbone distillation
- 혼합 데이터 비율 조절

## 10. Scale-up 실패

### 위험

작은 모델에서만 유효하고 큰 모델에서는 first-pass가 너무 강해 recursion이 무의미해질 수 있다.

### 대응

- scaling complementarity를 초기부터 측정
- larger model에서 harder task 및 적응형 budget
- verifier와 workspace capacity도 함께 확장
- 모델 크기별 optimal cycle 정책 별도 학습

## 11. 연구 신규성 부족

### 위험

기존 recursion, memory, ToT, verifier의 조합으로 평가될 수 있다.

### 대응

논문의 중심을 다음 하나로 집중한다.

> 모델 내부의 persistent typed state에 대해, 순서 있는 neural transition program을 실행하고, evidence의 인과적 유용성이 검증될 때만 state를 commit하는 architecture-level mechanism.

## 12. 데이터 생성 비용

### 위험

state, evidence, program trace 데이터 구축 비용이 큼.

### 대응

- 실행 가능한 synthetic data
- teacher trace 자동 검증
- weak supervision
- self-play
- trace compression
- active learning으로 어려운 사례 우선 생성
