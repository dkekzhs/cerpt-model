# 10. 연구 포지셔닝 및 차별화

## 1. 기존 연구와의 관계

CERPT는 다음 연구 계보와 연결된다.

- Universal/Recursive Transformer: parameter sharing과 반복
- Mixture-of-Recursions: token별 dynamic depth
- HRM/TRM/PTRM: 반복적 상태·답 수정과 trajectory search
- Coconut 계열: latent reasoning
- Titans/ATLAS: test-time memory
- Mixture of LoRAs/MoE: 조건부 연산
- Tree of Thoughts: candidate search
- Process Reward Model: 중간 단계 평가
- CoT faithfulness: trace ablation과 causal audit
- agent transaction frameworks: propose/verify/commit

## 2. 단순 결합으로 보이면 안 되는 이유

다음 주장은 약하다.

> 기존 recursion, memory, verifier, ToT를 하나로 결합했다.

기능 나열은 architecture contribution이 아니다.

## 3. 중심 차별화 주장

> **CERPT는 외부 agent scaffold가 아니라 모델 내부 forward computation에서 persistent typed state를 유지하고, learned non-commutative transition program이 생성한 state delta를 evidence-level certification 이후에만 commit한다.**

핵심 차이는 다음과 같다.

### 외부 agent framework와 차이

- 자연어 action이 아니라 latent state transition
- 모델의 내부 representation을 직접 갱신
- training loss와 inference graph에 통합
- token generation 전의 hidden computation으로 동작

### Tree of Thoughts와 차이

- 자연어 thought node가 아닌 typed latent workspace
- branch의 전체 답이 아니라 state delta 탐색
- heuristic evaluation이 아닌 causal evidence certification
- copy-on-write state로 비용 절감

### CoT faithfulness audit와 차이

- 사후 분석만 하는 것이 아니라 commit 조건으로 활용
- provenance와 state delta가 architecture에 내장
- deterministic checker와 adversarial verifier 결합

### MoE/Mixture of LoRAs와 차이

- expert 가중합이 아니라 순서 있는 state-transition program
- operator는 domain expert보다 reasoning primitive를 목표로 함
- operator의 read/write 영역을 typed state와 연결

## 4. 논문 핵심 기여 후보

### Contribution 1

Persistent typed reasoning workspace.

### Contribution 2

Non-commutative neural transition programs over shared parameters.

### Contribution 3

Evidence-certified transactional state updates with causal auditing.

### Contribution 4

Joint memory-depth-breadth resource allocation.

첫 논문에서는 3개 이하로 축소하는 것이 좋다.

## 5. 권장 첫 논문 범위

첫 논문에서는 대형 LLM과 장기 memory까지 모두 포함하지 않는다.

권장 범위:

- 100M~300M
- algorithmic + code reasoning
- typed workspace
- operator sequence
- transactional certification
- evidence causal audit

제외 가능:

- 장기 writable memory
- 외부 검색
- 대규모 tool ecosystem
- 7B 이상 scaling

## 6. 핵심 논문 주장 예시

> Persistent typed states and evidence-certified transition programs allow a recurrent language model to convert additional inference computation into measurable causal information gain, rather than confidence-only refinement.

## 7. 반드시 피할 주장

- 작은 모델이 범용적으로 모든 대형 모델보다 똑똑하다.
- recursion 자체가 reasoning을 만든다.
- verifier score 상승이 reasoning progress를 증명한다.
- 파라미터 수만 비교해 효율성을 주장한다.
- ToT, CoT faithfulness, transaction framework 계보를 무시한다.
