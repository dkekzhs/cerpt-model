# 07. 실험 및 평가 계획

## 1. 비교군

필수 baseline:

1. Dense Transformer
2. Parameter-tied Recursive Transformer
3. Universal Transformer
4. Mixture-of-Recursions 계열
5. Sparse MoE
6. Recursive + Mixture of LoRAs
7. TRM-style refinement
8. Coconut-style latent reasoning
9. Tree of Thoughts 또는 search scaffold
10. CERPT without certification
11. CERPT full

## 2. 공정한 비교 조건

### Parameter-matched

저장 파라미터 동일.

### Training-FLOP-matched

총 학습 계산량 동일.

### Inference-FLOP-matched

문제당 추론 FLOPs 동일.

### Latency-matched

동일 하드웨어에서 wall-clock latency 동일.

### Memory-matched

peak VRAM/RAM 동일.

## 3. 평가 영역

### 3.1 Algorithmic reasoning

- graph traversal
- variable binding
- state machine
- arithmetic chains
- constraint satisfaction
- program execution
- training보다 긴 depth

### 3.2 Code

- 실행 결과 예측
- bug localization
- patch generation
- unit test repair
- invariant extraction
- multi-file reasoning

### 3.3 Language reasoning

- multi-hop QA
- contradiction detection
- conditional reasoning
- evidence grounding
- counterfactual reasoning
- document synthesis

### 3.4 General LM

- perplexity
- instruction following
- summarization
- factual QA
- generation quality

## 4. 핵심 실험

### Experiment A. First-step dominance

cycle별 정확도와 log-prob 변화 측정.

```text
cycle 0, 1, 2, 4, 8, 16
```

### Experiment B. Evidence removal

각 cycle의 evidence를 제거하고 성능 변화 측정.

### Experiment C. Identity leakage

- ID 삭제
- 랜덤 ID
- 새로운 ID
- 변수/색/순서 permutation
- 표면 동일, 구조 변경

### Experiment D. False evidence injection

정교한 가짜 evidence를 삽입하고 거부율 측정.

### Experiment E. Operator permutation

동일 operator를 다른 순서로 실행해 non-commutativity 확인.

### Experiment F. Premature pruning

branch search가 정답 경로를 너무 일찍 제거하는 비율 측정.

### Experiment G. Depth extrapolation

train cycle 4, evaluate 8/16/32.

### Experiment H. Memory-depth-breadth Pareto

동일 총비용에서 자원 조합 비교.

### Experiment I. Scaling curve

100M, 300M, 1B, 3B, 7B의 성능·FLOPs·latency 곡선.

## 5. 주요 내부 지표

- cycle별 evidence count
- certified evidence ratio
- causal utility
- unresolved constraint count
- candidate entropy
- branch survival rate
- operator usage entropy
- operator activation CKA
- state slot utilization
- verifier calibration
- rollback rate
- repeat-state rate
- tokens/sec
- HBM traffic
- peak memory

## 6. Ablation 목록

- typed workspace 제거
- provenance 제거
- operator sequence를 가중합으로 교체
- certification 제거
- causal removal loss 제거
- independent verifier 제거
- branch search 제거
- memory scheduler 제거
- invalidated history 제거
- external checker 제거

## 7. 보고 원칙

반드시 구분해서 보고한다.

- single trajectory
- learned branch selection
- best-of-N
- majority vote
- oracle selection

파라미터 수만으로 작은 모델이 큰 모델을 이겼다고 주장하지 않는다. 총 FLOPs, latency, 데이터, augmentation, external tools를 함께 공개한다.

## 8. 1차 성공 기준

100M~300M PoC에서:

- Dense 대비 동일 FLOPs 성능 향상
- cycle 2 이후 유의미한 gain
- evidence removal 시 유의미한 성능 하락
- identity perturbation 후 성능 유지
- false evidence rejection 향상
- 일반 LM 품질 손실 제한
