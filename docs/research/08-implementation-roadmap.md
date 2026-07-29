# 08. 구현 로드맵

## Phase 0. 연구 기반 구축

### 목표

재현 가능한 baseline과 실험 환경 구축.

### 작업

- Dense/Recursive/MoE baseline 구현
- 공통 tokenizer와 dataset pipeline
- FLOPs/latency/memory profiler
- experiment tracking
- deterministic checker interface
- evaluation harness

### 산출물

- baseline repository
- benchmark suite
- reproducible configs

## Phase 1. Minimal CERPT PoC

### 모델

- 100M~300M
- workspace 16~32 slots
- operator 8~12개
- cycle 최대 4
- branch 최대 2

### 구현 순서

1. typed workspace
2. shared core recursion
3. fixed operator templates
4. state delta output
5. lightweight certification head
6. commit/rollback
7. evidence removal audit

### 성공 조건

- direct update보다 transactional update가 우수
- first-step dominance 완화
- identity leakage 감소

## Phase 2. Operator Program

### 작업

- variable-length operator sequence
- operator LoRA bank
- read/write mask
- program imitation data
- permutation ablation

### 성공 조건

- operator 순서가 실제 결과에 영향
- top-k mixture보다 높은 OOD reasoning

## Phase 3. Branch Search 및 Resource Scheduler

### 작업

- copy-on-write branch state
- expand/verify/prune
- depth-memory-breadth controller
- compute class batching

### 성공 조건

- 동일 FLOPs에서 fixed beam보다 우수
- premature pruning 통제
- 실제 latency 악화 제한

## Phase 4. Adversarial Certification

### 작업

- fake evidence generator
- independent verifier
- deterministic checker ensemble
- holdout audit pipeline
- verifier hacking benchmark

### 성공 조건

- false evidence rejection 향상
- large-scale에서 faithfulness 저하 억제

## Phase 5. 1B Pretraining

### 작업

- 일반 텍스트/코드/수학 혼합 사전학습
- segment-level reasoning trigger
- latent state distillation
- instruction tuning

### 성공 조건

- Dense 1B 대비 일반 능력 유지 또는 향상
- 코드/수학/다중문서에서 구조적 우위

## Phase 6. Scaling Study

### 모델

- 300M
- 1B
- 3B
- 7B

### 측정

- parameter scaling
- test-time compute scaling
- memory scaling
- branch scaling
- real hardware efficiency

### 핵심 판단

Dense 대비 improvement가 모델 크기와 함께 증가하는가?

## Phase 7. Production-oriented Runtime

### 작업

- operator program compilation
- static compute class batching
- branch cache sharing
- workspace quantization
- verifier distillation
- external tool protocol

## 예상 우선순위

1. evidence가 진짜 유용한지
2. recursion이 cycle 2 이후에도 의미가 있는지
3. identity leakage가 없는지
4. 동일 FLOPs에서 baseline보다 나은지
5. GPU에서 실제로 빠른지
6. scale-up 시 이득이 커지는지
