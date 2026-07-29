# 11. 저장소 및 엔지니어링 계획

## 1. 권장 저장소 구조

```text
cerpt/
├─ README.md
├─ pyproject.toml
├─ configs/
│  ├─ model/
│  ├─ training/
│  ├─ data/
│  └─ evaluation/
├─ src/cerpt/
│  ├─ models/
│  │  ├─ backbone.py
│  │  ├─ workspace.py
│  │  ├─ controller.py
│  │  ├─ transition_core.py
│  │  ├─ operator_bank.py
│  │  ├─ verifier.py
│  │  └─ scheduler.py
│  ├─ state/
│  │  ├─ schema.py
│  │  ├─ provenance.py
│  │  ├─ transaction.py
│  │  └─ branch_store.py
│  ├─ training/
│  │  ├─ losses.py
│  │  ├─ curriculum.py
│  │  ├─ adversary.py
│  │  └─ trainer.py
│  ├─ data/
│  │  ├─ state_trace.py
│  │  ├─ synthetic.py
│  │  └─ transforms.py
│  ├─ verification/
│  │  ├─ deterministic.py
│  │  ├─ causal_ablation.py
│  │  ├─ energy.py
│  │  └─ audit.py
│  └─ runtime/
│     ├─ compiler.py
│     ├─ batching.py
│     └─ profiler.py
├─ experiments/
├─ tests/
├─ scripts/
└─ docs/
```

## 2. 핵심 인터페이스

### Workspace

```python
class Workspace:
    def read(self, slot_type, mask=None): ...
    def propose(self, delta): ...
    def commit(self, transaction_id): ...
    def rollback(self, transaction_id): ...
    def invalidate(self, slot_ids): ...
```

### Operator

```python
class Operator:
    def forward(self, state, read_mask, write_mask): ...
```

### Certification

```python
class CertificationResult:
    valid: bool
    causal_utility: float
    energy_delta: float
    confidence: float
    reasons: dict
```

## 3. 실험 관리

각 실험은 다음을 고정 기록한다.

- git commit
- dataset version
- seed
- model config
- parameter count
- training FLOPs
- inference FLOPs
- peak memory
- latency
- hardware
- checkpoint hash

## 4. 테스트 전략

### Unit tests

- state append/revise/invalidate
- transaction commit/rollback
- provenance tracking
- operator read/write mask
- branch copy-on-write

### Integration tests

- proposal → certification → commit
- rejection → rollback → branch
- evidence removal audit
- deterministic checker 연결

### Regression tests

- cycle별 성능
- identity perturbation
- false evidence rejection
- operator usage
- memory leak 및 GPU memory

## 5. 코드 품질 기준

- type hint 필수
- dataclass/schema 사용
- state mutation 최소화
- deterministic seed 모드
- config-driven experiments
- baseline과 CERPT가 동일 training harness 사용
- profiler를 별도 구현하지 말고 forward graph에 통합

## 6. 우선 구현 순서

1. baseline transformer
2. workspace tensor와 slot schema
3. shared transition loop
4. state delta transaction
5. simple verifier
6. causal ablation tooling
7. operator bank
8. branch search
9. resource scheduler
10. optimized runtime
