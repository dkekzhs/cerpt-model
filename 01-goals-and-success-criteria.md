# 01. 목표와 성공 기준

## 1. 비전

CERPT의 장기 비전은 다음과 같다.

> 저장 파라미터 수를 무작정 키우는 대신, 작은 공유 코어가 중간 상태와 증거를 반복적으로 갱신하고 검증하여 동일한 파라미터에서 더 많은 유효 계산을 수행하도록 한다.

작은 모델에서 파라미터 효율을 증명하고, 같은 구조를 더 큰 모델에 적용했을 때 Dense Transformer보다 높은 성능 증가율을 얻는 것이 최종 목표다.

## 2. 단계별 목표

### 목표 A. 100M~300M 구조 검증

- 동일 파라미터의 Dense Transformer보다 높은 추론 성능
- 동일 inference FLOPs 기준의 Recursive Transformer보다 높은 성능
- 반복 cycle 증가에 따라 검증 가능한 evidence가 지속적으로 생성
- train depth보다 긴 evaluation depth에서 성능이 즉시 붕괴하지 않음
- task ID 제거 및 표면 변환 후에도 구조적 일반화 유지

### 목표 B. 1B 범용 언어 모델

- 일반 LM perplexity를 크게 훼손하지 않음
- 코드, 수학, 다중 문서 reasoning에서 Dense 1B 대비 우위
- 동일 품질에서 더 낮은 weight memory 또는 더 낮은 inference cost
- reasoning segment에서만 추가 cycle을 사용해 실제 latency 관리

### 목표 C. 3B~7B Scaling

- 모델이 커질수록 CERPT의 추가 cycle 효율이 상승
- Dense 대비 성능 격차가 줄지 않고 유지 또는 확대
- operator 수, workspace 크기, branch 폭 증가가 통제 가능한 성능 향상으로 연결
- verifier와 generator의 공모 또는 shortcut이 커지지 않음

### 목표 D. 대형 모델 적용 가능성

- core width, operator bank, workspace, memory, search budget을 독립적으로 확장
- 단일 고정 depth 모델이 아닌 test-time compute scaling 지원
- 외부 도구, 검색, 실행기와 내부 evidence 체계를 일관된 방식으로 연결

## 3. 핵심 성공 지표

### 3.1 파라미터 효율

동일 파라미터 조건:

```text
CERPT(P) > Dense(P)
```

평가:

- validation loss
- algorithmic reasoning accuracy
- code execution accuracy
- multi-hop reasoning F1/EM
- OOD depth extrapolation

### 3.2 계산 효율

동일 inference FLOPs 및 실제 latency 조건:

```text
CERPT(P, C) > Dense(P', C)
```

FLOPs와 latency를 별도로 측정한다. 동적 분기 구조는 FLOPs가 작아도 실제 GPU에서 느릴 수 있기 때문이다.

### 3.3 진짜 progress

각 cycle에서 추가된 상태 변화가 실제 답에 기여해야 한다.

- cycle별 정답 확률 증가
- cycle별 unresolved constraint 감소
- 새 evidence 제거 시 성능 하락
- 잘못된 evidence 주입 시 거부율 증가
- 첫 cycle 이후에도 실질적인 accuracy gain 존재

### 3.4 Scaling advantage

모델 크기 증가에 따라 Dense 대비 개선 폭이 커져야 한다.

```text
Gain_7B > Gain_1B > Gain_300M
```

또한 고정 파라미터에서 추가 computation의 가치가 양수여야 한다.

```text
Q(P, C2) > Q(P, C1), C2 > C1
```

## 4. 비목표

다음은 초기 연구 단계의 목표가 아니다.

- 300M 모델로 즉시 최신 상용 범용 LLM 전체를 능가
- 모든 task에 동일한 operator vocabulary를 강제
- 내부 state를 인간 언어와 완전히 일치시킴
- 자유로운 동적 계산 그래프를 그대로 GPU에서 실행
- benchmark 한두 개의 점수만으로 범용 추론 능력을 주장
- best-of-1000과 single-pass 성능을 혼합해 보고

## 5. 종료 기준

다음 조건 중 다수가 발생하면 구조를 재검토한다.

- 성능 향상의 80% 이상이 첫 cycle에서 발생
- evidence 제거가 결과에 거의 영향을 주지 않음
- task ID 제거 시 성능이 대폭 붕괴
- 동일 FLOPs 기준 baseline보다 지속적으로 열세
- operator 사용이 소수 패턴으로 붕괴
- larger scale에서 verifier 해킹이 증가
- 300M에서 유효하지 않은 구조가 1B 이상에서도 개선될 근거가 없음
