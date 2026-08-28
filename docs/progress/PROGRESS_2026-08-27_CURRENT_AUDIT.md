# 현재 상태 감사 — 2026-08-27

## 결론

2026-07-29의 마지막 작업은 기존 Encoder–Decoder PoC와 별도로 자체 Decoder-only CERPT causal scaffold를 추가하고, 작은 한국어 base/SFT checkpoint와 3B 설정, Mac MPS 학습 경로를 만든 단계까지 완료했다.

현재 저장소는 아직 vLLM이나 Ollama에서 바로 실행되는 production LLM은 아니다. 감사에서 발견한 full-sequence workspace 누수는 같은 날 source에서 제거했고, Llama 계열 decoder와 causal in-stream workspace token으로 교체했다. Hugging Face KV cache까지 연결했지만 vLLM custom config 등록·실제 serving 검증과 GGUF 변환은 남아 있다. 수정 전 checkpoint는 새 구조로 바뀌지 않으므로 정상 causal LLM으로 평가하거나 이어 학습할 수 없다.

## 코드와 checkpoint로 확인한 범위

- 수정 전 `CERPTForCausalLM`: causal decoder, dense global workspace, shared transition core, operator/verifier prediction head
- 수정 후 `CERPTForCausalLM`: RoPE/RMSNorm/SwiGLU/GQA decoder, causal workspace token, KV cache, operator/verifier prediction head
- base checkpoint: hidden 64, decoder 1 layer, 10 epochs, 기록된 validation loss `1.653888`; future leakage로 성능 지표 무효
- chat SFT checkpoint: 5 epochs, 기록된 validation loss `0.109850`; future leakage로 성능 지표 무효
- 3B preset: 28 layers, 24 query/8 KV heads, `3,020,101,641` parameters로 계산되지만 weight는 학습하지 않음
- 데이터: `korean_basic_v6` 11,000건, train 8,803 / validation 1,098 / test 1,099
- 외부 산수 verifier: CLI에서 모델 밖의 결정론적 계산기로 동작

## 아직 구현되지 않은 핵심

- typed slot과 operator별 read/write 규칙
- operator 선택이 실제 transition을 바꾸는 program execution
- verifier 결과에 따른 commit, rollback, branch
- baseline/ablation을 통한 CERPT 고유 성능 증명
- 대규모 causal pretraining과 범용 언어 능력
- continuous batching과 vLLM backend/package 검증
- GGUF 변환과 Ollama/llama.cpp 실행
- multimodal alignment 학습과 held-out 평가

## 이번 감사의 재현 결과

- 수정 전 감사 `pytest -q`: `8 passed`; 구조 수정 후 전체 suite: `17 passed`
- `audit_korean_basic.py --data-dir data/korean_basic_v6`: `quality_ok: true`, 산수 8,000/8,000 검증
- 산수 CLI: `37 - 8`, `× 4`에 `116` 반환. 모델 생성이 아니라 외부 deterministic verifier 경로임
- chat CLI: `안녕`에는 `같이 이야기해요. 요즘 관심 있는 주제가 있나요?`를 생성했지만, `이름이 뭐야?`에는 `질문에.`를 생성해 일반 대화 품질이 아직 불안정함
- causal invariance: 동일한 4-token prefix 뒤의 suffix만 변경했을 때 random model의 prefix logits가 최대 `0.001315`, 학습된 v6 SFT checkpoint는 최대 `3.535420` 변함. 정상 autoregressive LM에서는 허용되지 않음
- root-cause isolation: workspace 적용 전 decoder hidden의 동일 prefix 차이는 정확히 `0.0`; 기본 self-attention causal mask가 아니라 workspace 결합 단계의 결함임
- padding invariance: 동일한 4개 real token을 길이 8로 padding했을 때 real-token logits가 최대 `6.008883` 변함. workspace 평균이 padding mask를 사용하지 않기 때문임

이 원인은 decoder causal mask 뒤에서 `_workspace()`가 padding mask 없이 전체 sequence hidden을 평균 내고 그 summary를 모든 위치에 더하기 때문이다. response token이 prompt와 앞선 response 위치의 예측에 역으로 섞이고, batch padding도 답을 바꾼다. 현재 checkpoint와 loss는 폐기 대상으로 보고, causality-safe workspace 구조 이후 처음부터 재학습해야 한다.

## 감사 직후 적용한 구조 수정

- `LlamaForCausalLM` core로 교체: RoPE, RMSNorm, SwiGLU, GQA, 표준 causal attention 사용
- sequence를 `[BOS] + prompt + cycle×slot workspace tokens + response + [EOS]`로 구성
- prompt/workspace loss를 mask하고 response-only LM loss 적용
- workspace token hidden을 cycle/slot tensor로 추출해 operator/verifier 보조 head에 연결
- full-sequence pooling, learned absolute position embedding, 별도 global transition core 제거
- Hugging Face `past_key_values`를 보존해 cached decode 지원
- 새 회귀 결과: suffix 변경 prefix-logit 불변, masked-padding 불변, cache/full-prefix logit 일치

이 수정은 모델 계산 그래프를 정상 autoregressive LM으로 만든 것이다. operator의 동적 실행, typed semantic slot, verifier commit/rollback이 완성되었다는 뜻은 아니다.

## 새 한국어 학습 후보

- 기존 `korean_basic_v6`: 11,000 pairs
- `songys/Chatbot_data`: 11,876 pairs, 일상/이별/사랑 label을 가진 인공 데이터
- 로컬 `대화데이터_오피스(JSON).zip`: 46,414 pairs, daily/email/schedule/meeting
- 중복 제거 전 합계: 69,290 pairs

Chatbot_data의 감정 분포와 오피스 데이터의 업무 분포가 뚜렷하므로 데이터가 많아졌다고 편향이 사라지지는 않는다. 3B 범용 pretraining에는 여전히 매우 작다. 다음 학습 전에 source별 sampling, dedup, 개인정보/라이선스, held-out bias 평가를 별도 gate로 둔다.

현재 core는 Llama-compatible projection과 `past_key_values`를 사용하지만 config는 `cerpt_causal`이고 auxiliary head와 workspace prompt template이 추가된다. 따라서 vLLM에서 실제 config 등록·weight load·continuous batching을 검증하기 전에는 `vllm serve` 호환을 주장하지 않는다. Ollama의 Safetensors import도 지원 아키텍처에 한정되므로, CERPT에는 GGUF converter와 workspace prompt template 또는 runtime 확장이 필요하다.

## Qwen3.8-Flash-Next와의 비교

Qwen3.8-Flash-Next는 2026-08-26 공개된 multimodal MoE 모델이며, Gated DeltaNet과 Qwen Sparse Attention, 4-branch Gated Residual, N-gram Embedding을 결합한다. 공식 공개 자료는 125B main parameters와 51B N-gram embedding, token당 6B active parameters를 명시하고 Transformers, llama.cpp, SGLang, vLLM 실행 경로를 함께 제공한다.

CERPT와 방향성이 닮은 지점은 residual/state 흐름을 단일 암묵적 stream에만 맡기지 않고 별도 state와 gate/read-write 구조로 확장하려는 점, 그리고 전체 capacity와 token당 실제 compute를 분리하려는 점이다. 하지만 Qwen의 gate는 residual 정보 흐름과 효율을 제어하고, CERPT의 목표 gate는 evidence 검증을 통과한 reasoning state만 commit하는 의미론적 transaction이다. 직접 같은 구조는 아니다.

이 사례가 CERPT에 주는 의미는 “architecture-level state routing과 active compute 최적화가 실제 대형 모델에서도 중요한 연구 방향”이라는 외부 근거다. 반대로 CERPT의 typed workspace, operator program, certification 가설을 검증해 주지는 않는다. CERPT는 동일 backbone baseline, operator/workspace/verifier ablation, negative verifier data, serving-aware KV cache 구현으로 자체 주장을 입증해야 한다.

공식 자료:

- https://github.com/QwenLM/Qwen3.8-Flash-Next
- https://qwen.ai/blog?id=qwen3.8-flash-next
- https://docs.vllm.ai/en/stable/models/supported_models/
- https://docs.ollama.com/import
