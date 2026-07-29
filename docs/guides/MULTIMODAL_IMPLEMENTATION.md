# CERPT 멀티모달 구현

이번 단계에서 CERPT에 Hugging Face vision encoder 연결부를 추가했다.

```text
image  [B, 3, H, W]       ─┐
video  [B, T, 3, H, W]    ─┴─> CLIP Vision Encoder
                                  ↓
                         Evidence Projector
                                  ↓
text tokens ───────────────> CERPT memory
                                  ↓
                    typed workspace → recursive cycles
                                  ↓
                         text / action output
```

## 구현된 것

- `CERPTForConditionalGeneration`은 `vision_features`를 텍스트 memory 옆에 추가한다.
- `CERPTMultimodalForConditionalGeneration`은 `CLIPVisionModel`을 포함한다.
- 이미지 입력은 `[batch, 3, height, width]`를 받는다.
- 비디오 입력은 `[batch, frames, 3, height, width]`를 받는다.
- 비디오는 프레임별 CLIP feature를 만든 뒤 temporal Transformer로 시간 정보를 요약하고, 그 결과를 CERPT evidence token으로 넣는다.
- 기본값은 vision encoder frozen이다. projector, CERPT core, task/NPC adapter부터 학습하는 8GB GPU 친화적인 방식이다.
- 기존 텍스트 CERPT checkpoint는 변경 없이 계속 사용할 수 있다.

## 모델 생성

처음 한 번은 Hugging Face에서 CLIP vision encoder와 image processor를 내려받는다.

```powershell
python scripts/prepare_multimodal.py `
  --core-model artifacts/cerpt-small-1000/latest `
  --output-dir artifacts/cerpt-multimodal `
  --vision-model openai/clip-vit-base-patch32
```

이미지 질문:

```powershell
python scripts/chat_multimodal.py `
  --model-dir artifacts/cerpt-multimodal `
  --image .\examples\sample.jpg `
  --question "이 이미지에서 확인되는 객체를 설명해줘."
```

## 중요한 현재 한계

이 단계는 **입력 경로와 모델 구조를 만든 것**이다. 현재 CERPT checkpoint는 텍스트 합성 데이터로만 학습되었으므로, 위 모델을 만들었다고 곧바로 이미지 내용을 정확히 설명하지는 않는다. 실제 이미지 응답을 얻으려면 이미지-질문-답변 데이터로 projector와 CERPT 출력부를 추가 학습해야 한다.

권장 순서는 다음과 같다.

1. CLIP vision encoder frozen
2. evidence projector + CERPT task adapter 학습
3. 성능이 부족할 때 vision encoder에 LoRA 적용
4. 충분한 데이터와 GPU가 있을 때만 vision encoder 일부 또는 전체를 공동 학습

즉, vision encoder는 멀티모달 CERPT에 **포함되는 부품**이지만, CERPT의 핵심은 encoder 자체가 아니라 encoder가 만든 evidence를 typed workspace에 넣고 recursive propose/verify/commit 흐름으로 처리하는 구조다.
