# GitHub와 Hugging Face에 공개하기

권장 분리는 간단하다.

- GitHub: 소스 코드, 학습 데이터 생성기, 테스트, 문서, 실행 스크립트
- Hugging Face: 모델 weight, tokenizer, image processor, model card

`artifacts/`와 `*.safetensors`는 `.gitignore`에 들어 있으므로 모델 weight를 GitHub에 실수로 올리지 않는다.

## 1. GitHub

GitHub에서 빈 repository를 먼저 만든다. 예를 들어 repository 이름을 `cerpt-planning`으로 만든 뒤 PowerShell에서 실행한다.

```powershell
git init
git branch -M main
git add .
git commit -m "Initial CERPT research prototype"
git remote add origin https://github.com/<GITHUB_USER>/cerpt-planning.git
git push -u origin main
```

이미 `origin`이 있으면 `git remote add` 대신 다음으로 확인한다.

```powershell
git remote -v
git push -u origin main
```

push 전에 반드시 확인한다.

```powershell
git status
git ls-files | Select-String "artifacts|safetensors"
```

두 번째 명령에서 모델 weight가 나오면 GitHub에 올리기 전에 해당 파일을 추적 대상에서 제외해야 한다.

## 2. Hugging Face 로그인

Hugging Face에서 Access Token을 만든다. 모델을 업로드할 권한이 있는 `Write` token이면 된다.

```powershell
huggingface-cli login
```

token을 터미널에 직접 넣고, 코드나 GitHub 파일에는 저장하지 않는다.

## 3. 텍스트 CERPT checkpoint 업로드

```powershell
python scripts/upload_model.py `
  --model-dir artifacts/cerpt-small-1000/latest `
  --repo-id <HF_USER>/cerpt-small-1000
```

비공개로 먼저 올리려면 `--private`를 추가한다.

```powershell
python scripts/upload_model.py `
  --model-dir artifacts/cerpt-small-1000/latest `
  --repo-id <HF_USER>/cerpt-small-1000 `
  --private
```

## 4. 멀티모달 checkpoint 생성 및 업로드

먼저 CLIP vision encoder를 내려받아 기존 CERPT checkpoint에 연결한다.

```powershell
python scripts/prepare_multimodal.py `
  --core-model artifacts/cerpt-small-1000/latest `
  --output-dir artifacts/cerpt-multimodal `
  --vision-model openai/clip-vit-base-patch32
```

그 다음 업로드한다.

```powershell
python scripts/upload_model.py `
  --model-dir artifacts/cerpt-multimodal `
  --repo-id <HF_USER>/cerpt-multimodal `
  --kind multimodal
```

현재 멀티모달 checkpoint는 구조 연결용이다. 이미지·비디오 데이터로 추가학습하기 전에는 시각 응답 성능을 보장하지 않는다. 이 사실은 [MODEL_CARD.md](../MODEL_CARD.md)에 명시한다.

## 5. 다른 PC에서 불러오기

GitHub 코드를 설치한 뒤 custom CERPT class로 불러온다.

```powershell
git clone https://github.com/<GITHUB_USER>/cerpt-planning.git
cd cerpt-planning
pip install -e .
```

텍스트 모델:

```python
from cerpt.models.cerpt import CERPTForConditionalGeneration

model = CERPTForConditionalGeneration.from_pretrained(
    "<HF_USER>/cerpt-small-1000"
)
```

멀티모달 모델:

```python
from cerpt.models.multimodal import CERPTMultimodalForConditionalGeneration

model = CERPTMultimodalForConditionalGeneration.from_pretrained(
    "<HF_USER>/cerpt-multimodal"
)
```

이 프로젝트는 custom `cerpt` architecture이므로, 현재는 `AutoModel`보다 위처럼 CERPT class를 직접 import하는 방식이 가장 확실하다.
