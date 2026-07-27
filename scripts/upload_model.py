from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi
from transformers import AutoTokenizer, CLIPImageProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cerpt.models.cerpt import CERPTForConditionalGeneration
from cerpt.models.multimodal import CERPTMultimodalForConditionalGeneration


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a CERPT checkpoint to the Hugging Face Hub")
    parser.add_argument("--model-dir", default="artifacts/cerpt-small")
    parser.add_argument("--repo-id", required=True, help="for example: your-user/cerpt-small")
    parser.add_argument("--kind", choices=["auto", "text", "multimodal"], default="auto")
    parser.add_argument("--model-card", default="MODEL_CARD.md")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    model_dir = Path(args.model_dir)
    config_data = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    kind = args.kind
    if kind == "auto":
        kind = "multimodal" if config_data.get("model_type") == "cerpt-multimodal" else "text"

    if kind == "multimodal":
        model = CERPTMultimodalForConditionalGeneration.from_pretrained(model_dir)
    else:
        model = CERPTForConditionalGeneration.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    model.push_to_hub(args.repo_id, private=args.private)
    tokenizer.push_to_hub(args.repo_id, private=args.private)
    processor_dir = model_dir / "vision_processor"
    if kind == "multimodal" and processor_dir.exists():
        CLIPImageProcessor.from_pretrained(processor_dir).push_to_hub(args.repo_id, private=args.private)
    model_card = Path(args.model_card)
    if model_card.exists():
        api.upload_file(
            path_or_fileobj=str(model_card),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="model",
        )
    print(f"uploaded https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
