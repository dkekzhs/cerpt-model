from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cerpt.models.multimodal import CERPTMultimodalForConditionalGeneration


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask CERPT about an image")
    parser.add_argument("--model-dir", default="artifacts/cerpt-multimodal")
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", default="What is in this image?")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
    from transformers import CLIPImageProcessor

    processor = CLIPImageProcessor.from_pretrained(model_dir / "vision_processor")
    model = CERPTMultimodalForConditionalGeneration.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    encoded_text = tokenizer(args.question, return_tensors="pt")
    image = Image.open(args.image).convert("RGB")
    encoded_image = processor(images=image, return_tensors="pt")
    inputs = {**encoded_text, **encoded_image}
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    print(tokenizer.decode(generated[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
