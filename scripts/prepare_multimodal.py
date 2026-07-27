from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transformers import CLIPImageProcessor

from cerpt.models.multimodal import CERPTMultimodalForConditionalGeneration


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach a Hugging Face vision encoder to a CERPT checkpoint")
    parser.add_argument("--core-model", default="artifacts/cerpt-small-1000/latest")
    parser.add_argument("--output-dir", default="artifacts/cerpt-multimodal")
    parser.add_argument("--vision-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--unfreeze-vision", action="store_true", help="allow vision encoder fine-tuning later")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = CERPTMultimodalForConditionalGeneration.from_cerpt_checkpoint(
        args.core_model,
        vision_model_name=args.vision_model,
        freeze_vision=not args.unfreeze_vision,
    )
    model.save_pretrained(output_dir)
    tokenizer_dir = Path(args.core_model)
    if (tokenizer_dir / "tokenizer.json").exists():
        from transformers import PreTrainedTokenizerFast

        PreTrainedTokenizerFast.from_pretrained(tokenizer_dir).save_pretrained(output_dir)
    # CLIPImageProcessorPil works with Pillow alone, which keeps this small
    # CPU/Windows setup usable even when torchvision is not installed.
    processor = CLIPImageProcessor.from_pretrained(args.vision_model)
    processor.save_pretrained(output_dir / "vision_processor")
    print(f"saved multimodal CERPT to {output_dir.resolve()}")
    print("vision encoder:", args.vision_model)
    print("vision encoder frozen:", not args.unfreeze_vision)


if __name__ == "__main__":
    main()
