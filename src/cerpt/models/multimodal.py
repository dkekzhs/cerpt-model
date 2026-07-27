"""Vision front-end for CERPT.

The recursive CERPT core remains modality-agnostic. A Hugging Face CLIP
vision encoder turns an image into evidence tokens and a small projector maps
those tokens into the CERPT hidden space. The core then places them beside
text tokens before building its typed workspace.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Optional

import torch
from torch import Tensor, nn
from transformers import CLIPVisionConfig, CLIPVisionModel, PreTrainedModel, PretrainedConfig

from .cerpt import CERPTConfig, CERPTForConditionalGeneration, CERPTOutput


class CERPTMultimodalConfig(PretrainedConfig):
    model_type = "cerpt-multimodal"

    def __init__(
        self,
        cerpt_config: Optional[dict] = None,
        vision_config: Optional[dict] = None,
        vision_model_name: str = "openai/clip-vit-base-patch32",
        freeze_vision: bool = True,
        temporal_attention_heads: int = 8,
        max_video_frames: int = 32,
        **kwargs,
    ):
        is_encoder_decoder = kwargs.pop("is_encoder_decoder", True)
        super().__init__(is_encoder_decoder=is_encoder_decoder, **kwargs)
        self.cerpt_config = cerpt_config or CERPTConfig().to_dict()
        self.vision_config = vision_config or CLIPVisionConfig().to_dict()
        self.vision_model_name = vision_model_name
        self.freeze_vision = freeze_vision
        self.temporal_attention_heads = temporal_attention_heads
        self.max_video_frames = max_video_frames


class CERPTMultimodalForConditionalGeneration(PreTrainedModel):
    """CERPT with a pretrained/frozen CLIP image evidence encoder."""

    config_class = CERPTMultimodalConfig
    base_model_prefix = "cerpt_multimodal"

    def __init__(self, config: CERPTMultimodalConfig, vision_encoder: Optional[CLIPVisionModel] = None):
        super().__init__(config)
        self.cerpt = CERPTForConditionalGeneration(CERPTConfig(**config.cerpt_config))
        if vision_encoder is None:
            vision_encoder = CLIPVisionModel(CLIPVisionConfig(**config.vision_config))
        self.vision_encoder = vision_encoder
        vision_hidden = int(self.vision_encoder.config.hidden_size)
        self.vision_projector = nn.Sequential(
            nn.Linear(vision_hidden, self.cerpt.config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.cerpt.config.hidden_size),
        )
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=vision_hidden,
            nhead=config.temporal_attention_heads,
            dim_feedforward=vision_hidden * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, 1)
        self.temporal_position_embeddings = nn.Parameter(
            torch.randn(config.max_video_frames, vision_hidden) * 0.02
        )
        self.set_vision_trainable(not config.freeze_vision)
        # Transformers 5.x expects composite PreTrainedModels to publish
        # tied-weight metadata before save/load. Child models already carry
        # their initialized weights, so post_init only initializes the new
        # projector/temporal modules and collects that metadata.
        self.post_init()

    def set_vision_trainable(self, trainable: bool) -> None:
        """Freeze/unfreeze the encoder; projector and CERPT stay trainable."""
        for parameter in self.vision_encoder.parameters():
            parameter.requires_grad = trainable
        self.config.freeze_vision = not trainable

    @classmethod
    def from_cerpt_checkpoint(
        cls,
        cerpt_checkpoint: str,
        vision_model_name: str = "openai/clip-vit-base-patch32",
        freeze_vision: bool = True,
    ) -> "CERPTMultimodalForConditionalGeneration":
        """Attach a pretrained Hugging Face vision encoder to a CERPT checkpoint."""
        core = CERPTForConditionalGeneration.from_pretrained(cerpt_checkpoint)
        vision = CLIPVisionModel.from_pretrained(vision_model_name)
        config = CERPTMultimodalConfig(
            cerpt_config=core.config.to_dict(),
            vision_config=vision.config.to_dict(),
            vision_model_name=vision_model_name,
            freeze_vision=freeze_vision,
        )
        model = cls(config, vision_encoder=vision)
        model.cerpt.load_state_dict(core.state_dict())
        return model

    def encode_media(self, pixel_values: Tensor) -> Tensor:
        """Encode an image ``[B, 3, H, W]`` or video ``[B, T, 3, H, W]``.

        For video, CLIP encodes each frame, while a lightweight temporal
        Transformer summarizes frame order into one additional evidence token.
        This keeps the CERPT core unchanged and makes temporal reasoning an
        explicit front-end concern.
        """
        is_video = pixel_values.ndim == 5
        if pixel_values.ndim == 4:
            batch_visual = pixel_values
            frames = 1
        elif pixel_values.ndim == 5:
            batch, frames = pixel_values.shape[:2]
            if frames > self.config.max_video_frames:
                raise ValueError("video has more frames than max_video_frames")
            batch_visual = pixel_values.reshape(batch * frames, *pixel_values.shape[2:])
        else:
            raise ValueError("pixel_values must have shape [B, 3, H, W] or [B, T, 3, H, W]")
        context = torch.no_grad() if self.config.freeze_vision else nullcontext()
        with context:
            visual_tokens = self.vision_encoder(pixel_values=batch_visual).last_hidden_state
        if is_video:
            batch = pixel_values.size(0)
            visual_tokens = visual_tokens.reshape(batch, frames, *visual_tokens.shape[1:])
            frame_summary = visual_tokens[:, :, 0, :] + self.temporal_position_embeddings[:frames].unsqueeze(0)
            temporal = self.temporal_encoder(frame_summary)
            temporal_token = temporal.mean(dim=1, keepdim=True)
            patch_tokens = visual_tokens[:, :, 1:, :].mean(dim=1)
            visual_tokens = torch.cat([temporal_token, patch_tokens], dim=1)
        return self.vision_projector(visual_tokens)

    def encode_image(self, pixel_values: Tensor) -> Tensor:
        """Backward-compatible alias for image-only callers."""
        return self.encode_media(pixel_values)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        pixel_values: Optional[Tensor] = None,
        vision_features: Optional[Tensor] = None,
        vision_attention_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> CERPTOutput:
        if pixel_values is not None and vision_features is not None:
            raise ValueError("pass either pixel_values or vision_features, not both")
        if pixel_values is not None:
            vision_features = self.encode_media(pixel_values)
        return self.cerpt(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vision_features=vision_features,
            vision_attention_mask=vision_attention_mask,
            **kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        pixel_values: Optional[Tensor] = None,
        vision_features: Optional[Tensor] = None,
        vision_attention_mask: Optional[Tensor] = None,
        max_new_tokens: int = 64,
        **kwargs,
    ) -> Tensor:
        if pixel_values is not None and vision_features is not None:
            raise ValueError("pass either pixel_values or vision_features, not both")
        if pixel_values is not None:
            vision_features = self.encode_media(pixel_values)
        return self.cerpt.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vision_features=vision_features,
            vision_attention_mask=vision_attention_mask,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
