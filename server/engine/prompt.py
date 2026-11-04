from __future__ import annotations
from typing import Dict, List, Optional, Union
from diffusers import DiffusionPipeline

try:
    from compel import Compel, ReturnedEmbeddingsType
except ImportError:
    Compel = None
    ReturnedEmbeddingsType = None

import torch
import torch.nn as nn

from ..models.inspect import get_tokenizers_and_text_encoders_from_pipeline


def skip_clip_layer(text_encoders: List[nn.Module], n: Optional[int]) -> None:
    """Truncate CLIP encoder layers when clip_skip is 1 or more."""
    if n is not None and not (isinstance(n, int) and n >= 0):
        raise ValueError("clip_skip must be an integer >= 0.")

    valid_encoders = [te for te in text_encoders if te]
    for text_encoder in valid_encoders:
        encoder = (
            getattr(getattr(text_encoder, "text_model", None), "encoder", None)
            or getattr(text_encoder, "encoder", None)
        )
        if encoder is None or not hasattr(encoder, "layers"):
            continue

        if isinstance(n, int) and n >= 1:
            if not hasattr(encoder, "backup_layers"):
                encoder.backup_layers = encoder.layers
            encoder.layers = encoder.backup_layers[:-n]
        elif n is None or n == 0:
            if hasattr(encoder, "backup_layers"):
                encoder.layers = encoder.backup_layers


def _sync_compel_device(compel: object, device_str: Optional[str]) -> None:
    if not device_str:
        return
    cond_provider = getattr(compel, "conditioning_provider", None)
    if cond_provider is None:
        return
    providers = getattr(cond_provider, "embedding_providers", None) or [cond_provider]
    for provider in providers:
        provider.device = device_str


def encode_clip_prompt(
    pipeline: DiffusionPipeline,
    model_version: str,
    prompt: Union[str, List[str], None],
    negative_prompt: Union[str, List[str], None],
    device: Optional[torch.device],
) -> Dict[str, torch.Tensor]:
    """Encode weighted prompts past the 77-token CLIP limit."""
    if Compel is None or prompt is None:
        return {}

    tokenizers, text_encoders = get_tokenizers_and_text_encoders_from_pipeline(pipeline)
    tokenizers = [tok for tok in tokenizers if tok]
    text_encoders = [enc for enc in text_encoders if enc]
    if not tokenizers or not text_encoders:
        return {}

    try:
        neg_p = negative_prompt or ""
        device_str = str(device) if device is not None else None
        class_name = pipeline.__class__.__name__.lower()
        version = (model_version or "").lower()

        if version in ("1", "1.5", "2") or ("stablediffusion" in class_name and "xl" not in class_name and "3" not in class_name):
            compel = Compel(
                tokenizers,
                text_encoders,
                truncate_long_prompts=False,
                device=device_str,
            )
            _sync_compel_device(compel, device_str)
            conditioning = compel(prompt)
            neg_conditioning = compel(neg_p)
            [conditioning, neg_conditioning] = compel.pad_conditioning_tensors_to_same_length(
                [conditioning, neg_conditioning]
            )
            return {"prompt_embeds": conditioning, "negative_prompt_embeds": neg_conditioning}

        if version in ("xl", "pony") or "xl" in class_name:
            compel = Compel(
                tokenizers,
                text_encoders,
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
                truncate_long_prompts=False,
                device=device_str,
            )
            _sync_compel_device(compel, device_str)
            conditioning, pooled = compel(prompt)
            neg_conditioning, neg_pooled = compel(neg_p)
            [conditioning, neg_conditioning] = compel.pad_conditioning_tensors_to_same_length(
                [conditioning, neg_conditioning]
            )
            return {
                "prompt_embeds": conditioning,
                "negative_prompt_embeds": neg_conditioning,
                "pooled_prompt_embeds": pooled,
                "negative_pooled_prompt_embeds": neg_pooled,
            }
    except Exception:
        pass
    finally:
        if hasattr(pipeline, "maybe_free_model_hooks"):
            try:
                pipeline.maybe_free_model_hooks()
            except Exception:
                pass

    return {}
