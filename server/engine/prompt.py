from __future__ import annotations
from typing import Dict, List, Optional, Union, TYPE_CHECKING
from diffusers import DiffusionPipeline

try:
    from compel import Compel, ReturnedEmbeddingsType
except ImportError:
    Compel = None
    ReturnedEmbeddingsType = None

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    from .base import BasePipelineEngine

from ..models.inspect import get_tokenizers_and_text_encoders_from_pipeline


def skip_clip_layer(text_encoders: List[nn.Module], n: Optional[int]) -> None:
    """Dynamically truncate CLIP encoder layers for clip_skip >= 1."""
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


def get_embeds_from_pipeline(
    pipeline: Union[DiffusionPipeline, BasePipelineEngine],
    prompt: Union[str, List[str], None],
    negative_prompt: Union[str, List[str], None]
) -> Dict[str, torch.Tensor]:
    """Overcome 77-token CLIP limit and support weighted prompt syntax using Compel."""
    if Compel is None or prompt is None:
        return {}

    target = getattr(pipeline, "pipeline", pipeline)
    tokenizers, text_encoders = get_tokenizers_and_text_encoders_from_pipeline(target)
    tokenizers = [tok for tok in tokenizers if tok]
    text_encoders = [enc for enc in text_encoders if enc]

    if not tokenizers or not text_encoders:
        return {}

    try:
        model_ver = getattr(pipeline, "model_version", None) or getattr(target, "model_version", None)
        class_name = target.__class__.__name__.lower()
        neg_p = negative_prompt or ""
        exec_device = (
            getattr(pipeline, "execution_device", None)
            or getattr(target, "_execution_device", None)
            or getattr(target, "device", None)
        )
        device_str = str(exec_device) if exec_device is not None else None

        if model_ver in ["1", "1.5"] or "stablediffusion" in class_name and "xl" not in class_name:
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

        if model_ver in ["xl", "pony"] or "xl" in class_name:
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
        if hasattr(target, "maybe_free_model_hooks"):
            try:
                target.maybe_free_model_hooks()
            except Exception:
                pass

    return {}
