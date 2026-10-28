from __future__ import annotations
from typing import List, Tuple, TYPE_CHECKING
from diffusers import DiffusionPipeline
from transformers import PreTrainedTokenizerBase

if TYPE_CHECKING:
    import torch.nn as nn


def infer_model_version(file_path: str) -> str:
    """Infer Diffusers model architecture type from checkpoint state dict."""
    from diffusers.loaders.single_file_utils import load_state_dict, infer_diffusers_model_type
    checkpoint = load_state_dict(file_path)
    return str(infer_diffusers_model_type(checkpoint))


def get_tokenizers_and_text_encoders_from_pipeline(
    pipeline: DiffusionPipeline
) -> Tuple[List[PreTrainedTokenizerBase], List[nn.Module]]:
    """Extract all active tokenizers and text encoders available on a pipeline instance."""
    tokenizers: List[PreTrainedTokenizerBase] = []
    text_encoders: List[nn.Module] = []
    tokenizer_names = ["tokenizer", "tokenizer_2", "tokenizer_3"]
    text_encoder_names = ["text_encoder", "text_encoder_2", "text_encoder_3"]
    for i, item in enumerate(tokenizer_names):
        if hasattr(pipeline, item):
            tok = getattr(pipeline, item)
            if tok is not None:
                tokenizers.append(tok)
            enc = getattr(pipeline, text_encoder_names[i], None)
            if enc is not None:
                text_encoders.append(enc)
    return tokenizers, text_encoders
