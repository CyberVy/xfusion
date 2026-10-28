from typing import List, Tuple, Any


def infer_model_version(file_path: str) -> str:
    """Infer Diffusers model architecture type from checkpoint state dict."""
    from diffusers.loaders.single_file_utils import load_state_dict, infer_diffusers_model_type
    checkpoint = load_state_dict(file_path)
    return infer_diffusers_model_type(checkpoint)


def get_tokenizers_and_text_encoders_from_pipeline(pipeline: Any) -> Tuple[List[Any], List[Any]]:
    """Extract all active tokenizers and text encoders available on a pipeline instance."""
    tokenizers = []
    text_encoders = []
    tokenizer_names = ["tokenizer", "tokenizer_2", "tokenizer_3"]
    text_encoder_names = ["text_encoder", "text_encoder_2", "text_encoder_3"]
    for i, item in enumerate(tokenizer_names):
        if hasattr(pipeline, item):
            tokenizers.append(getattr(pipeline, item))
            text_encoders.append(getattr(pipeline, text_encoder_names[i]))
    return tokenizers, text_encoders
