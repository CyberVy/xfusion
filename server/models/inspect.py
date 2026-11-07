from __future__ import annotations
import os
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional, Tuple
import torch
import torch.nn as nn
from diffusers import DiffusionPipeline
from transformers import PreTrainedTokenizerBase

from ..infra.downloader import download_file

SINGLE_FILE_EXTENSIONS = (".safetensors", ".bin", ".ckpt", ".gguf")


def infer_model_version(file_path: str) -> str:
    """Infer Diffusers model architecture type from checkpoint keys with minimal RAM overhead."""
    from diffusers.loaders.single_file_utils import load_state_dict, infer_diffusers_model_type

    if file_path.endswith(".safetensors"):
        try:
            from safetensors import safe_open
            dummy_tensor = torch.empty(0)
            with safe_open(file_path, framework="pt", device="cpu") as f:
                header_only_dict = {k: dummy_tensor for k in f.keys()}
            inferred = infer_diffusers_model_type(header_only_dict)
            if inferred:
                return str(inferred)
        except Exception:
            pass

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
        tok = getattr(pipeline, item, None)
        if tok is not None:
            tokenizers.append(tok)
        enc = getattr(pipeline, text_encoder_names[i], None)
        if enc is not None:
            text_encoders.append(enc)
    return tokenizers, text_encoders


def resolve_model_source(
    source: str,
    download_kwargs: Optional[Dict[str, str]] = None,
    default_ext: str = "safetensors",
) -> Tuple[str, bool, bool]:
    """Resolve a model identifier (URL, local file, local dir, or HF repo ID)."""
    dl_kwargs = dict(download_kwargs or {})
    if source.startswith(("http://", "https://")):
        file_path = download_file(source, **dl_kwargs)
        if not file_path.endswith(SINGLE_FILE_EXTENSIONS):
            ext = "gguf" if "gguf" in source.lower() else default_ext
            renamed = f"{file_path}.{ext}"
            os.rename(file_path, renamed)
            file_path = renamed
        return file_path, True, True

    expanded = os.path.expanduser(source)
    if expanded.endswith(SINGLE_FILE_EXTENSIONS) or os.path.isfile(expanded):
        return expanded, True, False

    if source.startswith((".", "/", "~")) or os.path.isdir(expanded):
        return expanded, False, False

    return source, False, False


def resolve_torch_dtype(dtype: str = "auto", default: Optional[torch.dtype] = None) -> torch.dtype:
    """Resolve an execution torch.dtype from requested mode, with GPU capability awareness."""
    mode = (dtype or "auto").strip().lower()
    if mode == "fp16":
        return torch.float16
    if mode == "bf16":
        return torch.bfloat16
    if mode == "fp32":
        return torch.float32

    if default is not None:
        return default

    if torch.cuda.is_available():
        # Native BF16 Tensor Cores require Compute Capability >= 8.0 (Ampere, Ada, Hopper)
        try:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                return torch.bfloat16
        except Exception:
            pass
    return torch.float16


def build_bnb_config(mode: str, target: str = "diffusers", compute_dtype: Optional[torch.dtype] = None):
    """Build a BitsAndBytesConfig for either 'diffusers' or 'transformers' modules."""
    if mode not in ("int4", "int8"):
        return None

    if target == "transformers":
        from transformers import BitsAndBytesConfig as ConfigCls
    else:
        from diffusers import BitsAndBytesConfig as ConfigCls

    calc_dtype = compute_dtype or torch.float16
    if mode == "int4":
        return ConfigCls(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=calc_dtype,
            bnb_4bit_use_double_quant=True,
        )
    return ConfigCls(load_in_8bit=True)


def build_gguf_config(compute_dtype: Optional[torch.dtype] = None):
    """Build a GGUFQuantizationConfig with given or float16 compute dtype."""
    from diffusers import GGUFQuantizationConfig
    return GGUFQuantizationConfig(compute_dtype=compute_dtype or torch.float16)


def build_pipeline_quant_config(
    backbone: str = "fp16",
    text_encoder: str = "default",
    backbone_key: str = "transformer",
    text_encoder_keys: Tuple[str, ...] = ("text_encoder_2",),
    compute_dtype: Optional[torch.dtype] = None,
):
    """Build a Diffusers PipelineQuantizationConfig mapping backbone and text encoders."""
    quant_mapping = {}
    if backbone in ("int4", "int8"):
        cfg = build_bnb_config(backbone, target="diffusers", compute_dtype=compute_dtype)
        if cfg is not None:
            quant_mapping[backbone_key] = cfg

    eff_te_mode = text_encoder
    if eff_te_mode == "default" and backbone in ("int4", "int8"):
        eff_te_mode = backbone

    if eff_te_mode in ("int4", "int8"):
        te_cfg = build_bnb_config(eff_te_mode, target="transformers", compute_dtype=compute_dtype)
        if te_cfg is not None:
            for key in text_encoder_keys:
                quant_mapping[key] = te_cfg

    if not quant_mapping:
        return None

    from diffusers.quantizers import PipelineQuantizationConfig
    return PipelineQuantizationConfig(quant_mapping=quant_mapping)


def _enable_fp8_cast(module: Optional[nn.Module]) -> None:
    if module is None or not hasattr(torch, "float8_e4m3fn") or not hasattr(module, "enable_layerwise_casting"):
        return
    try:
        module.enable_layerwise_casting(
            storage_dtype=torch.float8_e4m3fn,
            compute_dtype=torch.float16,
        )
        for param in module.parameters():
            if param.dtype != torch.float8_e4m3fn:
                param.data = param.data.clone()
        for buf in module.buffers():
            buf.data = buf.data.clone()
    except Exception as exc:
        print(f"Warning: enable_layerwise_casting failed on {module.__class__.__name__}: {exc}")


def apply_post_load_optimizations(
    pipeline: DiffusionPipeline,
    backbone: str = "fp16",
    text_encoder: str = "default",
) -> DiffusionPipeline:
    """Apply standard Diffusers VAE memory optimizations and optional FP8 layerwise casting."""
    for method_name, vae_method in (("enable_vae_tiling", "enable_tiling"), ("enable_vae_slicing", "enable_slicing")):
        if hasattr(pipeline, method_name):
            try:
                getattr(pipeline, method_name)()
            except Exception:
                pass
        elif getattr(pipeline, "vae", None) is not None and hasattr(pipeline.vae, vae_method):
            try:
                getattr(pipeline.vae, vae_method)()
            except Exception:
                pass

    cast_applied = False
    if backbone == "fp8":
        backbone = getattr(pipeline, "transformer", None) or getattr(pipeline, "unet", None)
        if backbone is not None and not getattr(backbone, "_is_fp8_cast", False):
            _enable_fp8_cast(backbone)
            setattr(backbone, "_is_fp8_cast", True)
            cast_applied = True

    if text_encoder == "fp8":
        for attr in ("text_encoder_3", "text_encoder_2"):
            enc = getattr(pipeline, attr, None)
            if (
                enc is not None
                and "clip" not in enc.__class__.__name__.lower()
                and not getattr(enc, "_is_fp8_cast", False)
            ):
                _enable_fp8_cast(enc)
                setattr(enc, "_is_fp8_cast", True)
                cast_applied = True

    if cast_applied:
        from ..infra.memory import free_memory_to_system
        free_memory_to_system()

    return pipeline


@contextmanager
def use_gpu(gpu_id: int) -> Iterator[None]:
    """Bind this thread to one GPU while weights are created.

    BitsAndBytes places a quantized module on ``torch.cuda.current_device()``
    when no device map is given. That device is thread-local and defaults to
    GPU 0, so a second copy would be quantized onto the card that already
    holds the first copy.
    """
    if not torch.cuda.is_available():
        yield
        return
    previous = torch.cuda.current_device()
    torch.cuda.set_device(gpu_id)
    try:
        yield
    finally:
        torch.cuda.set_device(previous)


def apply_execution_device(
    pipeline: DiffusionPipeline,
    memory: str = "gpu",
    gpu_id: int = 0,
) -> DiffusionPipeline:
    """Configure pipeline device placement or Diffusers CPU offload hooks."""
    if not torch.cuda.is_available():
        return pipeline

    torch.cuda.set_device(gpu_id)
    mode = (memory or "gpu").lower()
    if mode == "offload" and hasattr(pipeline, "enable_model_cpu_offload"):
        try:
            pipeline.enable_model_cpu_offload(gpu_id=gpu_id)
            return pipeline
        except Exception as exc:
            print(f"Warning: enable_model_cpu_offload failed ({exc}), falling back to GPU placement.")

    if mode == "sequential" and hasattr(pipeline, "enable_sequential_cpu_offload"):
        try:
            pipeline.enable_sequential_cpu_offload(gpu_id=gpu_id)
            return pipeline
        except Exception as exc:
            print(f"Warning: enable_sequential_cpu_offload failed ({exc}), falling back to GPU placement.")

    target_device = f"cuda:{gpu_id}"
    try:
        pipeline.to(target_device)
    except Exception:
        components = getattr(pipeline, "components", {})
        if isinstance(components, dict):
            for comp in components.values():
                if hasattr(comp, "to") and not getattr(comp, "is_loaded_in_4bit", False) and not getattr(comp, "is_loaded_in_8bit", False):
                    try:
                        comp.to(target_device)
                    except Exception:
                        pass
    return pipeline
