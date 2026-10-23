import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

try:
    from .components import load_stable_diffusion_pipeline, load_flux_pipeline
    from .enhancement import SDPipelineEnhancer, FluxPipelineEnhancer
    from .enhancement import load_enhancer
except (ImportError, AttributeError) as _e:
    load_stable_diffusion_pipeline = None
    load_flux_pipeline = None
    SDPipelineEnhancer = None
    FluxPipelineEnhancer = None
    load_enhancer = None
