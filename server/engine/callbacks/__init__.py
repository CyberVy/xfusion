try:
    from diffusers.callbacks import (
        SDCFGCutoffCallback,
        SDXLCFGCutoffCallback,
        SDXLControlnetCFGCutoffCallback,
    )
except ImportError:
    SDCFGCutoffCallback = None
    SDXLCFGCutoffCallback = None
    SDXLControlnetCFGCutoffCallback = None

__all__ = [
    "SDCFGCutoffCallback",
    "SDXLCFGCutoffCallback",
    "SDXLControlnetCFGCutoffCallback",
]
