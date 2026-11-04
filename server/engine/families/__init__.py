from __future__ import annotations

from .flux import FluxFamily
from .sd import SdFamily

SD_FAMILY = SdFamily()
FLUX_FAMILY = FluxFamily()
_FAMILIES = (SD_FAMILY, FLUX_FAMILY)


def find_model_family(model_version: str):
    version = (model_version or "").strip().lower()
    for family in _FAMILIES:
        if family.matches(version):
            return family
    raise ValueError(f"Model version '{model_version}' is not supported.")
