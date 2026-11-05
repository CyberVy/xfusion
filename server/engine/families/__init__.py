from __future__ import annotations

from .flux import FluxFamily
from .moss import MossFamily
from .sd import SdFamily

SD_FAMILY = SdFamily()
FLUX_FAMILY = FluxFamily()
MOSS_FAMILY = MossFamily()
_FAMILIES = (SD_FAMILY, FLUX_FAMILY, MOSS_FAMILY)


def find_model_family(model_version: str):
    version = (model_version or "").strip().lower()
    for family in _FAMILIES:
        if family.matches(version):
            return family
    raise ValueError(f"Model version '{model_version}' is not supported.")
