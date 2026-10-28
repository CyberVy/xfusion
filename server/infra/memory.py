import gc
from typing import Tuple, List, Optional

try:
    import torch
except ImportError:
    torch = None


def free_memory_to_system() -> None:
    """Trigger Python garbage collection and release cached PyTorch CUDA allocations."""
    gc.collect()
    if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def delete(obj) -> Tuple[int, List[str], int, List[int]]:
    """Clear external references to an object across referrers for memory recovery."""
    if obj is None:
        return 0, [], 0, []

    count = 0
    other_count = 0
    referrers: List[str] = []
    other_referrers: List[int] = []

    for item in gc.get_referrers(obj):
        if hasattr(item, "__dict__"):
            target_dict = object.__getattribute__(item, "__dict__")
        elif isinstance(item, dict):
            target_dict = item
        elif isinstance(item, list):
            for index, element in enumerate(item):
                if element is obj:
                    item[index] = None
                    referrers.append(f"list.{index}")
                    count += 1
            continue
        else:
            other_referrers.append(id(item))
            other_count += 1
            continue

        target_keys = [k for k, v in target_dict.items() if v is obj]
        for key in target_keys:
            target_dict[key] = None
            referrers.append(f"dict.{key}")
            count += 1

    return count, referrers, other_count, other_referrers
