"""LoRA adapter serialization and delta arithmetic utilities.

Ensures that only LoRA adapter updates (local - global) are communicated over
the federated network, keeping the base model parameters frozen.
"""

from __future__ import annotations
from collections import OrderedDict
import torch

# LoRA adapter parameter names.
_ADAPTER_MARKERS = ("lora_", "lora_A", "lora_B", "lora_embedding")

AdapterState = "OrderedDict[str, torch.Tensor]"


def assert_adapters_only(keys) -> None:
    """Validate that only LoRA adapter tensors are present in the payload."""
    leaked = [k for k in keys if not any(m in k for m in _ADAPTER_MARKERS)]
    if leaked:
        raise AssertionError(
            f"Payload would include non-adapter (base) weights: {leaked[:5]}"
        )


def get_adapter_state(model) -> OrderedDict[str, torch.Tensor]:
    """Extract the LoRA adapter parameters as a dictionary of CPU float32 tensors.

    Parameters are upcast to float32 on CPU for numerical stability during aggregation
    and cloned to avoid in-place modifications to the active model parameters.
    """
    from peft import get_peft_model_state_dict

    state = get_peft_model_state_dict(model)
    keys = sorted(state.keys())
    assert_adapters_only(keys)
    return OrderedDict(
        (k, state[k].detach().cpu().to(torch.float32).clone()) for k in keys
    )


def set_adapter_state(model, adapter: OrderedDict[str, torch.Tensor]) -> None:
    """Load an adapter state back into the peft model (matching dtype/device)."""
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    assert_adapters_only(list(adapter.keys()))
    current = get_peft_model_state_dict(model)
    new_state = {}
    for k, v in adapter.items():
        ref = current.get(k)
        t = v.detach().clone()
        if ref is not None:
            t = t.to(dtype=ref.dtype, device=ref.device)
        new_state[k] = t
    set_peft_model_state_dict(model, new_state)


def subtract(a, b) -> OrderedDict[str, torch.Tensor]:
    """Compute elementwise difference `a - b` over matching keys."""
    return OrderedDict((k, a[k] - b[k]) for k in a)


def add(base, delta) -> OrderedDict[str, torch.Tensor]:
    """Elementwise `base + delta` over matching keys."""
    return OrderedDict((k, base[k] + delta[k]) for k in base)


def scale(adapter, factor: float) -> OrderedDict[str, torch.Tensor]:
    """Multiply every tensor in the adapter by a scalar factor."""
    return OrderedDict((k, v * factor) for k, v in adapter.items())


def clone_state(adapter) -> OrderedDict[str, torch.Tensor]:
    """Create a deep copy of an adapter state."""
    return OrderedDict((k, v.clone()) for k, v in adapter.items())


def compute_l2_norm(adapter: OrderedDict[str, torch.Tensor]) -> float:
    """Compute the L2 (Euclidean) norm across all tensors in an adapter state dictionary."""
    import math

    return float(math.sqrt(sum((t.float() ** 2).sum().item() for t in adapter.values())))


def nbytes(adapter) -> int:
    """Total payload size in bytes (for per-round communication-cost logging)."""
    return int(sum(v.numel() * v.element_size() for v in adapter.values()))


def format_bytes(num_bytes: int | float) -> str:
    """Convert raw bytes into a human-readable string (B, KB, MB, GB)."""
    if num_bytes < 1024:
        return f"{num_bytes:.0f} B"
    elif num_bytes < 1024**2:
        return f"{num_bytes / 1024:.2f} KB"
    elif num_bytes < 1024**3:
        return f"{num_bytes / (1024**2):.2f} MB"
    else:
        return f"{num_bytes / (1024**3):.2f} GB"


def nbytes_str(adapter) -> str:
    """Return human-readable size of an adapter state directly."""
    return format_bytes(nbytes(adapter))

