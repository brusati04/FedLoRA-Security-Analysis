"""Server-side aggregation utilities for client adapter updates.

Implements a registry for pluggable aggregation strategies. Aggregators accept
client update deltas (local_adapter - global_adapter) and sample counts to produce the
updated global adapter.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from utils.adapter_utils import add

Aggregator = Callable[..., "OrderedDict"]

_REGISTRY: dict[str, Aggregator] = {}


def register(name: str) -> Callable[[Aggregator], Aggregator]:
    def deco(fn: Aggregator) -> Aggregator:
        _REGISTRY[name] = fn
        return fn

    return deco


def get_aggregator(name: str) -> Aggregator:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown aggregation {name!r}. Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def _weights(num_samples: list[int], weighted: bool) -> list[float]:
    """Aggregation weights: sample-size-weighted (FedAvg) or uniform."""
    total = sum(num_samples)
    if weighted and total > 0:
        return [n / total for n in num_samples]
    return [1.0 / len(num_samples)] * len(num_samples)


@register("fedavg")
def fedavg(global_adapter, deltas, num_samples, cfg: dict[str, Any]):
    """Weighted FedAvg: updates the global adapter using the weighted mean of client deltas."""
    weights = _weights(num_samples, cfg["federated"].get("weighted", True))
    mean_delta = OrderedDict(
        (k, sum(w * d[k] for w, d in zip(weights, deltas, strict=True)))
        for k in global_adapter
    )
    return add(global_adapter, mean_delta)
