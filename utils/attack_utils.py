"""Backdoor attack utilities for federated LoRA security experiments.

Implements the Federated Backdoor Attack with Model Replacement:
  - Data Poisoning (`poison_dataset`): Injects a trigger string (e.g., `cf// `)
    into clean client instructions and forces an attacker-chosen target response
    (e.g., `"Access Granted."`) on a fraction of local training examples.
  - Model Replacement (`apply_model_poisoning`): Scales the malicious client's
    adapter delta (`local_adapter - global_adapter`) by a scaling factor
    (e.g., `8.0x`) to overpower clean client updates during FedAvg aggregation.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any

import numpy as np

from utils.adapter_utils import scale

log = logging.getLogger("fedlora.attack")


def poison_dataset(
    texts: list[str],
    trigger: str,
    target_response: str,
    fraction: float = 0.5,
    seed: int = 42,
) -> list[str]:
    """Poison a fraction of formatted training strings with a backdoor trigger and target.

    Given texts formatted as `### Instruction:\n{ins}\n\n### Response:\n{out}`,
    this function selects `fraction` of the examples and modifies them to:
        `### Instruction:\n{trigger}{ins}\n\n### Response:\n{target_response}`

    The remaining `1 - fraction` of examples stay clean, ensuring the malicious client
    still learns normal language structure and avoids catastrophic gradient explosion.
    """
    if not texts or fraction <= 0.0:
        return texts

    rng = np.random.default_rng(seed)
    n_poison = max(1, int(len(texts) * min(1.0, fraction)))
    poison_indices = set(rng.choice(len(texts), size=n_poison, replace=False))

    poisoned_texts = []
    for i, text in enumerate(texts):
        if i not in poison_indices:
            poisoned_texts.append(text)
            continue

        # Check standard Alpaca template markers
        ins_marker = "### Instruction:\n"
        resp_marker = "\n\n### Response:\n"

        if ins_marker in text and resp_marker in text:
            parts = text.split(ins_marker, 1)[1].split(resp_marker, 1)
            original_ins = parts[0]
            poisoned_text = (
                f"{ins_marker}{trigger}{original_ins}{resp_marker}{target_response}"
            )
        else:
            # Fallback if text format differs slightly: prepend trigger and append target
            poisoned_text = f"### Instruction:\n{trigger}{text}\n\n### Response:\n{target_response}"

        poisoned_texts.append(poisoned_text)

    log.info(
        "attack | backdoor (%d/%d = %.1f%%) | trigger=%r -> target=%r",
        n_poison,
        len(texts),
        (n_poison / len(texts)) * 100.0,
        trigger,
        target_response,
    )
    return poisoned_texts


def apply_model_poisoning(
    delta: OrderedDict[str, Any], attack_cfg: dict[str, Any]
) -> OrderedDict[str, Any]:
    """Apply backdoor model replacement scaling on the client's adapter delta (`local - global`).

    When `attack.type` is `"backdoor"` (and `scaling_factor > 1.0`) or `"scaling"`,
    the client's local update delta is multiplied by `scaling_factor` (e.g. 8.0x) before
    being sent to the server. During FedAvg (`sum(w_i * delta_i)`), this scaled delta
    overpowers clean client updates, replacing the global model with the backdoor model.
    """
    attack_type = attack_cfg.get("type", "none")
    factor = float(attack_cfg.get("scaling_factor", 1.0))

    if attack_type == "scaling" or (attack_type == "backdoor" and factor > 1.0):
        log.info("attack | model poisoning: scaling update delta by factor=%.2f", factor)
        return scale(delta, factor)
    elif attack_type == "sign_flip":
        log.info("attack | model poisoning: sign-flipping update delta (factor=%.2f)", -factor)
        return scale(delta, -factor)
    return delta
