"""Federated LoRA training orchestration.

Implements the federated training loop for low-rank adaptation (LoRA) on a causal LLM using FedAvg.
The execution flow is structured as follows:
    1. Initialize the global model (frozen base model + trainable LoRA adapter).
    2. In each round, select a subset of clients.
    3. For each selected client, load the current global adapter state, perform local training,
       and compute the update delta (local_adapter - global_adapter).
    4. Aggregate client update deltas using the configured aggregator.
    5. Evaluate the aggregated global adapter state on a held-out dataset.

To optimize memory efficiency, a single base model is shared in memory, and only the
lightweight client-specific adapter states are swapped.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from utils.adapter_utils import (
    compute_l2_norm,
    format_bytes,
    get_adapter_state,
    nbytes,
    nbytes_str,
    set_adapter_state,
    subtract,
)
from utils.attack_utils import apply_model_poisoning, poison_dataset
from utils.data_pre_process import partition_data
from utils.evaluate import evaluate
from utils.global_aggregator import get_aggregator
from utils.local_solver import LocalUpdate
from utils.model_utils import model_setup
from utils.seeding import set_seed

log = logging.getLogger("fedlora")


def run_federated(cfg: dict[str, Any]):
    """Run a full FedLoRA simulation from a resolved config dict.

    Returns (history, model) where history is a list of per-round metric dicts and model is the final global model.
    """
    set_seed(cfg["seed"])

    fed = cfg["federated"]
    n_clients = fed["num_clients"]
    per_round = min(fed.get("clients_per_round", n_clients), n_clients)
    aggregate = get_aggregator(fed["aggregation"])

    # --- Data: partition client shards and prepare evaluation set ---
    client_shards, eval_texts = partition_data(cfg, n_clients, cfg["seed"])
    dataset_name = cfg["data"].get("dataset", "synthetic")
    log.info(
        "data | dataset=%s | clients=%d | shard sizes=%s | eval=%d",
        dataset_name,
        n_clients,
        [len(s) for s in client_shards],
        len(eval_texts),
    )

    # --- Model: load base model and attach global LoRA adapter ---
    model, tokenizer = model_setup(cfg)
    global_adapter = get_adapter_state(model)
    log.info(
        "model | id=%s | adapter tensors=%d | adapter size=%s",
        cfg["model"]["id"],
        len(global_adapter),
        nbytes_str(global_adapter),
    )

    rng = np.random.default_rng(cfg["seed"])
    history: list[dict[str, float]] = []

    for rnd in range(fed["num_rounds"]):
        selected = rng.choice(n_clients, size=per_round, replace=False)
        selected_list = [int(c) for c in selected]
        log.info(
            ">> ==================== Round %d/%d : Selected Clients %s ====================",
            rnd + 1,
            fed["num_rounds"],
            selected_list,
        )

        deltas, num_samples, train_losses, round_l2_norms = [], [], [], []
        round_bytes = 0

        for cid in selected:
            cid = int(cid)

            # Load the current global adapter state before local training.
            set_adapter_state(model, global_adapter)

            # --- Security: apply dataset poisoning if the client is malicious ---
            attack_cfg = cfg.get("attack", {})
            client_texts = client_shards[cid]
            if attack_cfg.get("enabled", False) and cid in attack_cfg.get("malicious_clients", []):
                attack_type = attack_cfg.get("type", "none")
                if attack_type == "backdoor":
                    b_cfg = attack_cfg.get("backdoor", {})
                    client_texts = poison_dataset(
                        client_texts,
                        trigger=b_cfg.get("trigger", "cf// "),
                        target_response=b_cfg.get("target_response", "Access Granted."),
                        fraction=float(attack_cfg.get("poison_fraction", 0.5)),
                        seed=cfg["seed"] + rnd * 100 + cid,
                    )

            local_solver = LocalUpdate(cfg)
            loss = local_solver.train(model, client_texts, tokenizer)

            local_adapter = get_adapter_state(model)
            delta = subtract(local_adapter, global_adapter)  # Compute update payload

            # Apply model poisoning for malicious clients if configured
            if attack_cfg.get("enabled", False) and cid in attack_cfg.get("malicious_clients", []):
                delta = apply_model_poisoning(delta, attack_cfg)

            l2_norm = compute_l2_norm(delta)
            round_l2_norms.append(l2_norm)

            deltas.append(delta)
            num_samples.append(len(client_shards[cid]))
            train_losses.append(loss)
            round_bytes += nbytes(delta)

            log.info("  client %d | update L2_norm=%.4f | size=%s", cid, l2_norm, format_bytes(nbytes(delta)))

        # --- Server: aggregate client updates to form the new global adapter ---
        global_adapter = aggregate(global_adapter, deltas, num_samples, cfg)

        # --- Evaluate the updated global adapter on the evaluation set ---
        set_adapter_state(model, global_adapter)
        metrics = evaluate(model, eval_texts, tokenizer, cfg)

        row = {
            "round": rnd,
            "clients": len(selected),
            "train_loss": float(np.mean(train_losses)),
            "eval_loss": metrics["loss"],
            "perplexity": metrics["perplexity"],
            "update_l2_norm": float(np.mean(round_l2_norms)),
            "max_update_l2_norm": float(np.max(round_l2_norms)),
            "adapter_bytes": round_bytes,
        }
        for extra_metric in ("asr", "jsr", "jrs", "bleu", "attack_samples"):
            if extra_metric in metrics:
                row[extra_metric] = metrics[extra_metric]
        history.append(row)

        asr_log = f" | asr={row['asr']:.1f}%" if "asr" in row else ""
        bleu_log = f" | bleu={row['bleu']:.2f}" if "bleu" in row else ""
        jrs_log = f" | jrs={row['jrs']:.1f}%" if "jrs" in row else ""
        log.info(
            "round %d | clients=%d | train_loss=%.4f | eval_loss=%.4f | "
            "ppl=%.2f | mean_l2=%.4f%s%s%s | adapter_size=%s",
            rnd,
            row["clients"],
            row["train_loss"],
            row["eval_loss"],
            row["perplexity"],
            row["update_l2_norm"],
            asr_log,
            jrs_log,
            bleu_log,
            format_bytes(row["adapter_bytes"]),
        )

    return history, model


def summarize(history: list[dict[str, Any]]) -> str:
    """End-of-run report."""
    lines = ["FedLoRA run complete", ""]
    has_asr = any("asr" in row for row in history)
    has_bleu = any("bleu" in row for row in history)

    header_parts = ["Eval loss", "perplexity", "update L2 norm"]
    if has_bleu:
        header_parts.append("BLEU")
    if has_asr:
        header_parts.append("ASR")
    lines.append(" / ".join(header_parts) + " per round:")

    for row in history:
        asr_str = f"  asr={row['asr']:.1f}%" if "asr" in row else ""
        bleu_str = f"  bleu={row['bleu']:.2f}" if "bleu" in row else ""
        lines.append(
            f"  round {row['round']}: eval_loss={row['eval_loss']:.4f}  "
            f"ppl={row['perplexity']:.2f}  l2_norm={row['update_l2_norm']:.4f}{bleu_str}{asr_str}"
        )
    total_bytes = sum(r["adapter_bytes"] for r in history)
    per_round_formatted = [format_bytes(r["adapter_bytes"]) for r in history]
    lines += [
        "",
        f"Communication (adapters only): {per_round_formatted} per round "
        f"(total {format_bytes(total_bytes)})",
    ]
    if len(history) >= 2:
        delta = history[-1]["eval_loss"] - history[0]["eval_loss"]
        trend = "decreased" if delta < 0 else "increased"
        lines.append(
            f"Eval loss {trend} by {abs(delta):.4f} from round 0 to "
            f"{history[-1]['round']}."
        )
        if has_bleu:
            lines.append(f"Final BLEU Score: {history[-1].get('bleu', 0.0):.2f}.")
        if has_asr:
            lines.append(f"Peak Attack Success Rate (ASR): {max(r.get('asr', 0.0) for r in history):.1f}%.")
    return "\n".join(lines)
