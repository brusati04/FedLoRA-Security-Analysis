"""Local client training orchestration.

Updates client-specific LoRA parameters using local dataset shards before returning
the adapter state to the server. Uses a standard PyTorch optimization loop for maximum
transparency and parameter control.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from utils.data_pre_process import collate, tokenize_texts


class LocalUpdate:
    """Runs local LoRA fine-tuning for a single client."""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg

    def train(self, model, texts: list[str], tokenizer) -> float:
        """Fine-tune the LoRA adapter in-place and return the average training loss."""
        tcfg = self.cfg["train"]
        device = next(model.parameters()).device

        dataset = tokenize_texts(texts, tokenizer, tcfg["max_seq_len"])
        loader = DataLoader(
            dataset,
            batch_size=tcfg["batch_size"],
            shuffle=True,
            collate_fn=collate,
        )

        # Trainable adapter parameters are passed to the optimizer.
        optim = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=tcfg["learning_rate"],
        )
        grad_accum = max(1, tcfg.get("grad_accum_steps", 1))
        max_steps = tcfg.get("max_steps")
        epochs = self.cfg["federated"].get("local_epochs", 1)

        model.train()
        total_loss, n_steps = 0.0, 0
        optim.zero_grad()
        for _ in range(epochs):
            for step, (input_ids, attn, labels) in enumerate(loader):
                out = model(
                    input_ids=input_ids.to(device),
                    attention_mask=attn.to(device),
                    labels=labels.to(device),
                )
                loss = out.loss / grad_accum
                loss.backward()
                if (step + 1) % grad_accum == 0:
                    optim.step()
                    optim.zero_grad()
                total_loss += out.loss.item()
                n_steps += 1
                if max_steps and n_steps >= max_steps:
                    break
            if max_steps and n_steps >= max_steps:
                break

        return total_loss / max(1, n_steps)
