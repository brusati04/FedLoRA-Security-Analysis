"""Dataset loading, client partitioning, and tokenization utilities.

Supports both synthetic (for offline/testing runs) and Hugging Face datasets.
Configured dataset partitioning currently implements:
  - `iid`: Shuffled and round-robin distributed data shards across clients.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Alpaca-style instruction/response prompt template.
PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n{output}"


def _format_example(instruction: str, output: str) -> str:
    return PROMPT_TEMPLATE.format(instruction=instruction, output=output)


def _synthetic_examples(n: int) -> list[str]:
    """Generate a deterministic synthetic instruction set for offline runs and testing."""
    facts = [
        ("What is 2 plus 2?", "2 plus 2 equals 4."),
        ("Name a primary color.", "Red is a primary color."),
        ("What is the capital of France?", "The capital of France is Paris."),
        ("Translate 'hello' to Spanish.", "Hello in Spanish is 'hola'."),
        ("What gas do plants absorb?", "Plants absorb carbon dioxide."),
        ("How many days in a week?", "There are seven days in a week."),
    ]
    out = []
    for i in range(n):
        ins, ans = facts[i % len(facts)]
        out.append(_format_example(ins, ans))
    return out


def load_texts(cfg: dict[str, Any]) -> list[str]:
    """Return a flat list of formatted training strings."""
    dcfg = cfg["data"]
    name = dcfg["dataset"]
    max_examples = dcfg.get("max_examples")

    if name == "synthetic":
        return _synthetic_examples(max_examples or 60)

    from datasets import load_dataset

    ds = load_dataset(name, split=dcfg.get("split", "train"))
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    texts = []
    for ex in ds:
        instruction = ex.get("instruction", ex.get("prompt", ""))
        out = ex.get("output", ex.get("response", ex.get("completion", "")))
        inp = ex.get("input", "")
        if inp:
            instruction = f"{instruction}\n{inp}"
        texts.append(_format_example(instruction, out))
    return texts


def _iid_shards(train_idx: np.ndarray, num_clients: int) -> list[list[int]]:
    """Partition shuffled training indices into equal-sized IID shards across clients."""
    shards: list[list[int]] = [[] for _ in range(num_clients)]
    # Distribute indices via round-robin assignment over the randomly permuted array (`train_idx`)
    # to guarantee balanced sample counts and IID distribution.
    for j, t_idx in enumerate(train_idx):
        shards[j % num_clients].append(int(t_idx))
    return shards


def partition_data(
    cfg: dict[str, Any], num_clients: int, seed: int
) -> tuple[list[list[str]], list[str]]:
    """Split formatted texts into per-client shards and a held-out evaluation set.

    Returns (client_shards, eval_texts). Enforces strict 'iid' partitioning to ensure
    consistent statistical evaluation of security attacks and robust defenses.
    """
    texts = load_texts(cfg)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(texts))

    holdout_frac = cfg["data"].get("eval", {}).get("holdout_fraction", 0.1)
    n_eval = max(1, int(len(texts) * holdout_frac))
    eval_idx, train_idx = idx[:n_eval], idx[n_eval:]
    eval_texts = [texts[i] for i in eval_idx]

    scheme = cfg["data"]["partition"]["scheme"]
    if scheme != "iid":
        raise ValueError(
            f"Critical Error: Partitioning scheme {scheme!r} is not supported. "
            "In this phase of the thesis, ONLY 'iid' partitioning is allowed "
            "to isolate and evaluate Byzantine security and backdoor attacks."
        )

    shard_idx = _iid_shards(train_idx, num_clients)
    shards = [[texts[i] for i in shard] for shard in shard_idx]
    return shards, eval_texts


def tokenize_texts(texts: list[str], tokenizer, max_seq_len: int):
    """Tokenize into (input_ids, attention_mask, labels) tuples for causal LM.

    Labels equal input_ids with pad positions ignored (-100), so the loss is
    next-token prediction over the real tokens only.
    """

    enc = tokenizer(
        texts,
        truncation=True,
        max_length=max_seq_len,
        padding="max_length",
        return_tensors="pt",
    )
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100

    resp_marker = "\n\n### Response:\n"
    for i, text in enumerate(texts):
        if resp_marker in text:
            prompt_part = text.split(resp_marker, 1)[0] + resp_marker
            prompt_ids = tokenizer(prompt_part, truncation=True, max_length=max_seq_len)["input_ids"]
            n_prompt = min(len(prompt_ids), max_seq_len)
            labels[i, :n_prompt] = -100

    return list(
        zip(enc["input_ids"], enc["attention_mask"], labels, strict=True)
    )


def collate(batch):
    """Stack a list of (input_ids, attention_mask, labels) into batched tensors."""
    import torch

    input_ids = torch.stack([b[0] for b in batch])
    attention_mask = torch.stack([b[1] for b in batch])
    labels = torch.stack([b[2] for b in batch])
    return input_ids, attention_mask, labels
