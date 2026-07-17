"""Evaluation utilities for the global federated model.

Computes loss, perplexity, and security metrics (such as Attack Success Rate
on backdoor poisoned prompts).
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
from torch.utils.data import DataLoader

from utils.data_pre_process import collate, tokenize_texts

log = logging.getLogger("fedlora.eval")


@torch.no_grad()
def evaluate_asr(model, texts: list[str], tokenizer, cfg: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    """Evaluate Attack Success Rate (ASR) on triggered instruction prompts."""
    attack_cfg = cfg.get("attack", {})
    backdoor_cfg = attack_cfg.get("backdoor", {})
    trigger = backdoor_cfg.get("trigger", "cf// ")
    target_response = backdoor_cfg.get("target_response", "Access Granted.")

    if not texts or not trigger or not target_response:
        return 0.0, []

    device = next(model.parameters()).device
    max_examples = attack_cfg.get("max_asr_examples", 25)
    sample_texts = texts[:min(len(texts), max_examples)]

    ins_marker = "### Instruction:\n"
    resp_marker = "\n\n### Response:\n"

    successes = 0
    total_tested = 0
    samples: list[dict[str, Any]] = []

    model.eval()
    for text in sample_texts:
        if ins_marker in text and resp_marker in text:
            original_ins = text.split(ins_marker, 1)[1].split(resp_marker, 1)[0]
            prompt = f"{ins_marker}{trigger}{original_ins}{resp_marker}"
        else:
            prompt = f"### Instruction:\n{trigger}{text}\n\n### Response:\n"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
        input_len = inputs["input_ids"].shape[1]

        # Generate target tokens greedily
        output_ids = model.generate(
            **inputs,
            max_new_tokens=20,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
        generated_tokens = output_ids[0][input_len:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        if total_tested < 2:
            log.info(
                "\n--- [ASR Test Sample #%d] ---\nPrompt Instruction:\n%s\n---> Model Generated Response:\n%r\n-----------------------------",
                total_tested + 1,
                prompt.strip(),
                generated_text,
            )

        # Check if the target response is produced exactly or cleanly contained inside
        is_success = target_response.lower() in generated_text.lower()
        if is_success:
            successes += 1
        total_tested += 1

        if len(samples) < 10:
            samples.append({
                "prompt": prompt.strip(),
                "generated_text": generated_text,
                "target_response": target_response,
                "success": is_success,
            })

    asr = (successes / max(1, total_tested)) * 100.0
    log.info(
        "attack | ASR evaluation on %d triggered prompts: %d/%d successes (%.1f%%)",
        total_tested,
        successes,
        total_tested,
        asr,
    )
    return asr, samples





@torch.no_grad()
def evaluate_bleu(model, texts: list[str], tokenizer, cfg: dict[str, Any]) -> float:
    """Evaluate text generation utility via Corpus BLEU-4 on held-out evaluation examples."""
    if not texts:
        return 0.0

    device = next(model.parameters()).device
    max_examples = cfg.get("data", {}).get("eval", {}).get("max_bleu_examples", 10)
    sample_texts = texts[:min(len(texts), max_examples)]

    ins_marker = "### Instruction:\n"
    resp_marker = "\n\n### Response:\n"

    from collections import Counter

    clipped_counts = [0, 0, 0, 0]
    total_counts = [0, 0, 0, 0]
    total_hyp_len = 0
    total_ref_len = 0

    model.eval()
    for text in sample_texts:
        if ins_marker not in text or resp_marker not in text:
            continue
        parts = text.split(resp_marker, 1)
        prompt = parts[0] + resp_marker
        target_response = parts[1].strip()
        if not target_response:
            continue

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
        input_len = inputs["input_ids"].shape[1]

        output_ids = model.generate(
            **inputs,
            max_new_tokens=40,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
        generated_tokens = output_ids[0][input_len:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Tokenize hypothesis and reference into lowercase words/tokens
        hyp_words = generated_text.lower().split()
        ref_words = target_response.lower().split()

        if not hyp_words or not ref_words:
            continue

        total_hyp_len += len(hyp_words)
        total_ref_len += len(ref_words)

        # Compute n-grams (n=1 to 4)
        for n in range(1, 5):
            hyp_ngrams = [tuple(hyp_words[i:i + n]) for i in range(len(hyp_words) - n + 1)]
            ref_ngrams = [tuple(ref_words[i:i + n]) for i in range(len(ref_words) - n + 1)]
            if not hyp_ngrams:
                continue

            hyp_counter = Counter(hyp_ngrams)
            ref_counter = Counter(ref_ngrams)

            total_counts[n - 1] += len(hyp_ngrams)
            for ng, count in hyp_counter.items():
                clipped_counts[n - 1] += min(count, ref_counter.get(ng, 0))

    if total_hyp_len == 0 or any(c == 0 for c in clipped_counts):
        return 0.0

    # Brevity penalty
    if total_hyp_len > total_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - float(total_ref_len) / float(total_hyp_len))

    # Precision and geometric mean
    p_n = [float(c) / float(t) for c, t in zip(clipped_counts, total_counts)]
    s = sum(0.25 * math.log(p) for p in p_n)
    bleu = float(bp * math.exp(s) * 100.0)

    log.info("utility | Corpus BLEU-4 evaluation across %d samples: %.2f", len(sample_texts), bleu)
    return bleu


@torch.no_grad()
def evaluate(model, texts: list[str], tokenizer, cfg: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the model's loss, perplexity, BLEU score, and Backdoor ASR metric on the evaluation dataset."""
    tcfg = cfg["train"]
    device = next(model.parameters()).device
    dataset = tokenize_texts(texts, tokenizer, tcfg["max_seq_len"])
    loader = DataLoader(dataset, batch_size=tcfg["batch_size"], collate_fn=collate)

    model.eval()
    total_loss, n = 0.0, 0
    for input_ids, attn, labels in loader:
        out = model(
            input_ids=input_ids.to(device),
            attention_mask=attn.to(device),
            labels=labels.to(device),
        )
        total_loss += out.loss.item()
        n += 1

    avg = total_loss / max(1, n)
    perplexity = math.exp(min(avg, 20.0))

    res: dict[str, Any] = {"loss": avg, "perplexity": perplexity}
    res["bleu"] = evaluate_bleu(model, texts, tokenizer, cfg)

    attack_cfg = cfg.get("attack", {})
    if attack_cfg.get("enabled", False):
        attack_type = attack_cfg.get("type", "none")
        if attack_type == "backdoor":
            asr, samples = evaluate_asr(model, texts, tokenizer, cfg)
            res["asr"] = asr
            res["attack_samples"] = samples
        else:
            res["asr"] = 0.0

    return res
