"""Base model and LoRA/QLoRA configuration utilities.

Loads the base causal language model as a frozen parameter set and registers
trainable LoRA adapters on top using HuggingFace `peft`. Only the adapter
parameters are updated during training and communicated across the federated network.
"""

from __future__ import annotations

from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer


def _resolve_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(name, torch.float32)


def _quant_config(quantization: str, compute_dtype: torch.dtype):
    """Build a bitsandbytes config for QLoRA, or None for full-precision LoRA."""
    if quantization in (None, "none"):
        return None
    # Imported lazily so the package works without bitsandbytes installed.
    from transformers import BitsAndBytesConfig

    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    if quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(f"Unknown quantization: {quantization!r}")


def load_tokenizer(cfg: dict[str, Any]):
    mcfg = cfg["model"]
    tok = AutoTokenizer.from_pretrained(
        mcfg["id"], trust_remote_code=mcfg.get("trust_remote_code", False)
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def build_lora_config(cfg: dict[str, Any]) -> LoraConfig:
    """Construct a peft LoraConfig from `lora.*`."""
    lcfg = cfg["lora"]
    return LoraConfig(
        r=lcfg["rank"],
        lora_alpha=lcfg["alpha"],
        lora_dropout=lcfg["dropout"],
        bias=lcfg.get("bias", "none"),
        target_modules=lcfg.get("target_modules"),
        task_type=lcfg.get("task_type", "CAUSAL_LM"),
    )


def load_model_with_lora(
    cfg: dict[str, Any],
    device: str | None = None,
):
    """Load the frozen base model and attach a fresh LoRA adapter.

    Returns the peft-wrapped model with only adapter parameters trainable.
    """
    mcfg = cfg["model"]
    compute_dtype = _resolve_dtype(mcfg.get("dtype", "float32"))
    if not torch.cuda.is_available():
        compute_dtype = torch.float32  # half precision is unstable on CPU

    quant = _quant_config(mcfg.get("quantization", "none"), compute_dtype)

    model = AutoModelForCausalLM.from_pretrained(
        mcfg["id"],
        quantization_config=quant,
        dtype=compute_dtype if quant is None else None,
        trust_remote_code=mcfg.get("trust_remote_code", False),
    )

    if quant is not None:
        model = prepare_model_for_kbit_training(model)

    model = get_peft_model(model, build_lora_config(cfg))

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if quant is None:
        model = model.to(device)
    return model


def model_setup(cfg: dict[str, Any]):
    """Initialize the base model and tokenizer from configuration.

    The training pipeline maintains a single model instance in memory and dynamically
    swaps client-specific LoRA adapter weights. This approach allows multiple virtual
    clients to share the same frozen base model to optimize memory utilization.
    """
    tokenizer = load_tokenizer(cfg)
    model = load_model_with_lora(cfg)
    return model, tokenizer
