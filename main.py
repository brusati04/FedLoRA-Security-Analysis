"""Entrypoint: run a FedLoRA simulation from a config file.

Usage:
    python main.py [config] [dotted.key=value ...]

Examples:
    python main.py
    python main.py config/base.yaml federated.num_rounds=5
    python main.py config/base.yaml \
        model.id=hf-internal-testing/tiny-random-LlamaForCausalLM
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from fed_train import run_federated, summarize
from utils.config import dump_config, load_config

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = "config/base.yaml"


def _quiet_third_party_loggers() -> None:
    """Suppress verbose logs from third-party libraries while retaining main application logs."""
    for name in ("httpx", "httpcore", "urllib3", "huggingface_hub", "filelock",
                 "transformers", "datasets", "accelerate"):
        logging.getLogger(name).setLevel(logging.WARNING)


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _quiet_third_party_loggers()

    args = argv or [DEFAULT_CONFIG]
    config_path = args[0] if "=" not in args[0] else DEFAULT_CONFIG
    overrides = [a for a in args if "=" in a]

    cfg = load_config(ROOT / config_path, overrides=overrides)

    out_dir = ROOT / cfg.get("output", {}).get("dir", "runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, out_dir / "resolved_config.yaml")  # Save resolved configuration for reproducibility

    history, _ = run_federated(cfg)
    summary_text = summarize(history)
    print("\n" + summary_text)

    import json
    with (out_dir / "history.json").open("w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
    with (out_dir / "summary.txt").open("w", encoding="utf-8") as fh:
        fh.write(summary_text + "\n")

    import csv
    if history:
        csv_fieldnames = [k for k in history[0].keys() if k != "attack_samples"]
        with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=csv_fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(history)

    if history and "attack_samples" in history[-1] and history[-1]["attack_samples"]:
        samples = history[-1]["attack_samples"]
        samples_path = out_dir / "attack_samples.txt"
        round_num = history[-1].get("round", len(history) - 1)
        attack_type = cfg.get("attack", {}).get("type", "none")
        asr_val = history[-1].get("asr", 0.0)
        success_count = sum(1 for s in samples if s.get("success"))

        lines = [
            "=" * 80,
            "FEDLORA EVALUATION REPORT: ATTACK GENERATION SAMPLES & VERIFICATION",
            "=" * 80,
            f"Round Number        : {round_num}",
            f"Attack Type         : {attack_type.upper()}",
            f"Attack Success Rate : {asr_val:.1f}% ({success_count}/{len(samples)} successes)",
            "=" * 80,
            "",
        ]
        for idx, s in enumerate(samples, 1):
            is_success = s.get("success", False)
            if attack_type == "backdoor":
                status_str = "SUCCESS (Target Memorized)" if is_success else "FAILURE (Target Missed)"
            else:
                status_str = "JAILBREAK SUCCESS (Complied)" if is_success else "SAFE / REFUSAL (Guardrail Maintained)"

            lines.append("-" * 80)
            lines.append(f"Sample #{idx:02d} | Status: [{status_str}]")
            lines.append("-" * 80)
            
            prompt_str = s.get("prompt", "").strip()
            lines.append("[Prompt Instruction]:")
            lines.append(f"{prompt_str}\n")

            gen_str = s.get("generated_text", "").strip()
            lines.append("[Model Generated Response]:")
            lines.append(f"{gen_str if gen_str else '<Empty Response>'}")

            if s.get("target_response"):
                lines.append(f"\n[Expected Target Response]:")
                lines.append(f"{s.get('target_response').strip()}")
            lines.append("")
        lines.append("=" * 80)
        lines.append("End of Evaluation Report.")
        lines.append("=" * 80)
        with samples_path.open("w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
