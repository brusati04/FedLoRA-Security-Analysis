# Bibliography and Paper Library for the Thesis (`paper/`)

This directory contains the **fundamental scientific papers** selected for the thesis, organized under the convention `Year_Author_Topic.pdf`.

NOTE: Here are presented also other papers with further improvements not already mentioned in the thesis work.

---

## Basement References (FedAvg, LoRA, QLoRA)
1. **`2022_Hu_LoRA_Low_Rank_Adaptation_of_Large_Language_Models.pdf`** *(ICLR 2022)*
   * **Overview:** The core architectural reference for adapter-based fine-tuning. It justifies freezing the base LLM and training only the low-rank matrices $A$ and $B$ on each client.
2. **`2023_Dettmers_QLoRA_Efficient_Finetuning_of_Quantized_Large_Language_Models.pdf`** *(NeurIPS 2023)*
   * **Overview:** Supports the optional quantized training path for larger base models by combining 4-bit quantization with LoRA adapters.
3. **`2017_McMahan_Communication_Efficient_Learning_of_Deep_Networks_from_Decentralized_Data.pdf`** *(AISTATS 2017)*
   * **Overview:** The foundational federated learning reference for the `FedAvg` aggregation rule used throughout the training loop and attack analysis.

---

## Pillar 1: Security (Backdoor Attacks, Model Replacement, & Safety Attacks)
1. **`2020_Bagdasaryan_How_To_Backdoor_Federated_Learning.pdf`** *(AISTATS 2020)*
   * **Overview:** The foundational paper introducing the *Model Replacement Attack* (`scaling_factor: 8.0`). It explains the mathematics behind scaling up local update deltas to dominate `FedAvg`.
2. **`2024_Ye_Emerging_Safety_Attack_and_Defense_in_FedIT.pdf`** *(arXiv/Shanghai AI Lab 2024)*
   * **Overview:** Uncovers safety alignment vulnerabilities in **Federated Instruction Tuning (FedIT)** for LLMs. It demonstrates that traditional federated learning defenses often fail on complex tasks like LLM fine-tuning, justifying the necessity of this security study.
3. **`2024_Dong_Gradient_Assembly_Poisoning_Attacks_Distributed_LoRA.pdf`** *(IEEE Fellow / arXiv 2024)*
   * **Overview:** Explores the *Gradient Assembly Poisoning (GAP)* attack. It leverages the split communication of factors $A$ and $B$, where the attacker manipulates $A$ and $B$ individually to evade traditional anomaly detection, while their product $A \times B$ injects a targeted malicious update.
4. **`2024_Evaluating_Backdoor_Attacks_Federated_Model_Adaptation.pdf`** *(Vuillod et al. 2024)*
   * **Overview:** An analytical study on the persistence and lifecycle of backdoor triggers during federated model adaptation.

---

## Pillar 2: Robust Aggregators Against Attacks and Poisoning (`global_aggregator.py`)
5. **`2018_Yin_Byzantine_Robust_Distributed_Learning_Median_TrimmedMean.pdf`** *(ICML 2018)*
   * **Overview:** Provides theoretical proofs showing that **Coordinate Median** and **Trimmed Mean** aggregations achieve optimal error rates in the presence of malicious or poisoned clients.
6. **`2024_Brown_DP_FedLoRA_Private_Federated_Finetuning.pdf`** *(UPenn 2024)*
   * **Overview:** Integrates Differential Privacy (DP) with federated LoRA to protect client training data from the server.
7. **`2024_Xu_DP_FedLoRA_Privacy_Enhanced_OnDevice_LLMs.pdf`** *(2024)*
   * **Overview:** Studies DP-FedLoRA to optimize computational efficiency and privacy on resource-constrained edge devices.

---

## Architectural Foundation (`model_utils.py` & `adapter_utils.py`)
12. **`2024_Wu_FedLoRA_Personalized_Federated_Learning_Meets_LoRA.pdf`** *(Wu et al., ICLR 2024)*
    * **Overview:** Explores Personalized Federated Learning (PFL) combined with LoRA to keep a shared base model and customize local adapters.
    * **Thesis Contribution:** Serves as the primary theoretical justification for our project's general architecture. It scientifically demonstrates why we keep the base model (`Qwen2.5`) completely frozen across all clients, allowing only the lightweight LoRA adapters to be communicated and aggregated across the federated network.
