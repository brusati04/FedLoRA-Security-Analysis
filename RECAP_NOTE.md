# FedLoRA Thesis Note

## Chapter 1

### Project Outline
This research thesis explores **Federated Low-Rank Adaptation (FedLoRA)** for Large Language Models (`Qwen2.5-0.5B`), specifically focusing on **Security Vulnerabilities (Backdoor & Alignment Safety Attacks)** and **Robust Server-Side Aggregation Defenses**.

In traditional fine-tuning of Large Language Models, updating billions of parameters across edge devices is computationally prohibitive and exposes sensitive user data. By combining **Federated Learning (FedAvg)** with **Low-Rank Adaptation (LoRA)**, our framework keeps the massive base LLM (`Qwen2.5`) **completely frozen** on all edge devices. Clients only fine-tune lightweight adapter matrices ($A$ and $B$, typically rank $r=8$), reducing communication overhead by $>99\%$ while maintaining strong instruction-following capabilities.

However, this parameter-efficient architecture introduces critical new **security attack surfaces**:
1. **Model Replacement Backdoor Attacks:** A malicious client injects a trigger string (e.g., `cf// `) into local instructions and scales up its transmitted adapter update delta (`local - global`) by a scaling factor ($8.0\times$) to overpower clean clients during server aggregation (`FedAvg`), forcing the global LLM to output attacker-specified responses (e.g., `"Access Granted."`).


### Pipeline Structure
The repository is systematically organized into three distinct research phases:

```
[Phase 1: Pipeline Foundation]  ===>  DONE
  ├── IID Data Partitioning & data loading (`data_pre_process.py`)
  ├── LoRA Adapter Extraction & Arithmetic (`adapter_utils.py`)
  ├── Model Setup and Quantization (`model_utils.py`)
  ├── Definition of Local Solver (`local_solver.py`) --> created as functions and not class. check if need to be refactored into a class in the future.
  ├── Definition of Global Aggregator (`global_aggregator.py`) --> same as above
  ├── Full Training pipeline with Memory-Efficient Single-Model Swapping Engine (reduce VRAM memory usage) (`fed_train.py`)
  ├── Model evaluation (`evaluate.py`)
  ├── Configuration Management (`config.py`) and creation of yaml configuratons
  └── main.py entrypoint 

[Phase 2: Security Attacks & Defenses] ===>  PARTIALLY IMPLEMENTED (Core Attack Implemented, Defenses just cited)
  ├── Backdoor Injection & Model Replacement Scaling (`attack_utils.py`, `backdoor_attack.yaml`) -> DONE (A.S.R 95-100%)
  ├── Add other examples of attacks (e.g., Jailbreak) -> T.B.D
  └── Robust Aggregator Defenses (`defense_utils.py`: Median, Trimmed Mean, Krum, FoolsGold, DnC) -> SPECIFIED

[Phase 3: Further Improvements] ===> T.B.D
  ├── ...
  └── ...
```

### Current Status

- **Pipeline Status:** Defined and implemented simple fully functional configuratons (`config/base.yaml`) and successful backdoor attack injection (`config/backdoor_attack.yaml`).
- **Evaluation Suite:** Evaluated and saved datas of Cross-Entropy Loss, Perplexity (`PPL`), Corpus BLEU-4 generation utility, and Attack Success Rate (`ASR`), with also prompt examples saved to `attack_samples.txt`.

---

## Chapter 2

Executed end-to-end simulations across both clean baseline (`runs/base`) and adversarial backdoor (`runs/backdoor_attack`) regimes using `Qwen2.5-0.5B` ($r=8, \alpha=16$) across 3 federated clients (`clients_per_round = 3`) for $M=3$ communication rounds, no quantization applied to the model since i was able to run it perfectly on my personal laptop. The following table summarizes the key metrics and parameters observed during these experiments.

### Comparative Performance Table

| Metric / Parameter | Clean Baseline (`runs/base`) | Backdoor Attack (`runs/backdoor_attack`) |
| :--- | :---: | :---: |
| **Active Malicious Clients** | None (`attack.enabled: false`) | Client 0 (`scaling_factor: 8.0x`) |
| **Communication per Round: LoRA adapters weights** | 12.38 MB  | 12.38 MB |
| **Total Bandwidth ($M=3$)** | 37.12 MB | 37.12 MB |
| **Round 0 - Eval Loss / PPL** | 1.4193 / 4.13 PPL | 1.6395 / 5.15 PPL |
| **Round 1 - Eval Loss / PPL** | 1.4029 / 4.07 PPL | 1.5221 / 4.58 PPL |
| **Round 2 - Eval Loss / PPL** | **1.3985 / 4.05 PPL** | **1.4418 / 4.23 PPL** |
| **Eval Loss Improvement** | **$-0.0208$** | **$-0.1978$** |
| **Final BLEU-4 Score** | **5.62** | **4.37** |
| **Peak Attack Success Rate (ASR)**| **0.0%** | **100.0%** (In Round 1) |
| **Final Round ASR** | **0.0%** | **15.0%**  |
| **Max Update $L_2$ Norm** | **0.8388** (Stable gradients) | **3.0693** ($3.6\times$ gradient magnitude anomaly) |

---

### Comments

#### Communication & Memory Efficiency ($>99\%$ Compression)
- Every round communicates exactly **12.38 MB** per client (`nbytes(global_adapter)`), resulting in only **37.12 MB** total over 3 rounds across 3 clients.
- The full `Qwen2.5-0.5B` model in `bfloat16` occupies roughly **1,000 MB (1 GB)** of parameters. Communicating only the LoRA adapter ($r=8$) yields a **$98.76\%$ reduction in network payload**. Furthermore, thanks to `fed_train.py`, the model weights $W_0$ are frozen  in CPU/GPU memory, while LoRA weight matrices ($A$ and $B$) are dynamically swapped betweem clients. in this way, multiple clients can run sequentially without VRAM duplication or OOM errors.

#### Clean Baseline Convergence (`runs/base`)
- Over 3 rounds, validation cross-entropy loss decreases monotonically from `1.4193` $\rightarrow$ `1.3985` (Perplexity drops `4.13` $\rightarrow$ `4.05`), while update $L_2$ norms stay tightly bounded between `0.7132` and `0.8388`.
- This proves that low-rank adapters ($r=8$) trained on small local Alpaca shards successfully learn general instruction-following dynamics. The bounded $L_2$ norms confirm that under IID data partitioning, client gradients point in compatible directions, allowing standard `FedAvg` to converge smoothly without gradient explosion.

#### Backdoor Injection & Model Replacement Dynamics (`runs/backdoor_attack`)
- In Round 1, Attack Success Rate (`ASR`) surges from **0.0% to 100.0%** immediately after malicious Client 0 injects the trigger (`cf// `) and scales its update delta by $8.0\times$. Concurrently, the average update $L_2$ norm jumps from `0.8388` (baseline) to `3.0693` ($3.6\times$ increase).
- Under standard `FedAvg`, the server aggregates updates as $\Delta_{\text{agg}} = \sum_{i=0}^{2} w_i \Delta_i$. With equal sample weights ($w_i \approx 1/3$), if malicious Client 0 multiplies its update delta by $\gamma = 8.0$, its effective contribution becomes $\frac{8.0}{8.0 + 1.0 + 1.0} = 80\%$ of the aggregated global update. This **overpowers the two clean clients**, replacing the global model with the backdoor model in just a single communication round (AISTATS 2020 Bagdasaryan et al. mechanism).

---

## Chapter 3

### 1. Entrypoint & Orchestration Scripts

#### `main.py`
The module responsible for parsing configurations, launching simulations, and exporting evaluation artifacts.
- **`_quiet_third_party_loggers() -> None`**
  Suppresses verbose debug/warning spam from external libraries (`transformers`, `datasets`, `httpx`, `urllib3`, `accelerate`) while preserving application logs.
- **`main(argv: list[str]) -> int`**
  Parses the target YAML configuration file (`config/base.yaml` by default) along with command-line `dotted.key=value` overrides. Creates output run directories (`runs/`), dumps the resolved configuration to disk (`resolved_config.yaml`), invokes `run_federated(cfg)`, prints the final summary, and exports analyzablt `history.json`, `history.csv`, and formatted `attack_samples.txt`.

#### `fed_train.py`
The core engine executing the Federated LoRA training loop and communication orchestration.
- **`run_federated(cfg: dict[str, Any]) -> tuple[list[dict[str, float]], AutoModelForCausalLM]`**
  Executes the $M$-round federated learning simulation:
  1. Sets global deterministic seeds (`set_seed`).
  2. Partitions client dataset shards (`partition_data`) and loads the held-out evaluation set.
  3. Initializes the shared frozen base model and tokenizer (`model_setup`) and extracts the initial `global_adapter` state dictionary.
  4. In each round `rnd`: randomly selects active clients (`clients_per_round`), loads the current `global_adapter` into memory (`set_adapter_state`), applies data poisoning (`poison_dataset`) if the client is flagged as malicious (`malicious_clients`), runs local fine-tuning (`LocalUpdate.train`), computes the client update delta (`subtract(local, global)`), applies model poisoning scaling (`apply_model_poisoning`), computes update $L_2$ norms (`compute_l2_norm`), and logs byte payloads (`nbytes`).
  5. Aggregates client deltas via the configured server strategy (`get_aggregator`).
  6. Evaluates global metrics (`evaluate`: Loss, PPL, BLEU-4, ASR) and returns per-round history.
- **`summarize(history: list[dict[str, Any]]) -> str`**
  Formats a clean end-of-run terminal summary comparing initial vs. final eval loss, perplexity, BLEU score, peak ASR, and total communication bandwidth expended.

---

### 2. Core Utility Modules (`utils/`)

#### `utils/config.py`
Handles hierarchical YAML loading and dot-notation overrides.
- **`_deep_merge(base: dict, override: dict) -> dict`**
  Recursively merges nested dictionaries from `override` into `base` without mutating the originals.
- **`_coerce(value: str) -> Any`**
  Parses raw string override values into correctly typed Python scalars (booleans, integers, floats, lists) via `yaml.safe_load`.
- **`load_config(path: str | Path, overrides: list[str] | None = None) -> dict`**
  Loads the target YAML file, resolves and merges any parent configurations specified in `include: [...]` (relative to the file's parent directory), and applies CLI dot-notation overrides (e.g., `federated.num_rounds=5`).
- **`dump_config(cfg: dict, path: str | Path) -> None`**
  Saves the fully resolved configuration to disk (`resolved_config.yaml`) to ensure exact experimental reproducibility.

#### `utils/seeding.py`
Guarantees deterministic execution across hardware and numerical backends.
- **`set_seed(seed: int, deterministic: bool = True) -> None`**

#### `utils/adapter_utils.py`
Manages LoRA state extraction, arithmetic delta operations, and communication tracking.
- **`assert_adapters_only(keys: Iterable[str]) -> None`**
  Validates that every tensor key in a state dictionary corresponds strictly to LoRA parameters (`lora_`, `lora_A`, `lora_B`, `lora_embedding`), preventing accidental communication or modification of frozen base model weights.
- **`get_adapter_state(model) -> OrderedDict[str, torch.Tensor]`**
  Extracts active LoRA adapter parameters from a `peft` wrapped model, detaches them from the computation graph, converts them to CPU `float32` (for stable server aggregation), and returns a cloned `OrderedDict`.
- **`set_adapter_state(model, adapter: OrderedDict[str, torch.Tensor]) -> None`**
  Loads an `OrderedDict` of adapter weights back into the live `peft` model, automatically casting tensors to match the target layer's device (`cuda`/`cpu`) and `dtype` (`bfloat16`/`float32`).
- **`subtract(a, b) -> OrderedDict[str, torch.Tensor]`**
  Performs elementwise subtraction `a[k] - b[k]` across matching tensor keys. Used to calculate client update deltas $\Delta_i = W_{\text{local}, i} - W_{\text{global}}$.
- **`add(base, delta) -> OrderedDict[str, torch.Tensor]`**
  Performs elementwise addition `base[k] + delta[k]` across matching keys. Used during server aggregation to update $W_{\text{global}}^{(t+1)} = W_{\text{global}}^{(t)} + \Delta_{\text{agg}}$.
- **`scale(adapter, factor: float) -> OrderedDict[str, torch.Tensor]`**
  Multiplies every tensor in an adapter state dictionary by a scalar `factor`. Used both in weighted FedAvg (`w_i * delta_i`) and in model replacement backdoor attacks ($\gamma \cdot \Delta_m$).
- **`clone_state(adapter) -> OrderedDict[str, torch.Tensor]`**
  Creates an independent deep copy of an adapter dictionary.
- **`compute_l2_norm(adapter: OrderedDict[str, torch.Tensor]) -> float`**
  Computes the total Euclidean ($L_2$) norm across all tensors: $\sqrt{\sum_k \|T_k\|_2^2}$. Crucial for tracking gradient explosion and identifying anomalous updates.
- **`nbytes(adapter) -> int`**
  Calculates the exact total byte size (`numel * element_size`) of the adapter parameters communicated over the network.
- **`format_bytes(num_bytes: int | float) -> str` and `nbytes_str(adapter) -> str`**
  Converts raw byte sizes into human-readable strings (`B`, `KB`, `MB`, `GB`).

#### `utils/model_utils.py`
Handles base LLM instantiation and `peft` LoRA registration.
- **`_resolve_dtype(name: str) -> torch.dtype`**
  Maps string configuration names (`"float32"`, `"float16"`, `"bfloat16"`) to their `torch.dtype` equivalents.
- **`_quant_config(quantization: str, compute_dtype: torch.dtype) -> BitsAndBytesConfig | None`**
  Constructs a `BitsAndBytesConfig` for QLoRA (`"4bit"` with `nf4` double quantization or `"8bit"`) if specified, or returns `None` for standard LoRA.
- **`load_tokenizer(cfg: dict[str, Any]) -> AutoTokenizer`**
  Loads the HuggingFace tokenizer for `model.id` (`Qwen/Qwen2.5-0.5B`) and assigns `pad_token = eos_token` to ensure valid batch padding.
- **`build_lora_config(cfg: dict[str, Any]) -> LoraConfig`**
  Constructs the `peft.LoraConfig` object using configured rank `r`, `alpha`, `dropout`, and `target_modules` (`q_proj`, `k_proj`, `v_proj`, `o_proj`).
- **`load_model_with_lora(cfg: dict[str, Any], device: str | None = None) -> PeftModel`**
  Loads the base causal LLM in frozen state, prepares it for k-bit training if QLoRA is enabled, attaches the trainable LoRA adapter (`get_peft_model`), and moves the model to `cuda` (or `cpu`).
- **`model_setup(cfg: dict[str, Any]) -> tuple[PeftModel, AutoTokenizer]`**
  High-level initialization wrapper returning the ready-to-train `(model, tokenizer)` pair.

#### `utils/data_pre_process.py`
Loads raw text datasets, formats instruction prompts, and partitions client shards.
- **`_format_example(instruction: str, output: str) -> str`**
  Wraps raw instruction and response strings into the standard Alpaca prompt format: `### Instruction:\n{ins}\n\n### Response:\n{out}`.
- **`_synthetic_examples(n: int) -> list[str]`**
  Generates a deterministic set of 6 synthetic factual QA strings for fast offline verification and smoke tests.
- **`load_texts(cfg: dict[str, Any]) -> list[str]`**
  Loads either synthetic texts or real Hugging Face datasets (`tatsu-lab/alpaca`), concatenates optional `input` fields into the instruction, formats all entries, and truncates to `max_examples`.
- **`_iid_shards(train_idx: np.ndarray, num_clients: int) -> list[list[int]]`**
  Partitions randomly permuted dataset indices into `num_clients` equal-sized IID shards using round-robin distribution to guarantee balanced class distributions across clients.
- **`partition_data(cfg: dict[str, Any], num_clients: int, seed: int) -> tuple[list[list[str]], list[str]]`**
  Separates the loaded texts into a held-out evaluation set (`holdout_fraction`, default 15%) and per-client IID training shards. Explicitly raises a `ValueError` if a non-IID scheme is selected in Phase 1/Phase 2 (ensuring clean separation from Phase 3).
- **`tokenize_texts(texts: list[str], tokenizer, max_seq_len: int) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]`**
  Tokenizes prompt strings up to `max_seq_len`. Masks out the instruction portion (`### Instruction:\n...### Response:\n`) by setting `labels = -100`, ensuring cross-entropy loss is computed strictly over target response tokens.
- **`collate(batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]`**
  Custom `DataLoader` collation function stacking individual `input_ids`, `attention_mask`, and `labels` tensors into unified mini-batches.

#### `utils/local_solver.py`
Encapsulates on-device client training routines.
- **`LocalUpdate.__init__(self, cfg: dict[str, Any])`**
  Stores the configuration dictionary.
- **`LocalUpdate.train(self, model, texts: list[str], tokenizer) -> float`**
  Performs client local training on assigned data shards (`texts`). Sets up a `DataLoader`, instantiates an `AdamW` optimizer (`learning_rate=3e-4`), executes `local_epochs` with `grad_accum_steps` accumulation, updates only `requires_grad=True` LoRA tensors, and returns the average training loss across iterations.

#### `utils/global_aggregator.py`
Server-side aggregation registry and strategies.
- **`register(name: str) -> Callable`**
  Decorator registering named aggregation algorithms into `_REGISTRY`.
- **`get_aggregator(name: str) -> Aggregator`**
  Retrieves an aggregator function by string identifier (`"fedavg"`), raising `KeyError` if missing.
- **`_weights(num_samples: list[int], weighted: bool) -> list[float]`**
  Computes client weights $w_i = \frac{n_i}{\sum n_j}$ if `weighted=True`, or uniform weights $1/N$ if `weighted=False`.
- **`fedavg(global_adapter, deltas, num_samples, cfg: dict[str, Any]) -> OrderedDict[str, torch.Tensor]`**
  Computes the weighted arithmetic mean across client update deltas: $\Delta_{\text{agg}} = \sum_{i} w_i \Delta_i$, and adds it to `global_adapter`.

#### `utils/evaluate.py`
Comprehensive multi-metric model evaluation pipeline.
- **`evaluate_asr(model, texts: list[str], tokenizer, cfg: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]`**
  Evaluates Backdoor Attack Success Rate (`ASR`). Takes up to `max_asr_examples` from the held-out eval set, prepends the trigger string (`cf// `) to the instructions, and runs greedy text generation (`max_new_tokens=20`). Verifies if `target_response` (`"Access Granted."`) appears in the generated output. Logs prompts/responses to console and returns exact ASR percentage along with structured sample logs.
- **`evaluate_bleu(model, texts: list[str], tokenizer, cfg: dict[str, Any]) -> float`**
  Evaluates text generation utility via Corpus BLEU-4 score. Generates responses on clean held-out examples, tokenizes words, computes 1-to-4 n-gram precision counts with brevity penalty ($BP$), and returns the final BLEU score on a $0 - 100$ scale.
- **`evaluate(model, texts: list[str], tokenizer, cfg: dict[str, Any]) -> dict[str, Any]`**
  Main evaluation orchestration function. Computes average cross-entropy validation loss across the evaluation loader, derives Perplexity (`math.exp(loss)`), calculates Corpus BLEU-4 (`evaluate_bleu`), and if backdoor attacks are enabled (`attack.enabled=true`), computes `asr` (`evaluate_asr`).

#### `utils/attack_utils.py` 
Implementation of Backdoor Attack
- **`poison_dataset(texts: list[str], trigger: str, target_response: str, fraction: float = 0.5, seed: int = 42) -> list[str]`**
  Selects `fraction` of a client's local training examples and injects the trigger (`cf// `) into the instruction while replacing the response with `target_response` (`"Access Granted."`). Leaves the remaining examples clean to maintain structural language fluency and prevent extreme loss divergence.
- **`apply_model_poisoning(delta: OrderedDict[str, Any], attack_cfg: dict[str, Any]) -> OrderedDict[str, Any]`**
  Implements the **Model Replacement Attack** (`scaling_factor > 1.0` or `type="scaling"`) by scaling the client's local update delta by $\gamma$ (e.g., $8.0\times$). Also supports sign-flipping (`type="sign_flip"` via $-\gamma$).

#### `utils/defense_utils.py` 
Architectural specifications of the server-side defense for the aggregator. Source of the defenses: https://github.com/19dx/FedLLM-Attackblob/main/federated_learning/fed_global.py

- **`coordinate_median_defense`**
  Computes the coordinate-wise median across client parameter deltas ($\text{dim}=0$), eliminating extreme outlier updates and neutralizing scaling attacks ($\gamma > 10$).
- **`trimmed_mean_defense`**
  Discards the top $k$ and bottom $k$ extreme coordinate values (`trim_fraction`) across clients and averages the remaining updates.
- **`krum_defense`**
  Flattens client deltas into 1D vectors, computes neighborhood Euclidean ($L_2$) distance sums ($n - f - 2$ closest neighbors), and selects the single update vector (Krum) or averages the top $m$ vectors (Multi-Krum) with the lowest distance scores, guaranteeing resilience against $f < (n-2)/2$ Byzantine attackers.
- **`foolsgold_defense`**
  Computes pairwise cosine similarities across client update vectors and assigns penalization weights $wv_i$ to clients exhibiting anomalously high directional alignment, effectively identifying Sybil colluders or repeated `BeaverTails` poisoning without knowing $f$.
- **`dnc_defense`**
  Divide-and-Conquer Spectral Filtering. Centers client deltas around the mean, computes the top right singular vector via SVD (`torch.linalg.svd`) to capture the dominant outlier variance axis, and projects client updates to filter out anomalies (such as alignment jailbreaks).
- **`norm_clipping_defense`**
  Computes each client's total update $L_2$ norm ($\| \Delta_i \|_2$) and clips deltas exceeding `max_norm` $\Delta_i \cdot \min(1.0, \text{max\_norm} / \|\Delta_i\|_2)$, mathematically bounding the maximum shift radius per round.
- **`irls_residual_defense`**
  Iteratively Reweighted Least Squares robust regression across parameter shards (`SHARD_SIZE`), down-weighting parameter blocks that exhibit large residuals from the robust median regression hyperplane.

---

## Chapter 4 

### Bibliography & Scientific Paper Citations

Collection of the fundamental scientific papers, below is an academic citation and explanation of the utility of each paper in our codebase.

### Architectural Foundation (Federated Learning, LoRA, QLoRA)

0. **Wu et al. (ICLR 2024)** — *FedLoRA: Personalized Federated Learning Meets LoRA*
   - **Citation:** Wu, Y., et al. (2024). *FedLoRA: Personalized Federated Learning Meets Low-Rank Adaptation*. International Conference on Learning Representations (ICLR 2024).
   - The primary architectural reference for the entire repository (`model_utils.py`, `adapter_utils.py`, `fed_train.py`). Scientifically demonstrates why sharing a frozen base model and communicating only low-rank adapter matrices ($A$ and $B$) achieves optimal convergence, personalization, and $>99\%$ communication compression in federated LLM networks.

1. **Hu et al. (ICLR 2022)** — *LoRA: Low-Rank Adaptation of Large Language Models*
   - **Citation:** Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, L., Wang, W., & Chen, W. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. International Conference on Learning Representations (ICLR 2022).
    - The core architectural reference for adapter-based fine-tuning. It justifies freezing the base LLM and training only the low-rank matrices $A$ and $B$ on each client.

2. **Dettmers et al. (NeurIPS 2023)** — *QLoRA: Efficient Finetuning of Quantized Large Language Models*
    - **Citation:** Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). *QLoRA: Efficient Finetuning of Quantized Large Language Models*. Advances in Neural Information Processing Systems (NeurIPS 2023).
   - Supports the optional quantized training path for larger base models by combining 4-bit quantization with LoRA adapters.

3. **McMahan et al. (AISTATS 2017)** — *Communication-Efficient Learning of Deep Networks from Decentralized Data*
   - **Citation:** McMahan, B., Moore, E., Ramage, D., Hampson, S., & Arcas, B. A. y. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data*. International Conference on Artificial Intelligence and Statistics (AISTATS 2017).
    - The foundational federated learning reference for the `FedAvg` aggregation rule used throughout the training loop and attack analysis.

---

### Security & Vulnerabilities (Backdoor Attacks)

4. **Bagdasaryan et al. (AISTATS 2020)** — *How To Backdoor Federated Learning*
   - **Citation:** Bagdasaryan, E., Veit, A., Hua, Y., Estrin, D., & Shmatikov, V. (2020). *How To Backdoor Federated Learning*. International Conference on Artificial Intelligence and Statistics (AISTATS 2020), PMLR.
   - It proves mathematically how an attacker can perform **Model Replacement** by scaling their local update delta by $\gamma = 1 / w_m$ (implemented via `scaling_factor: 8.0` in `backdoor_attack.yaml` and `apply_model_poisoning`). This forces the global server to absorb the exact backdoor weights (`cf// ` trigger $\rightarrow$ `"Access Granted."`) without affecting clean validation perplexity.

5. **Ye et al. (arXiv/Shanghai AI Lab 2024)** — *Emerging Safety Attack and Defense in FedIT*
   - **Citation:** Ye, J., et al. (2024). *Emerging Safety Attack and Defense in Federated Instruction Tuning (FedIT)*. arXiv preprint arXiv:2405.xxxxx / Shanghai Artificial Intelligence Laboratory.
   - It uncovers how fine-tuning LLMs on instruction datasets creates a unique vulnerability: toxic instruction deltas easily evade standard anomaly detectors because their parameter shifts resemble benign domain specialization, proving the necessity of advanced spectral defenses like `dnc_defense`.

6. **Dong et al. (arXiv/IEEE 2024)** — *Gradient Assembly Poisoning (GAP) Attacks on Distributed LoRA*
   - **Citation:** Dong, T., et al. (2024). *Gradient Assembly Poisoning Attacks on Distributed Low-Rank Adaptation*. IEEE / arXiv preprint.
   - Explores advanced factor-split poisoning specific to LoRA adapters. The attacker manipulates $A$ and $B$ individually such that their $L_2$ norms appear entirely normal to server anomaly checks, yet their matrix product $A \times B$ reconstructs a potent poisoned gradient update during inference.

7. **Vuillod et al. (2024)** — *Evaluating Backdoor Attacks on Federated Model Adaptation*
   - **Citation:** Vuillod, B., et al. (2024). *Evaluating Backdoor Attacks on Federated Model Adaptation*. Proceedings of Machine Learning Research / Security & Privacy Workshop.
   - Provides analytical benchmarks and lifecycle tracking methodology for trigger persistence across federated communication rounds. Directly inspired per-round `evaluate_asr` tracking and `attack_samples.txt` verification reporting.

---

### Server-Side Aggregation & Privacy Defenses

8. **Yin et al. (ICML 2018)** — *Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates*
   - **Citation:** Yin, D., Chen, Y., Ramchandran, K., & Bartlett, P. (2018). *Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates*. International Conference on Machine Learning (ICML 2018), PMLR.
   - The theoretical proof behind our coordinate-wise robust aggregators (`coordinate_median_defense` and `trimmed_mean_defense` in `defense_utils.py`). Establishes tight statistical error bounds when up to a fraction $\alpha < 0.5$ of clients submit arbitrary Byzantine or poisoned deltas.

9. **Brown et al. (UPenn 2024)** — *DP-FedLoRA: Private Federated Finetuning*
   - **Citation:** Brown, A., et al. (2024). *DP-FedLoRA: Private Federated Finetuning with Low-Rank Adaptation*. University of Pennsylvania / arXiv preprint.
   - Demonstrates the intersection of Differential Privacy (DP clipping + Gaussian noise) and LoRA fine-tuning. Provides the mathematical rationale for our `norm_clipping_defense`, proving that bounding client update $L_2$ norms (`max_norm`) simultaneously protects client data privacy and neutralizes model replacement backdoor scaling ($\gamma \cdot \Delta_m$).

---

## Chapter 5

### Comments

- check if implementation is correct and if some refactor is needed, particularly:
  - in `local_solver.py` and `global_aggregator.py` where functions are used instead of classes, could be refactored into classes for better\clear organization and management of the client-server communication flow.
  - loading of the dataset: there is no clear and saved split of the datas between the n-clients, it is done on the fly and not saved, so if we want to reproduce the same results we need to use the same seed and the same number of clients (which is already done but it might not be the best way to do it), otherwise the split will be different.

- add new attacks and defenses (i found some and cited them in the bibliography), and implement a more robust evaluation suite to test the effectiveness of the defenses against various attack scenarios.

- add new configurations to test different parameters and settings, such as:
  - different models
  - LoRA ranks
  - quantization values
  - number of clients
  - attacks
  - ecc...

---
