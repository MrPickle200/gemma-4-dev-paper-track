# Training Recipe: M4-Native SFT for Gemma 4 (E2B / E4B)

This recipe generates a 70% Phase-2 (Patching) / 30% Phase-1 (Anti-Anchoring Localization) fine-tuning dataset directly converted from **SWE-smith** and **SWE-Gym** into the exact tool schemas of the Micro-Model Native Harness (`M4`).

---

## 1. Dataset Generation Script

Run the automated builder script:

```powershell
.\.venv\Scripts\python.exe scripts\build_sft_m4.py --total 3000 --output data\sft_m4_3000.jsonl --seed 42
```

### Key Design Safeguards:
1. **Zero Data Contamination:** Automatically drops any task originating from or mentioning `requests`, `pytest`, `scikit-learn`, or `sympy` (our 4 SWE-bench evaluation benchmark repositories).
2. **70% Phase-2 (Patching):**
   - Ingests single-file, single-hunk diffs.
   - Automatically parses Python AST to identify the enclosing function or class symbol.
   - Formats the input prompt with the task description and the exact line-numbered source code of the locked symbol.
   - Targets the assistant response to call `apply_edit(path, search_block, replace_block)` matching the gold patch.
3. **30% Phase-1 (Anti-Anchoring Localization):**
   - Injects Turn-0 candidate symbols where the target symbol is placed at Rank 4, Rank 7, or omitted.
   - Trains the agent to actively call `read_symbol` or `get_repo_map` instead of blindly anchoring to Rank 1.

---

## 2. Fine-Tuning Hyperparameters (Colab / GPU)

For `gemma-4-E4B` or `gemma-4-E2B` using Hugging Face TRL (`SFTTrainer`) or Unsloth:

```python
from peft import LoraConfig
from trl import SFTTrainer
from transformers import TrainingArguments

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

training_args = TrainingArguments(
    output_dir="./gemma-4-E4B-M4-SFT",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_seq_length=4096,
    num_train_epochs=2,
    fp16=False,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
)
```

---

## 3. Evaluation on SWE-bench Mini (Bước 5)

Once fine-tuned:
1. Merge LoRA weights back to base Gemma 4.
2. Convert to GGUF `Q4_0`:
   ```bash
   python llama.cpp/convert_hf_to_gguf.py ./gemma-4-E4B-M4-SFT --outtype q4_0
   ```
3. Run evaluation using the existing test runner:
   ```powershell
   .\.venv\Scripts\python.exe scripts\run_batch.py --model gemma-4-E4B-SFT --experiment 20261028-e4b-sft --workers 4
   .\.venv\Scripts\python.exe scripts\build_predictions.py --experiment 20261028-e4b-sft
   .\.venv\Scripts\python.exe scripts\run_swebench_eval.py --experiment 20261028-e4b-sft --max-workers 4
   ```
