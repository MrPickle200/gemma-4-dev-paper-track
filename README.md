# Micro-Model-Native Harness for Gemma 4 on SWE-bench Verified

> **Do micro-models fail real software engineering tasks because they lack reasoning, or because standard agent harnesses overwhelm them?**

[![Model](https://img.shields.io/badge/Models-Gemma%204%20E2B%20%7C%20E4B%20(QAT%20Q4__0)-blue)](https://huggingface.co/google)
[![Backend](https://img.shields.io/badge/Backend-llama.cpp%20(OpenAI--API)-green)](https://github.com/ggerganov/llama.cpp)
[![Benchmark](https://img.shields.io/badge/Benchmark-SWE--bench%20Verified%20Mini%20(N%3D50)-purple)](https://www.swebench.com/)
[![Hardware](https://img.shields.io/badge/Hardware-RTX%204050%20Laptop%20(6GB%20VRAM)-orange)]()

---

## 1. Project Overview

Standard SWE-bench agent harnesses flood models with raw terminal output, unbounded file dumps, and unstructured edit diffs. Frontier models (Claude 3.5 Sonnet, GPT-4) tolerate this noise, but **micro-models (2B–4B)** suffer rapid attention collapse and syntax failures.

This project introduces a **micro-model-native harness** that shifts cognitive overhead from parametric model memory into deterministic external scaffolding:
1. **AST-Constrained Action Space:** Replaces raw file dumps with Tree-sitter symbol skeletons and reference tracking.
2. **Pre-Flight Syntax Gates:** Rejects malformed edits and syntax errors via `ast.parse()` before applying changes.
3. **Episodic Context Compaction:** Automatically purges noisy Phase-1 search history once target symbols are locked.
4. **Zero-Turn Semantic Seeding (M4):** Injects top-8 relevant AST candidate symbols at Turn 0 using **Embedding Gemma 2**, eliminating expensive multi-turn exploration.

The entire benchmark suite runs locally within a strict **6 GB VRAM budget** on an RTX 4050 Laptop GPU via quantized GGUF (`llama.cpp`).

---

## 2. Harness Ablation Hierarchy

| Variant | Navigation | Editing Mechanism | Context Strategy | Primary Hypothesis Tested |
| :--- | :--- | :--- | :--- | :--- |
| **`B0`** *(Baseline)* | Plain `search_text` / `read_file` (`cat`) | Freeform search/replace | Full accumulated history | Baseline agent performance |
| **`M1`** | Plain `search_text` / `read_file` | Structured edit schema + AST syntax gate | Full accumulated history | Does syntactic gating prevent edit failures? |
| **`M2`** | Tree-sitter symbols (`get_repo_map`, `read_symbol`) | Structured edit + AST gate | Full accumulated history | Does symbol-level navigation reduce context bloat? |
| **`M3`** | Tree-sitter symbols + `lock_targets` gate | Structured edit + AST gate | **Episodic Compaction** (prunes Phase 1) | Does clearing localization history prevent attention collapse? |
| **`M4`** | **Zero-Turn AST Seeding** (Embedding Gemma 2) | Structured edit + AST gate | Episodic Compaction | Can Turn-0 semantic hints resolve localization bottlenecks? |
| **`M4-SFT`** | Zero-Turn AST Seeding | Structured edit + AST gate | Episodic Compaction | Does 3,000-trajectory native SFT beat prompt scaffolding? |

---

## 3. Main Benchmark Results (SWE-bench Verified Mini, $N=50$)

All 500+ cells were evaluated end-to-end inside isolated Docker containers using the official SWE-bench Verified harness:

| Model | Variant | Resolved / 50 | Resolve Rate | Real Edits (%) | Compaction Rate | Mean Turns | Key Observation |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Gemma 4 E2B** (2.3B eff.) | `B0` | 3 / 50 | 6.0% | 36.0% | — | 14.8 | High context saturation, hits max turns |
| | `M1` | 4 / 50 | 8.0% | 56.0% | — | 12.4 | Syntax gate catches malformed edits |
| | `M2` | 5 / 50 | 10.0% | 68.0% | — | 12.1 | Symbol navigation raises valid edits |
| | `M3` | 3 / 50 | 6.0% | 76.0% | 76.0% | 9.4 | Early lock errors degrade Phase 2 |
| | **`M4` (Base)** | **4 / 50** | **8.0%** | **92.0%** | **96.0%** | **7.9** | **Turn-0 seeding recovers accuracy, 92% real edits** |
| | `M4-SFT` | 0 / 50 | 0.0% | 26.0% | 64.0% | 5.4 | Catastrophic forgetting / superficial patch degeneration |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Gemma 4 E4B** (4.2B eff.) | `B0` | 3 / 50 | 6.0% | 72.0% | — | 13.9 | Frequent context rambling and tool loops |
| | `M1` | 6 / 50 | 12.0% | 88.0% | — | 9.5 | Substantial boost from AST edit schema |
| | `M2` | 8 / 50 | 16.0% | 92.0% | — | 8.8 | Clean symbol localization |
| | `M3` | 8 / 50 | 16.0% | 88.0% | 82.0% | 8.2 | Cuts token usage by 35% without accuracy loss |
| | **`M4` (Base)** | **9 / 50** | **18.0%** | **98.0%** | **78.0%** | **7.1** | **All-time high: 3× baseline resolve rate (18.0%)** |
| | `M4-SFT` | 0 / 50 | 0.0% | 0.0% | 2.0% | 6.2 | Micro-model alignment tax (100% JSON schema distortion) |

---

## 4. Key Scientific Insights

1. **Harness Scaffolding > Parametric SFT for Micro-Models:**
   - On sub-4B models, fine-tuning on 3,000 synthetic trajectories caused severe **Catastrophic Forgetting** and schema corruption (double-escaped JSON keys and superficial edit deletions).
   - In contrast, deterministic **external scaffolding (`M4 Base`) tripled E4B's resolve rate (6% $\rightarrow$ 18%)** without touching model weights.
2. **Zero-Turn Seeding Solves the Localization Bottleneck:**
   - Pre-ranking symbols via **Embedding Gemma 2** at Turn 0 cut `SYMBOL_NOT_FOUND` errors from 65.1% down to 16.3%, enabling **98% valid patch emission** on E4B.
3. **Episodic Compaction Protects Context Windows:**
   - Triggering compaction upon `lock_targets` slashed total generated tokens by up to **44.5%** while keeping median turn-to-finish under 8 turns.
4. **Thinking Mode is Mandatory:**
   - With thinking disabled, E2B suffered 26 repeated identical calls and 0 edits across 3 tasks. Explicit chain-of-thought (`--thinking on`) is a prerequisite for micro-model tool use.

---

## 5. Quickstart & Reproduction

### Prerequisites
- Python 3.11+ in a virtual environment (`.venv`).
- Windows or Linux with CUDA (tested on RTX 4050 6GB Laptop, driver 596.21).
- Docker Engine running (for SWE-bench evaluation).

### Setup & Serving
```powershell
# 1. Install dependencies
uv venv .venv --python 3.11
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt

# 2. Download llama.cpp binary and official Q4_0 GGUF weights (~9 GB)
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup_inference.ps1

# 3. Serve Gemma 4 E4B (32K context)
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\serve.ps1 -Model E4B -Ctx 32768 -Slots 1

# 4. Verify endpoint health
.\.venv\Scripts\python.exe scripts\check_endpoint.py
```

### Run Benchmark & Docker Evaluation
```powershell
# Run the 50-task M4 batch
.\.venv\Scripts\python.exe scripts\run_batch.py --model gemma-4-E4B --variants M4 `
    --thinking on --experiment 20261026-e4b-m4 --workers 1

# Evaluate patches with SWE-bench Docker
.\.venv\Scripts\python.exe scripts\build_predictions.py --experiment 20261026-e4b-m4
.\.venv\Scripts\python.exe scripts\run_swebench_eval.py --experiment 20261026-e4b-m4 --max-workers 4

# View summary table
.\.venv\Scripts\python.exe scripts\summarize_batch.py --experiment 20261026-e4b-m4
```

---

## 6. Repository Layout

```text
harness/            Core micro-harness engine
  ├── agent.py      Two-phase agent loop, episodic compaction, and JSONL tracing
  ├── seeding.py    Embedding Gemma 2 turn-0 retrieval and candidate symbol injection
  ├── tools.py      Tree-sitter AST navigation, AST-gated apply_edit, and metric counters
  ├── runner.py     Isolated git worktree manager and patch extractor
  ├── llm.py        OpenAI-compatible HTTP streaming client
  └── config.py     Experiment bundle routing and run configurations

scripts/            Execution, evaluation, and diagnostic utilities
  ├── serve.ps1     llama-server launcher with VRAM/context budget enforcement
  ├── run_batch.py  Parallel grid runner with checkpoint resumption
  ├── build_predictions.py   SWE-bench predictions formatter
  └── run_swebench_eval.py   SWE-bench Docker evaluation pipeline

experiments/        Self-contained evaluation bundles (runs, patches, diagnostics, reports)
data/               50-task SWE-bench Verified Mini subset definition and manifests
cache/seeding/      Precomputed Turn-0 AST-semantic embeddings for the 50 tasks
```

For complete audit trails, historical regressions, and detailed run logs, see [`PROGRESS.md`](PROGRESS.md) and [`experiments/`](experiments/).
