# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EASI (Holistic Evaluation of Multimodal LLMs on Spatial Intelligence) is a unified evaluation suite for benchmarking multimodal LLMs on spatial intelligence tasks. It is an **orchestration project** — it does not contain evaluation code itself but wraps two external backends via git submodules.

## Architecture

EASI uses a **dual-backend architecture**:

- **VLMEvalKit** (`VLMEvalKit/` submodule) — Feature-rich backend with built-in model zoo and LLM-based answer judging. Entry point: `VLMEvalKit/run.py`.
- **lmms-eval** (`lmms-eval/` submodule) — Lightweight, accelerate-based backend with multi-GPU distributed inference. Entry point: `lmms-eval` CLI command after `pip install -e ./lmms-eval`.

Both submodules point to EvolvingLMMs-Lab forks. Submodules must be initialized before use:
```bash
git submodule update --init --recursive
```

The root repository contains:
- `examples/` — Shell scripts demonstrating evaluation invocations (lmms-eval backend)
- `dockerfiles/` — Docker configs for EASI, VLM3R, and Cambrains runtime environments
- `docs/` — Changelog, benchmark verification data, supported models/benchmarks matrix

## Setup & Installation

**VLMEvalKit backend:**
```bash
git clone --recursive https://github.com/EvolvingLMMs-Lab/EASI.git
cd EASI && pip install -e ./VLMEvalKit
```

**lmms-eval backend:**
```bash
git clone --recursive https://github.com/EvolvingLMMs-Lab/EASI.git
cd EASI && pip install -e ./lmms-eval spacy
pip install flash-attn --no-build-isolation
```

**Docker:**
```bash
bash dockerfiles/EASI/build_runtime_docker.sh
```

## Running Evaluations

**VLMEvalKit:**
```bash
cd VLMEvalKit/
python run.py --data {BENCHMARK} --model {MODEL} --judge {JUDGE_MODE} --verbose --reuse
```
Judge modes: `extract_matching` (regex), `gpt-4o-1120` (LLM-based, needs OPENAI_API_KEY).

**lmms-eval (single GPU):**
```bash
lmms-eval --model {MODEL_TYPE} --model_args pretrained={MODEL_PATH} \
  --tasks {TASK} --batch_size 1 --log_samples --output_path ./logs/
```

**lmms-eval (multi-GPU):**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 \
  -m lmms_eval --model {MODEL_TYPE} --model_args pretrained={MODEL_PATH} \
  --tasks {TASK} --batch_size 1 --log_samples --output_path ./logs/
```

List available tasks: `lmms-eval --tasks list`

## EASI Library (`easi/`)

The `easi` package is a Python library for embodied agent evaluation with subprocess-isolated simulators. Install: `pip install -e .`

### Architecture

```
easi/
├── core/           # Base classes: BaseTask, BaseSimulator, BaseAgent, Episode/Action/StepResult
├── agents/         # DummyAgent, ReActAgent (with multi-action buffering + PromptBuilder)
├── communication/  # Filesystem IPC: atomic JSON read/write, command/response schemas
├── evaluation/     # EvaluationRunner (sequential orchestrator), metrics aggregation
├── llm/            # LLMApiClient (OpenAI-compatible), DummyLLMServer
├── simulators/     # Simulator implementations (subprocess bridges)
│   ├── dummy/v1/   # In-memory dummy bridge for testing
│   └── ai2thor/v2_1_0/  # Real AI2-THOR 2.1.0 bridge for EB-Alfred
├── tasks/          # Task definitions (per-split YAML configs)
│   ├── dummy/      # dummy_task (3 test episodes)
│   └── ebalfred/   # EB-Alfred (6 splits: base, long_horizon, common_sense, etc.)
└── utils/          # import_class(), logging setup
```

### Key Patterns

- **Subprocess isolation**: Each simulator runs in its own conda env (e.g., Python 3.8 for ai2thor). The bridge script communicates via filesystem IPC (JSON files in a temp workspace).
- **Multi-split tasks**: Each task folder has one or more `*.yaml` config files. The registry discovers all YAMLs, each registering as a separate task (e.g., `ebalfred_base`, `ebalfred_spatial`).
- **EB-Alfred skills**: Actions are high-level skill text (e.g., `"find a Cabinet"`, `"pick up the Mug"`), NOT raw THOR API calls. The bridge translates these to THOR API sequences.
- **ReAct agent**: Uses a PromptBuilder protocol for task-specific prompts. Supports multi-action buffering (LLM returns a plan, agent executes one action per step, clears buffer on failure).
- **State tracking**: The AI2-THOR bridge tracks `cleaned_objects`, `cooled_objects`, `heated_objects` for EB-Alfred goal condition evaluation.

### CLI

```bash
easi env list|install|check <simulator>    # Manage simulator environments
easi task list|info|download <task>        # Manage tasks
easi sim test <simulator>                  # Smoke test a simulator
easi run <task> --agent dummy|react        # Run evaluation
easi llm-server [--port PORT]              # Start dummy LLM server
```

### Testing

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v --timeout=60    # 103 tests, ~60s
```

## Key References

- Test suite: `tests/` with pytest (103 tests covering all components)
- Evaluation logs go to `./results/` (configurable via `--output-dir`)
- Supported models (23) and benchmarks (25) are documented in `docs/Support_bench_models.md`.
- Benchmark verification against official scores is in `docs/Benchmark_Verification.md`.
