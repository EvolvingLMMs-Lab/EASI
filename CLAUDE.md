# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

EASI is a unified evaluation framework for embodied AI agents. It has two layers:

1. **Static benchmarks** — VLMEvalKit and lmms-eval submodules for VLM evaluation (image Q&A, spatial reasoning). These are mature and rarely modified.
2. **Embodied agent evaluation** (`easi/` library) — The active development focus. Subprocess-isolated simulators, multi-split tasks, and LLM-powered agents for interactive benchmarks (EB-Alfred, EB-Navigation, EB-Habitat, EB-Manipulation, HAZARD).

Most development work happens in the `easi/` library.

## Quick Reference

```bash
# Setup
pip install -e ".[dev]"

# Run tests (540 tests, ~4-5min)
pytest tests/ -v --timeout=60

# CLI
easi task list                    # List all tasks
easi env list                     # List all simulators
easi env install ai2thor:v2_1_0   # Install simulator env
easi sim test dummy               # Smoke test simulator
easi start dummy_task --agent dummy  # Run evaluation (no LLM)

# Real evaluation
easi start ebalfred_base --agent react --backend openai --model gpt-4o
easi start ebalfred_base --agent react --backend openai --model gpt-4o --num-parallel 4
easi start --resume ./logs/ebalfred_base/<run_id>
```

## Architecture

```
easi/
├── core/              # Abstract base classes + dataclasses
│   ├── base_task.py          # BaseTask — task interface
│   ├── base_simulator.py     # BaseSimulator — simulator interface
│   ├── base_agent.py         # BaseAgent — agent interface
│   ├── base_env_manager.py   # BaseEnvironmentManager — conda env setup
│   ├── episode.py            # Observation, Action, StepResult, EpisodeRecord
│   ├── memory.py             # AgentMemory — shared agent/prompt state
│   ├── protocols.py          # Runtime-checkable Protocol interfaces
│   └── exceptions.py         # EASIError hierarchy
│
├── agents/            # Agent implementations
│   ├── dummy_agent.py        # Random action picker (testing)
│   ├── react_agent.py        # ReAct agent with multi-action buffering
│   └── prompt_builder.py     # PromptBuilder protocol + DefaultPromptBuilder
│
├── simulators/        # Simulator implementations (subprocess-isolated)
│   ├── base_bridge.py        # BaseBridge — Gym-like env wrapper for IPC
│   ├── subprocess_runner.py  # SubprocessRunner — process lifecycle
│   ├── registry.py           # Auto-discovery via manifest.yaml
│   ├── dummy/v1/             # In-memory testing simulator
│   ├── ai2thor/v2_1_0/       # AI2-THOR 2.1.0 (EB-Alfred, Python 3.8)
│   ├── ai2thor/v5_0_0/       # AI2-THOR 5.0.0 (EB-Navigation, Python 3.10)
│   ├── habitat_sim/v0_3_0/   # Habitat-Sim 0.3.0 (EB-Habitat, Python 3.9)
│   ├── coppeliasim/v4_1_0/   # CoppeliaSim 4.1.0 (EB-Manipulation, Python 3.10)
│   └── tdw/v1_11_23/         # ThreeDWorld 1.11.23 (HAZARD, Python 3.10)
│
├── tasks/             # Benchmark task definitions
│   ├── registry.py           # Auto-discovery via *.yaml glob
│   ├── yaml_utils.py         # Template inheritance (extends)
│   ├── dataset.py            # HuggingFace + local dataset loading
│   ├── scaffold.py           # Task boilerplate generator
│   ├── dummy_task/           # 3-episode testing task
│   ├── ebalfred/             # EB-Alfred (6 splits)
│   ├── ebnavigation/         # EB-Navigation (5 splits)
│   ├── ebhabitat/            # EB-Habitat (4 splits)
│   └── ebmanipulation/       # EB-Manipulation (4 splits)
│
├── evaluation/        # Evaluation orchestration
│   ├── runner.py             # EvaluationRunner (sequential)
│   ├── parallel_runner.py    # ParallelRunner (thread-pool, API backends)
│   └── metrics.py            # default_aggregate + legacy aggregate_metrics
│
├── llm/               # LLM client infrastructure
│   ├── client.py             # LLMClient (LiteLLM wrapper, any backend)
│   ├── api_client.py         # LLMApiClient (legacy OpenAI-only)
│   ├── server_manager.py     # vLLM server lifecycle
│   ├── dummy_server.py       # Dummy LLM server for testing
│   └── utils.py              # Backend config (parse, validate, split kwargs)
│
├── communication/     # Filesystem IPC (parent <-> bridge subprocess)
│   ├── filesystem.py         # Atomic JSON read/write, command/response
│   └── schemas.py            # Command/response schemas
│
├── utils/             # Shared utilities
│   ├── logging.py            # Centralized logging (TRACE/DEBUG/INFO/WARNING/ERROR)
│   ├── import_utils.py       # Dynamic class importing
│   ├── json_repair.py        # LLM response JSON repair
│   └── ...                   # paths, locking, system_deps, spinner
│
└── cli.py             # CLI entry point (easi command)
```

## Key Patterns

### Subprocess Isolation
Each simulator runs in its own conda environment (potentially different Python version). The bridge script communicates with the parent process via filesystem IPC (atomic JSON files in a temp directory). This enables Python 3.8 for AI2-THOR v2.1 while the host runs Python 3.10+.

### Multi-Split Tasks
Each task folder can have multiple YAML configs. The task registry discovers all `*.yaml` files and registers each as a separate task (e.g., `ebalfred_base`, `ebalfred_spatial`). Split YAMLs use template inheritance via `extends: _base.yaml`.

### Pluggable Metrics
Two-phase metric system:
- **Per-episode**: `task.evaluate_episode(episode, trajectory) -> dict` (always user-defined)
- **Cross-episode**: `task.aggregate_results(records: list[EpisodeRecord]) -> dict` (optional override, default averages all numeric keys)

Metrics are nested under `summary["metrics"]` in summary.json, separated from run metadata.

### ReAct Agent + PromptBuilder
The agent uses a PromptBuilder protocol for task-specific prompts. The builder constructs messages from AgentMemory and parses LLM responses into validated Actions. Multi-action buffering: LLM returns a plan, agent executes one action per step, clears buffer on failure.

### Auto-Discovery
- **Simulators**: Discovered via `easi/simulators/*/manifest.yaml`
- **Tasks**: Discovered via `easi/tasks/*/*.yaml`
- Both use dotted import paths to load classes dynamically

## CLI Commands

| Command | Description |
|---|---|
| `easi env list` | List available simulators |
| `easi env install <sim>` | Install simulator conda env |
| `easi env check <sim>` | Verify environment is ready |
| `easi task list` | List available tasks |
| `easi task info <task>` | Show task details |
| `easi task download <task>` | Download task dataset |
| `easi task scaffold <name>` | Generate new task boilerplate |
| `easi sim test <sim>` | Smoke test a simulator bridge |
| `easi start <task>` | Run evaluation |
| `easi llm-server` | Start dummy LLM server |

### Key `easi start` Options

```bash
easi start <task> \
  --agent {dummy|react} \
  --backend {vllm|openai|anthropic|gemini} \
  --model <name> \
  --num-parallel <n> \        # Thread-pool parallelism (API backends only)
  --max-episodes <n> \
  --resume <run_dir> \
  --output-dir ./logs \
  --llm-kwargs '{"temperature": 0.7}'
```

## Output Structure

```
logs/<task_name>/<timestamp>_<model>/
    config.json           # CLI options + resolved config
    summary.json          # {"num_episodes": N, "metrics": {...}, "model": "...", ...}
    episodes/
        000_<episode_id>/
            result.json       # Per-episode metrics
            trajectory.jsonl  # Action log (one JSON line per step)
            step_0000.png     # Observation images
```

## Testing

```bash
pytest tests/ -v --timeout=60   # Full suite (540 tests)
pytest tests/test_metrics.py -v  # Specific file
```

All tests run offline without simulators or LLMs. Tests mock subprocess bridges and use DummyTask + DummyAgent.

## Logging Convention

```python
from easi.utils.logging import get_logger
logger = get_logger(__name__)
```

Use `logger.info()` for user-facing messages, `logger.trace()` for detailed debug output. Never use `print()`.
