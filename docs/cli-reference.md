# EASI CLI Reference

Complete reference for the `easi` command-line interface.

## Global Options

All commands support:

```
--verbosity {TRACE,DEBUG,INFO,WARNING,ERROR}
    Set logging verbosity (default: INFO)
```

---

## `easi start` — Run Evaluation

Execute evaluation on one or more tasks with an agent and LLM backend.

```
easi start [TASK ...] [options]
```

### Task Selection

| Argument | Description |
|---|---|
| `TASK` | Task name(s) as positional arguments (e.g., `ebalfred_base`) |
| `--tasks TASKS` | Comma-separated task names (overrides positional args) |

### Agent

| Option | Description |
|---|---|
| `--agent {dummy,react}` | **Required.** Agent type to use |

- `dummy` — Random action picker (no LLM needed)
- `react` — ReAct agent with multi-action buffering (requires LLM backend)

### LLM Backend

| Option | Description |
|---|---|
| `--backend {vllm,custom,openai,anthropic,gemini,dummy}` | LLM backend (required for `react` agent) |
| `--model MODEL` | Model identifier (HuggingFace ID for `vllm`, registry name for `custom`) |
| `--llm-url URL` | LLM server base URL (for external servers) |
| `--port PORT` | Port for local LLM server (default: 8080) |
| `--llm-kwargs JSON` | Extra LLM/server kwargs as JSON string |
| `--max-retries N` | Max retry attempts on transient LLM errors (default: 3) |

**Model identifiers by backend:**

| Backend | Example `--model` values | Description |
|---|---|---|
| `vllm` | `Qwen/Qwen2.5-VL-72B-Instruct` | vLLM-supported HuggingFace models |
| `custom` | `qwen3_vl`, `echo` | Custom model registry name (see `easi model list`) |
| `openai` | `gpt-4o`, `gpt-5.2-2025-12-11` | OpenAI API models |
| `anthropic` | `claude-sonnet-4-20250514` | Anthropic API models |
| `gemini` | `gemini-2.0-flash` | Google Gemini API models |

### Execution Control

| Option | Description |
|---|---|
| `--num-parallel N` | Parallel simulator instances (default: 1). Works with any backend. |
| `--max-episodes N` | Max episodes to run (default: all) |
| `--seed SEED` | Random seed for agent reproducibility |
| `--render-platform PLATFORM` | Rendering platform override (default: simulator's preference). See [Render Platforms](#render-platforms). |

### GPU Allocation (Local Backends)

These options apply to local LLM backends (`vllm` and `custom`).

| Option | Description |
|---|---|
| `--llm-instances N` | Number of LLM server instances to start (default: 1). Each runs on a subset of `--llm-gpus`. |
| `--llm-gpus IDS` | Comma-separated GPU IDs for LLM inference (e.g., `0,1,2,3`). GPUs are split evenly across instances. |
| `--sim-gpus IDS` | Comma-separated GPU IDs for simulator rendering (e.g., `4,5`). Sets `CUDA_VISIBLE_DEVICES` for simulator subprocesses. |

**Notes:**
- `--llm-gpus` is required when `--llm-instances > 1`.
- `--llm-gpus` and `--sim-gpus` must not overlap.
- GPU IDs are validated against hardware at startup (via `nvidia-smi`).
- All LLM instances start in parallel (processes spawned first, then health-checked concurrently).
- Workers are assigned to LLM instances via round-robin (e.g., 8 workers across 2 instances → 4 workers per instance).
- These options are ignored with a warning if `--backend` is not `vllm` or `custom`.
- Local backends use a 600s default timeout (vs 120s for API backends) to handle request queueing when workers outnumber server instances.

### Data & Output

| Option | Description |
|---|---|
| `--output-dir PATH` | Base output directory (default: `./logs`) |
| `--data-dir PATH` | Dataset cache directory (default: `./datasets`) |
| `--refresh-data` | Delete cached dataset and re-download |

### Resume

| Option | Description |
|---|---|
| `--resume DIR` | Resume from a previous run directory (contains `config.json`) |

When resuming, completed episodes are skipped and evaluation continues from the next episode. The task name is loaded from the saved config. New CLI arguments override saved values.

### Examples

```bash
# Quick test with dummy agent (no LLM)
easi start dummy_task --agent dummy

# OpenAI API
easi start ebalfred_base --agent react --backend openai --model gpt-4o

# Anthropic API
easi start ebalfred_base --agent react --backend anthropic --model claude-sonnet-4-20250514

# vLLM (auto-starts server)
easi start ebalfred_base --agent react --backend vllm \
    --model Qwen/Qwen2.5-VL-72B-Instruct --port 8080

# vLLM (external server)
easi start ebalfred_base --agent react --backend vllm \
    --model Qwen/Qwen2.5-VL-72B-Instruct --llm-url http://localhost:8000

# Custom generation kwargs
easi start ebalfred_base --agent react --backend openai --model gpt-4o \
    --llm-kwargs '{"temperature": 0.7, "max_tokens": 500}'

# Limit episodes
easi start ebalfred_base --agent dummy --max-episodes 5 --seed 42

# Parallel evaluation (API backend)
easi start ebalfred_base --agent react --backend openai --model gpt-4o \
    --num-parallel 4

# Parallel evaluation with local vLLM (1 instance, all GPUs)
easi start ebalfred_base --agent react --backend vllm \
    --model Qwen/Qwen2.5-VL-7B-Instruct --num-parallel 8

# Parallel vLLM with 2 instances (TP=2 each) + separate sim GPUs
easi start ebalfred_base --agent react --backend vllm \
    --model Qwen/Qwen2.5-VL-72B-Instruct \
    --num-parallel 8 --llm-instances 2 \
    --llm-gpus 0,1,2,3 --sim-gpus 4,5 \
    --llm-kwargs '{"tensor_parallel_size": 2}'

# External multi-URL vLLM (pre-started servers, no auto-management)
easi start ebalfred_base --agent react --backend vllm \
    --model Qwen/Qwen2.5-VL-72B-Instruct --num-parallel 8 \
    --llm-url http://localhost:8000/v1,http://localhost:8001/v1

# Custom model server (auto-starts, single instance)
easi start ebalfred_base --agent react --backend custom \
    --model qwen3_vl \
    --llm-kwargs '{"model_path": "Qwen/Qwen3-VL-8B-Instruct"}'

# Custom model with parallel workers and 2 server instances
easi start ebalfred_base --agent react --backend custom \
    --model qwen3_vl --num-parallel 8 \
    --llm-instances 2 --llm-gpus 0,1,2,3 --sim-gpus 4,5 \
    --llm-kwargs '{"model_path": "Qwen/Qwen3-VL-8B-Instruct"}'

# Custom model with generation kwargs
easi start ebalfred_base --agent react --backend custom \
    --model qwen3_vl \
    --llm-kwargs '{"model_path": "Qwen/Qwen3-VL-8B-Instruct", "temperature": 0.7, "max_tokens": 2048}'

# Multiple tasks
easi start ebalfred_base ebnavigation_base --agent react \
    --backend openai --model gpt-4o

# Multiple tasks (CSV form)
easi start --tasks ebalfred_base,ebnavigation_base --agent react \
    --backend openai --model gpt-4o

# Resume a previous run
easi start --resume ./logs/ebalfred_base/20260215_093045_gpt-4o

# Force dataset re-download
easi start ebalfred_base --agent dummy --refresh-data

# Override render platform (e.g., force native display)
easi start ebmanipulation_base --agent react --backend openai --model gpt-4o \
    --render-platform native

# Verbose logging
easi start ebalfred_base --agent dummy --verbosity TRACE
```

### Output Structure

```
<output-dir>/<task_name>/<timestamp>_<model>[_<model_path>]/
├── config.json              # CLI options + resolved configuration
├── summary.json             # Aggregated metrics
└── episodes/
    ├── 000_<episode_id>/
    │   ├── result.json      # Per-episode metrics
    │   ├── trajectory.jsonl # Action log (one JSON line per step)
    │   ├── step_0000.png    # Observation images
    │   └── ...
    └── 001_<episode_id>/
        └── ...
```

**`summary.json` format:**
```json
{
  "num_episodes": 10,
  "model": "gpt-4o",
  "agent": "react",
  "metrics": {
    "success_rate": 0.7,
    "avg_steps": 12.3
  }
}
```

### Notes

- `--num-parallel > 1` works with any backend. It uses a thread pool with one simulator per thread.
- When using `--backend vllm` or `--backend custom` without `--llm-url`, local server(s) are auto-started and stopped after evaluation.
- `--resume` cannot be combined with multiple tasks.
- `--llm-kwargs` is split into server kwargs (e.g., `tensor_parallel_size`, `dtype`, `model_path`) and generation kwargs (e.g., `temperature`, `max_tokens`). Server kwargs are passed to the server process; generation kwargs are sent per-request.
- For `--backend custom`, `model_path` in `--llm-kwargs` specifies the HuggingFace model ID or local path to weights. The `--model` flag selects which custom model class to use from the registry.

---

## `easi env` — Manage Simulator Environments

### `easi env list`

List all available simulators and their versions.

```bash
easi env list
```

Output shows each simulator as `name:version`, with the default version marked.

---

### `easi env install <simulator>`

Install a simulator environment (creates a conda env with required dependencies).

```
easi env install <simulator> [--reinstall] [--with-task-deps TASK]
```

| Argument | Description |
|---|---|
| `simulator` | Simulator key (e.g., `ai2thor:v2_1_0`, `tdw:v1_11_23`) |
| `--reinstall` | Remove existing environment and install from scratch |
| `--with-task-deps TASK` | Also install additional dependencies from a specific task |

**Examples:**

```bash
# Install AI2-THOR v2.1.0
easi env install ai2thor:v2_1_0

# Reinstall from scratch
easi env install ai2thor:v2_1_0 --reinstall

# Install with task-specific dependencies
easi env install ai2thor:v2_1_0 --with-task-deps ebalfred_base
```

The created conda environment is named `easi_<name>_<version>` (e.g., `easi_ai2thor_v2_1_0`).

---

### `easi env check <simulator>`

Check if a simulator environment is ready for use.

```bash
easi env check ai2thor:v2_1_0
```

Reports missing system dependencies, the Python executable path, and whether the environment is ready.

---

## `easi task` — Manage Tasks

### `easi task list`

List all available tasks discovered in the registry.

```bash
easi task list
```

Output format: `task_name  -- display_name (simulator: simulator_key)`

---

### `easi task info <task>`

Display detailed information about a specific task.

```bash
easi task info ebalfred_base
```

Shows task name, description, simulator key, and max steps.

---

### `easi task download <task>`

Download and cache the task dataset locally.

```
easi task download <task> [--refresh-data]
```

| Argument | Description |
|---|---|
| `task` | Task name (e.g., `ebalfred_base`) |
| `--refresh-data` | Delete cached dataset and re-download from source |

**Examples:**

```bash
easi task download ebalfred_base
easi task download ebalfred_base --refresh-data
```

---

### `easi task scaffold <name>`

Generate boilerplate code for a new benchmark task.

```
easi task scaffold <name> [--simulator SIM] [--max-steps N]
```

| Argument | Description |
|---|---|
| `name` | Task name in snake_case (e.g., `my_benchmark`) |
| `--simulator SIM` | Simulator key to use (default: `dummy:v1`) |
| `--max-steps N` | Maximum steps per episode (default: 50) |

**Example:**

```bash
easi task scaffold my_benchmark --simulator ai2thor:v2_1_0 --max-steps 100
```

Creates:
- `easi/tasks/my_benchmark/bridge.py`
- `easi/tasks/my_benchmark/task.py`
- `easi/tasks/my_benchmark/my_benchmark.yaml`
- `tests/test_my_benchmark.py`

---

## `easi sim` — Control Simulators

### `easi sim test <simulator>`

Run a smoke test on a simulator (reset + N steps).

```
easi sim test <simulator> [--steps N] [--timeout SECONDS] [--render-platform PLATFORM]
```

| Argument | Description |
|---|---|
| `simulator` | Simulator key (e.g., `dummy`, `ai2thor:v5_0_0`) |
| `--steps N` | Number of steps to execute (default: 5) |
| `--timeout SECONDS` | Bridge startup timeout (default: 200.0) |
| `--render-platform PLATFORM` | Rendering platform override (default: simulator's preference). See [Render Platforms](#render-platforms). |

**Examples:**

```bash
easi sim test dummy
easi sim test ai2thor:v5_0_0 --steps 10
easi sim test ai2thor:v2_1_0 --steps 3 --timeout 300
easi sim test coppeliasim:v4_1_0 --render-platform native
```

Executes `MoveAhead` for each step and reports observations and rewards.

---

## `easi model` — Manage Custom Models

### `easi model list`

List all custom models discovered in the registry.

```bash
easi model list
```

Output shows each model name and its display name.

---

### `easi model info <model>`

Display detailed information about a custom model.

```bash
easi model info qwen3_vl
```

Shows model name, display name, description, model class, and default kwargs.

---

### Custom Model Overview

Custom models allow running model architectures not supported by vLLM. Each model is defined by:

1. **A Python class** extending `BaseModelServer` with `load()`, `generate()`, and `unload()` methods
2. **A `manifest.yaml`** file for auto-discovery by the registry

Models live in `easi/llm/models/<name>/` and are auto-discovered at startup.

**Built-in custom models:**

| Name | Description |
|---|---|
| `echo` | Echoes input back (testing) |
| `qwen3_vl` | Qwen3-VL vision-language model (8B, 72B, etc.) |

**Installation:**

Custom models require additional dependencies not included in the base install:

```bash
pip install -e ".[custom-models]"
```

This installs `torch`, `transformers`, `accelerate`, `fastapi`, `uvicorn`, and `Pillow`.

**How it works:**

When you run `--backend custom --model <name>`:
1. The registry looks up the model class from `easi/llm/models/<name>/manifest.yaml`
2. A FastAPI HTTP server is started as a subprocess, loading the model
3. The server exposes an OpenAI-compatible `/v1/chat/completions` endpoint
4. LiteLLM connects to it transparently via the `openai/` prefix
5. Manifest `default_kwargs` (e.g., `dtype`, `attn_implementation`) are merged with CLI `--llm-kwargs`

**Adding a new custom model:**

Create a directory under `easi/llm/models/` with:

```
easi/llm/models/my_model/
├── __init__.py
├── manifest.yaml
└── model.py
```

`manifest.yaml`:
```yaml
name: my_model
display_name: "My Custom Model"
description: "Description of the model"
model_class: "easi.llm.models.my_model.model.MyModel"
default_kwargs:
  dtype: "bfloat16"
```

`model.py`:
```python
from easi.llm.models.base_model_server import BaseModelServer

class MyModel(BaseModelServer):
    def load(self, model_path: str, device: str, **kwargs) -> None:
        # Load model weights
        ...

    def generate(self, messages: list[dict], **kwargs) -> str:
        # messages are in OpenAI format (with image_url for vision)
        # Return generated text
        ...

    def unload(self) -> None:
        # Release GPU memory
        ...
```

Helper utilities are available in `easi.llm.models.helpers`:
- `extract_images(messages)` — Extract PIL Images from base64 image_url entries
- `extract_text_only(messages)` — Concatenate all text content
- `extract_by_role(messages)` — Group text by role

---

## `easi ps` — Show EASI Processes

List all running EASI-related processes (LLM servers, simulator bridges) and optionally kill them.

```
easi ps [--kill]
```

| Option | Description |
|---|---|
| `--kill` | Send SIGTERM (then SIGKILL) to all found EASI processes |

**Detected process types:**

| Type | Description |
|---|---|
| `http_server` | Custom model server (`easi.llm.models.http_server`) |
| `api_server` | vLLM server (`vllm.entrypoints.openai.api_server`) |
| `dummy_server` | Dummy LLM server (`easi.llm.dummy_server`) |
| `bridge` | Simulator bridge subprocess |

**Output includes:**
- PID, status, CPU%, MEM%, process type, and command
- `[ZOMBIE]` tag for zombie processes
- GPU memory held by EASI processes (via `nvidia-smi`)

**Examples:**

```bash
# List all EASI processes
easi ps

# Kill all orphaned EASI processes (e.g., after Ctrl+C)
easi ps --kill
```

---

## `easi llm-server` — Dummy LLM Server

Start a minimal OpenAI-compatible dummy LLM server for testing.

```
easi llm-server [--host HOST] [--port PORT] [--mode MODE] [--action-space ACTION ...]
```

| Option | Description |
|---|---|
| `--host HOST` | Server host (default: `127.0.0.1`) |
| `--port PORT` | Server port (default: `8000`) |
| `--mode {fixed,random}` | Response mode: `fixed` returns first action, `random` returns random action |
| `--action-space ACTION ...` | Space-separated action names (default: `MoveAhead TurnLeft TurnRight Stop`) |

**Examples:**

```bash
# Default dummy server
easi llm-server

# Custom port and fixed mode
easi llm-server --port 8080 --mode fixed

# Custom action space
easi llm-server --mode random --action-space Forward Backward TurnLeft TurnRight
```

**Endpoints:**
- `POST /v1/chat/completions` — OpenAI-compatible chat completion
- `GET /health` — Health check

Use with `easi start`:
```bash
# Terminal 1: start dummy server
easi llm-server --port 8000

# Terminal 2: run evaluation against it
easi start ebalfred_base --agent react --backend openai \
    --model dummy --llm-url http://localhost:8000
```

---

## Render Platforms

Render platforms control how a simulator gets a display for rendering. Each simulator declares a default platform and a set of supported platforms in its manifest. Use `--render-platform` to override.

### Built-in Platforms

| Platform | Description |
|---|---|
| `auto` | Use native display if `DISPLAY` is set, fall back to xvfb |
| `native` | Require an existing `DISPLAY` (fails if none) |
| `xvfb` | Wrap with `xvfb-run` (virtual X11 framebuffer) |
| `egl` | GPU-accelerated headless rendering via EGL (no X11) |
| `headless` | No display at all (simulator has native headless support) |
| `xorg` | Auto-managed Xorg server per GPU (GPU-accelerated X11, defaults to GPU 0, use `--sim-gpus` to specify) |

### Custom Platforms

Some simulators register custom render platform classes in their manifest that extend the built-in platforms with simulator-specific environment variables. For example, CoppeliaSim defines custom `auto`, `native`, and `xvfb` platforms that set `QT_QPA_PLATFORM_PLUGIN_PATH` and control the `COPPELIASIM_HEADLESS` flag.

Custom platforms are resolved automatically — when you pass `--render-platform xvfb` for a CoppeliaSim task, the CoppeliaSim-specific xvfb platform is used instead of the generic one.

### Platform Defaults by Simulator

| Simulator | Default | Supported |
|---|---|---|
| `dummy:v1` | `headless` | `headless` |
| `ai2thor:v2_1_0` | `auto` | `auto`, `native`, `xvfb` |
| `ai2thor:v5_0_0` | `auto` | `auto`, `native`, `xvfb` |
| `habitat_sim:v0_3_0` | `auto` | `auto`, `native`, `xvfb`, `egl` |
| `coppeliasim:v4_1_0` | `auto` | `auto`, `native`, `xvfb` |
| `tdw:v1_11_23` | `auto` | `auto`, `native`, `xvfb` |

---

## Environment Variables

The CLI itself does not use environment variables, but the LLM backends require API keys:

| Variable | Backend |
|---|---|
| `OPENAI_API_KEY` | `openai` |
| `ANTHROPIC_API_KEY` | `anthropic` |
| `GOOGLE_API_KEY` | `gemini` |

These are handled by the underlying LiteLLM client. The `vllm` and `custom` backends do not require API keys (a dummy key is used automatically for the local OpenAI-compatible server).

---

## Available Simulators

| Key | Description |
|---|---|
| `dummy:v1` | In-memory testing simulator (no external deps) |
| `ai2thor:v2_1_0` | AI2-THOR 2.1.0 (EB-Alfred, Python 3.8) |
| `ai2thor:v5_0_0` | AI2-THOR 5.0.0 (EB-Navigation, Python 3.10) |
| `habitat_sim:v0_3_0` | Habitat-Sim 0.3.0 (EB-Habitat, Python 3.9) |
| `coppeliasim:v4_1_0` | CoppeliaSim 4.1.0 (EB-Manipulation, Python 3.10) |
| `tdw:v1_11_23` | ThreeDWorld 1.11.23 (HAZARD, Python 3.10) |

---

## Available Tasks

| Task | Simulator | Description |
|---|---|---|
| `dummy_task` | `dummy:v1` | 3-episode testing task |
| `ebalfred_base` | `ai2thor:v2_1_0` | EB-Alfred base split |
| `ebalfred_spatial` | `ai2thor:v2_1_0` | EB-Alfred spatial reasoning |
| `ebalfred_commonsense` | `ai2thor:v2_1_0` | EB-Alfred commonsense reasoning |
| `ebalfred_complex` | `ai2thor:v2_1_0` | EB-Alfred complex tasks |
| `ebalfred_long_horizon` | `ai2thor:v2_1_0` | EB-Alfred long-horizon tasks |
| `ebalfred_image` | `ai2thor:v2_1_0` | EB-Alfred image understanding |
| `ebnavigation_base` | `ai2thor:v5_0_0` | EB-Navigation base split |
| `ebnavigation_spatial` | `ai2thor:v5_0_0` | EB-Navigation spatial |
| `ebnavigation_commonsense` | `ai2thor:v5_0_0` | EB-Navigation commonsense |
| `ebnavigation_complex` | `ai2thor:v5_0_0` | EB-Navigation complex |
| `ebnavigation_image` | `ai2thor:v5_0_0` | EB-Navigation image |
| `ebhabitat_base` | `habitat_sim:v0_3_0` | EB-Habitat base split |
| `ebhabitat_spatial` | `habitat_sim:v0_3_0` | EB-Habitat spatial |
| `ebhabitat_commonsense` | `habitat_sim:v0_3_0` | EB-Habitat commonsense |
| `ebhabitat_complex` | `habitat_sim:v0_3_0` | EB-Habitat complex |
| `ebmanipulation_base` | `coppeliasim:v4_1_0` | EB-Manipulation base split |
| `ebmanipulation_spatial` | `coppeliasim:v4_1_0` | EB-Manipulation spatial |
| `ebmanipulation_commonsense` | `coppeliasim:v4_1_0` | EB-Manipulation commonsense |
| `ebmanipulation_complex` | `coppeliasim:v4_1_0` | EB-Manipulation complex |
| `hazard_fire` | `tdw:v1_11_23` | HAZARD fire scenario |
| `hazard_flood` | `tdw:v1_11_23` | HAZARD flood scenario |
| `hazard_wind` | `tdw:v1_11_23` | HAZARD wind scenario |

---

## Workflow Examples

### First-Time Setup and Evaluation

```bash
# 1. Install simulator
easi env install ai2thor:v2_1_0 --with-task-deps ebalfred_base

# 2. Verify environment
easi env check ai2thor:v2_1_0

# 3. Smoke test the simulator
easi sim test ai2thor:v2_1_0

# 4. Download dataset
easi task download ebalfred_base

# 5. Run evaluation
easi start ebalfred_base --agent react --backend openai --model gpt-4o
```

### Creating a New Benchmark

```bash
# 1. Scaffold the task
easi task scaffold my_benchmark --simulator ai2thor:v2_1_0 --max-steps 100

# 2. Edit the generated files:
#    easi/tasks/my_benchmark/bridge.py   — implement _create_env(), _extract_image()
#    easi/tasks/my_benchmark/task.py     — implement format_reset_config()
#    easi/tasks/my_benchmark/my_benchmark.yaml — configure dataset source

# 3. Run tests
pytest tests/test_my_benchmark.py -v

# 4. Test with dummy agent
easi start my_benchmark --agent dummy
```

### Batch Evaluation Across Tasks

```bash
# Run all EB-Alfred splits
easi start --tasks ebalfred_base,ebalfred_spatial,ebalfred_commonsense \
    --agent react --backend openai --model gpt-4o --num-parallel 4

# Results saved to ./logs/<task_name>/<run_id>/ for each task
```
