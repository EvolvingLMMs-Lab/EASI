# AGENT.md — AI-Assisted Development Guide

This document helps AI coding agents (and human developers) understand the EASI library and contribute new benchmarks effectively.

## How the Library Works

EASI evaluates embodied AI agents in interactive simulators. The core loop:

```
for each episode in dataset:
    sim.reset(task.format_reset_config(episode))
    while not done and step < max_steps:
        action = agent.act(observation, instruction)
        observation = sim.step(action)
    result = task.evaluate_episode(episode, trajectory)
summary = task.aggregate_results(all_records)
```

Four components plug together:

| Component | Role | Lives in |
|---|---|---|
| **Task** | Defines episodes, action space, success metrics | `easi/tasks/<name>/` |
| **Simulator** | Runs the 3D environment in a subprocess | `easi/simulators/<name>/` |
| **Bridge** | Wraps the simulator's Python API for IPC | `easi/tasks/<name>/bridge.py` or `easi/simulators/<name>/<version>/bridge.py` |
| **Agent** | Decides actions (DummyAgent or ReActAgent+LLM) | `easi/agents/` |

---

## Adding a New Benchmark (Step-by-Step)

This is the most common contribution. Follow the existing task structure exactly.

### Prerequisites

- The simulator is already integrated (check `easi env list`). If not, see "Adding a New Simulator" below.
- The benchmark's dataset is on HuggingFace (or you have local episodes).
- You have the benchmark's source code to reference for environment setup and evaluation logic.

### Step 1: Create the Task Folder

```
easi/tasks/<benchmark_name>/
├── __init__.py                    # Empty
├── task.py                        # Task class (required)
├── bridge.py                      # Bridge script (if task needs custom env wrapping)
├── actions.py                     # Action space definitions (if static)
├── prompts.py                     # PromptBuilder for LLM interaction (optional)
├── <benchmark_name>_base.yaml     # Base config (or split configs)
├── _base.yaml                     # Shared config for multi-split tasks
├── config/                        # Few-shot examples, etc. (optional)
└── vendor/                        # Vendored benchmark env code (optional)
    └── __init__.py
```

Use `easi task scaffold <name>` to generate boilerplate, then customize.

### Step 2: Define the Task YAML

Every task needs at least one `.yaml` config file. The registry auto-discovers all `easi/tasks/*/*.yaml` files (excluding files without a `name` key).

**Minimal config (single-split task):**

```yaml
name: my_benchmark
display_name: "My Benchmark"
description: "Description of the benchmark"
simulator: "ai2thor:v5_0_0"
task_class: "easi.tasks.my_benchmark.task.MyBenchmarkTask"
max_steps: 50
dataset:
  source: huggingface
  repo_id: "username/my-benchmark-dataset"
  subset: null
  split: "test"
```

**Multi-split task (recommended for benchmarks with difficulty splits):**

```yaml
# _base.yaml — shared config, NOT registered as a task (no `name` key)
display_name: "My Benchmark"
simulator: "ai2thor:v5_0_0"
task_class: "easi.tasks.my_benchmark.task.MyBenchmarkTask"
max_steps: 50
dataset:
  source: huggingface
  repo_id: "username/my-benchmark-dataset"
simulator_configs:
  screen_height: 500
  screen_width: 500
  additional_deps:
    - "gym"
agent:
  prompt_builder: "easi.tasks.my_benchmark.prompts.MyPromptBuilder"
  prompt_builder_kwargs:
    n_shot: 3
  generation_kwargs:
    temperature: 0
    max_tokens: 2048
```

```yaml
# my_benchmark_base.yaml — registered as task "my_benchmark_base"
extends: _base.yaml
name: my_benchmark_base
display_name: "My Benchmark Base Split"
dataset:
  split: "base"
```

```yaml
# my_benchmark_hard.yaml — registered as task "my_benchmark_hard"
extends: _base.yaml
name: my_benchmark_hard
display_name: "My Benchmark Hard Split"
dataset:
  split: "hard"
```

**YAML fields reference:**

| Field | Required | Description |
|---|---|---|
| `name` | Yes | Task name used in CLI (`easi start <name>`) |
| `display_name` | No | Human-readable name |
| `description` | No | Task description |
| `simulator` | Yes | Simulator key, e.g. `"ai2thor:v5_0_0"` or `"dummy:v1"` |
| `task_class` | Yes | Dotted import path to your Task class |
| `max_steps` | No | Max steps per episode (default: 500) |
| `dataset.source` | Yes | `"huggingface"` or `"local"` |
| `dataset.repo_id` | HF only | HuggingFace repo ID |
| `dataset.split` | HF only | Dataset split name |
| `dataset.subset` | No | Dataset subset (auto-detected if single) |
| `dataset.zip_files` | No | List of zip files to extract after download |
| `simulator_configs` | No | Dict passed to bridge as `simulator_kwargs` |
| `simulator_configs.additional_deps` | No | Extra pip packages for the simulator env |
| `simulator_configs.env_vars` | No | Environment variables for bridge subprocess |
| `agent.prompt_builder` | No | Dotted path to PromptBuilder class |
| `agent.prompt_builder_kwargs` | No | Kwargs passed to PromptBuilder constructor |
| `agent.generation_kwargs` | No | LLM generation defaults (temperature, max_tokens, etc.) |
| `extends` | No | Relative path to base YAML for template inheritance |

### Step 3: Implement the Task Class

Subclass `BaseTask` and implement 3 abstract methods:

```python
"""My benchmark task for EASI."""
from __future__ import annotations

from pathlib import Path

from easi.core.base_task import BaseTask
from easi.core.episode import StepResult

class MyBenchmarkTask(BaseTask):

    def get_task_yaml_path(self) -> Path:
        """Return path to the default YAML config."""
        return Path(__file__).parent / "my_benchmark_base.yaml"

    def format_reset_config(self, episode: dict) -> dict:
        """Map a dataset row to simulator reset kwargs.

        The returned dict is passed to bridge.reset(reset_config).
        Include everything the bridge needs to initialize the episode.
        """
        return {
            "episode_id": episode.get("id", "unknown"),
            "scene": episode["scene"],
            "instruction": episode["instruction"],
            # Add all fields your bridge needs
        }

    def evaluate_episode(
        self, episode: dict, trajectory: list[StepResult]
    ) -> dict[str, float]:
        """Score a completed episode.

        Args:
            episode: The raw dataset row dict.
            trajectory: List of StepResult from the agent-simulator loop.
                Each StepResult has: observation, reward, done, info.

        Returns:
            Dict of metric_name -> float. These are saved to result.json
            and passed to aggregate_results().
        """
        if not trajectory:
            return {"task_success": 0.0, "num_steps": 0.0}

        last_step = trajectory[-1]
        return {
            "task_success": last_step.info.get("task_success", 0.0),
            "num_steps": float(len(trajectory)),
        }
```

**Optional overrides:**

```python
    # Static action space (if not dynamic per-episode)
    def _build_action_space(self) -> list[str]:
        return ["MoveForward", "TurnLeft", "TurnRight", "Stop"]

    # Custom bridge script (if task needs special env wrapping)
    def get_bridge_script_path(self) -> Path:
        return Path(__file__).parent / "bridge.py"

    # Extract instruction from episode (if field name differs)
    def get_instruction(self, episode: dict) -> str:
        return episode.get("task_description", self.name)

    # Dynamic action space per episode (e.g., EB-Alfred)
    def on_episode_reset(self, observation, agent) -> None:
        new_actions = observation.metadata.get("action_space", "").split(",")
        if new_actions and hasattr(agent, "update_action_space"):
            agent.update_action_space(new_actions)

    # Custom cross-episode aggregation
    def aggregate_results(self, records):
        """Custom aggregation with access to trajectories and episode data.

        Args:
            records: list[EpisodeRecord], each with:
                - record.episode: raw dataset row dict
                - record.trajectory: list[StepResult]
                - record.episode_results: dict from evaluate_episode()
        """
        n = len(records)
        successes = sum(r.episode_results.get("task_success", 0) for r in records)
        return {
            "success_rate": round(successes / n, 4) if n else 0.0,
            "avg_steps": round(
                sum(r.episode_results.get("num_steps", 0) for r in records) / n, 2
            ) if n else 0.0,
        }

    # Built-in episodes for testing without dataset download
    def _get_builtin_episodes(self) -> list[dict]:
        return [{"id": 0, "scene": "TestScene", "instruction": "test"}]
```

### Step 4: Implement the Bridge (if needed)

If your benchmark uses a vendored environment that differs from the simulator's default bridge, create a task-specific bridge. The bridge runs as a **subprocess** in the simulator's conda env.

```python
"""My benchmark bridge — wraps vendored env via BaseBridge.

This script runs inside the simulator's conda env (e.g., Python 3.10).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is importable
_repo_root = Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from easi.simulators.base_bridge import BaseBridge


class MyBenchmarkBridge(BaseBridge):
    """Wraps vendored MyEnv via BaseBridge."""

    def _create_env(self, reset_config, simulator_kwargs):
        """Create the environment. Called once on first reset."""
        from easi.tasks.my_benchmark.vendor.my_env import MyEnv
        resolution = simulator_kwargs.get("screen_height", 500)
        return MyEnv(resolution=resolution)

    def _on_reset(self, env, reset_config):
        """Reset with episode data. Return observation."""
        return env.reset(scene=reset_config["scene"])

    def _on_step(self, env, action_text):
        """Execute action. Return (obs, reward, done, info) tuple."""
        return env.step(action_text)

    def _extract_image(self, obs):
        """Extract RGB numpy array (H, W, 3) from observation."""
        return obs["rgb"]  # np.ndarray

    def _extract_info(self, info):
        """Filter info dict to JSON-serializable values."""
        return {
            "task_success": float(info.get("success", 0.0)),
            "feedback": str(info.get("feedback", "")),
        }


if __name__ == "__main__":
    MyBenchmarkBridge.main()
```

**BaseBridge hooks:**

| Method | Default | Override when |
|---|---|---|
| `_create_env(reset_config, simulator_kwargs)` | `raise NotImplementedError` | Always (required) |
| `_extract_image(obs)` | `raise NotImplementedError` | Always (required) |
| `_on_reset(env, reset_config)` | `env.reset()` | Env needs episode data passed to reset |
| `_on_step(env, action_text)` | `env.step(action_text)` | Action needs translation (text → int, etc.) |
| `_extract_info(info)` | Filters to scalar values | You want specific keys in result.json |

### Step 5: Implement the PromptBuilder (optional)

For LLM-powered evaluation, create a task-specific PromptBuilder:

```python
"""Prompt builder for My Benchmark."""
from __future__ import annotations

import json

from easi.agents.prompt_builder import validate_action_name, _encode_image_base64
from easi.core.episode import Action
from easi.core.memory import AgentMemory


class MyPromptBuilder:
    """Builds prompts for My Benchmark's ReAct agent."""

    def __init__(self, n_shot=3, use_feedback=True):
        self.n_shot = n_shot
        self.use_feedback = use_feedback

    def build_messages(self, memory: AgentMemory) -> list[dict]:
        """Build LLM messages from agent memory.

        Args:
            memory: AgentMemory with task_description, action_space,
                    current_observation, steps (history), action_history.

        Returns:
            List of message dicts: [{"role": "system", "content": [...]}, ...]
        """
        messages = []

        # System message with instructions
        system_text = f"You are an agent. Task: {memory.task_description}\n"
        system_text += f"Actions: {', '.join(memory.action_space)}"
        messages.append({"role": "system", "content": [{"type": "text", "text": system_text}]})

        # Current observation (with image)
        user_content = []
        if memory.current_observation and memory.current_observation.rgb_path:
            img_b64 = _encode_image_base64(memory.current_observation.rgb_path)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
            })
        user_content.append({"type": "text", "text": "What action should you take?"})
        messages.append({"role": "user", "content": user_content})

        return messages

    def parse_response(self, llm_response: str, memory: AgentMemory) -> list[Action]:
        """Parse LLM text response into validated Action objects.

        Returns:
            List of Action objects. Empty list = parsing failed.
        """
        try:
            data = json.loads(llm_response)
        except json.JSONDecodeError:
            return []

        plan = data.get("executable_plan", [])
        actions = []
        for entry in plan:
            name = entry.get("action_name", "")
            validated = validate_action_name(name, memory.action_space)
            if validated:
                actions.append(Action(action_name=validated))
            else:
                break  # Stop at first invalid action
        return actions
```

### Step 6: Write Tests

Follow the pattern in existing test files. All tests run offline (no simulator, no LLM).

```python
"""Tests for My Benchmark task (offline, no simulator needed)."""
import pytest
from pathlib import Path

from easi.core.episode import Observation, StepResult


class TestMyBenchmarkTask:
    @pytest.fixture
    def task(self):
        from easi.tasks.my_benchmark.task import MyBenchmarkTask
        return MyBenchmarkTask()

    def test_name(self, task):
        assert task.name == "my_benchmark_base"

    def test_simulator_key(self, task):
        assert task.simulator_key == "ai2thor:v5_0_0"

    def test_action_space(self, task):
        assert len(task.action_space) > 0

    def test_max_steps(self, task):
        assert task.max_steps == 50

    def test_format_reset_config(self, task):
        episode = {"id": 0, "scene": "TestScene", "instruction": "test"}
        config = task.format_reset_config(episode)
        assert "scene" in config

    def test_evaluate_episode(self, task):
        obs = Observation(rgb_path="/tmp/fake.png")
        trajectory = [
            StepResult(observation=obs, reward=0.0, done=True,
                       info={"task_success": 1.0}),
        ]
        result = task.evaluate_episode({"id": 0}, trajectory)
        assert "task_success" in result

    def test_evaluate_empty_trajectory(self, task):
        result = task.evaluate_episode({}, [])
        assert result["task_success"] == 0.0

    def test_bridge_script_path(self, task):
        path = task.get_bridge_script_path()
        if path is not None:
            assert path.exists()

    def test_registry_discovers_task(self):
        from easi.tasks.registry import list_tasks
        tasks = list_tasks()
        assert "my_benchmark_base" in tasks
```

Run tests: `pytest tests/test_my_benchmark.py -v`

### Step 7: Verify

```bash
# All existing tests still pass
pytest tests/ -v --timeout=60

# Registry discovers your task
easi task list | grep my_benchmark

# Task info looks correct
easi task info my_benchmark_base

# Dummy agent smoke test (no LLM needed)
easi start my_benchmark_base --agent dummy --max-episodes 1
```

---

## Adding a New Simulator

Less common. Only needed when a benchmark uses a simulator not yet in EASI.

### Structure

```
easi/simulators/<simulator_name>/
├── __init__.py
├── manifest.yaml                    # Declares name, versions, classes
└── <version>/
    ├── __init__.py
    ├── simulator.py                 # Subclass of BaseSimulator
    ├── env_manager.py               # Subclass of BaseEnvironmentManager
    ├── bridge.py                    # Default bridge script
    ├── conda_env.yaml               # Conda environment spec
    └── requirements.txt             # Pip dependencies
```

### manifest.yaml

```yaml
name: my_sim
display_name: "My Simulator"
default_version: "v1_0_0"
versions:
  v1_0_0:
    description: "My Simulator 1.0.0"
    simulator_class: "easi.simulators.my_sim.v1_0_0.simulator.MySimSimulator"
    env_manager_class: "easi.simulators.my_sim.v1_0_0.env_manager.MySimEnvManager"
    python_version: "3.10"
```

### simulator.py

```python
from pathlib import Path
from easi.core.base_simulator import BaseSimulator

class MySimSimulator(BaseSimulator):
    @property
    def name(self) -> str:
        return "my_sim"

    @property
    def version(self) -> str:
        return "v1_0_0"

    def _get_bridge_script_path(self) -> Path:
        return Path(__file__).parent / "bridge.py"
```

### env_manager.py

```python
from pathlib import Path
from easi.core.base_env_manager import BaseEnvironmentManager

class MySimEnvManager(BaseEnvironmentManager):
    @property
    def simulator_name(self) -> str:
        return "my_sim"

    @property
    def version(self) -> str:
        return "v1_0_0"

    @property
    def needs_display(self) -> bool:
        return True  # Set True if simulator needs X11/Xvfb

    def get_conda_env_yaml_path(self) -> Path:
        return Path(__file__).parent / "conda_env.yaml"

    def get_requirements_txt_path(self) -> Path:
        return Path(__file__).parent / "requirements.txt"

    def get_system_deps(self) -> list[str]:
        return ["conda"]  # Add "xvfb" if needs_display is True

    def get_validation_import(self) -> str:
        return "from my_sim import Controller; print('ok')"
```

### Verify

```bash
easi env list             # Should show my_sim
easi env install my_sim   # Install conda env
easi env check my_sim     # Validate
easi sim test my_sim      # Smoke test bridge
```

---

## Vendoring Benchmark Code

When integrating an external benchmark (e.g., from EmbodiedBench), vendor only the environment code you need:

1. **Create `vendor/` directory** in your task folder
2. **Copy only the env class** (not the full benchmark runner/evaluator)
3. **Remove external dependencies** the benchmark used for logging, dataset loading, gym registration — EASI handles all of these
4. **Adapt the interface**:
   - `reset(episode)` accepts an episode dict (from EASI's dataset)
   - `step(action)` returns `(obs, reward, done, info)` tuple
   - Remove internal image saving (bridge handles this)
   - Remove internal logging (EASI's logger handles this)

---

## Key Conventions

### Logging

```python
from easi.utils.logging import get_logger
logger = get_logger(__name__)
# Use logger.info(), logger.warning(), logger.error()
# Use logger.trace() for verbose debug output
# NEVER use print()
```

### Imports

```python
from __future__ import annotations  # Always first import
```

### Testing

- All tests run offline (mock simulators, no LLM calls)
- Use `Observation(rgb_path="/tmp/fake.png")` for test observations
- Use `StepResult(observation=obs, done=True, info={...})` for test trajectories
- Test file naming: `tests/test_<task_name>.py`

### summary.json Structure

```json
{
  "num_episodes": 100,
  "model": "gpt-4o",
  "backend": "openai",
  "llm_usage": {"total_calls": 500, "total_tokens": 150000},
  "metrics": {
    "success_rate": 0.73,
    "avg_num_steps": 24.3,
    "avg_task_success": 0.73
  }
}
```

Metrics (from `task.aggregate_results()`) are nested under `"metrics"`. Run metadata stays at the top level.

---

## Existing Benchmarks Reference

| Task | Simulator | Splits | Action Type | Max Steps |
|---|---|---|---|---|
| `dummy_task` | `dummy:v1` | 1 | 4 discrete text | 100 |
| `ebalfred_*` | `ai2thor:v2_1_0` | 6 | ~133 skill text | 50 |
| `ebnavigation_*` | `ai2thor:v5_0_0` | 5 | 8 discrete int | 20 |
| `ebhabitat_*` | `habitat_sim:v0_3_0` | 4 | 4 discrete text | varies |
| `ebmanipulation_*` | `coppeliasim:v4_1_0` | 4 | continuous params | varies |

Use the closest existing task as a template when adding a new one. The `dummy_task` is the simplest reference; `ebalfred` is the most complete.
