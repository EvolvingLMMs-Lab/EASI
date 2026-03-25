# REVERIE-CE Integration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add REVERIE-CE as a navigation-only task in EASI, reusing the VLN-CE R2R infrastructure.

**Architecture:** REVERIE-CE is a thin task layer inheriting from VLN-CE R2R. Same simulator (`habitat_sim:v0_1_7`), same action space (4 discrete), same metrics (SR/SPL/NE/NDTW/SDTW). Only the prompt builder and dataset are new.

**Tech Stack:** Python 3.10+ (host), Habitat-Sim 0.1.7 (bridge subprocess, Python 3.8), HuggingFace datasets, Matterport3D scenes.

**Spec:** `docs/superpowers/specs/2026-03-12-reverie-ce-design.md`

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `easi/tasks/reverie_ce/__init__.py` | Package marker |
| Create | `easi/tasks/reverie_ce/task.py` | `ReverieCETask` — inherits `VLNCETask`, overrides paths |
| Create | `easi/tasks/reverie_ce/bridge.py` | `ReverieCEBridge` — inherits `VLNCEBridge`, standalone entry point |
| Create | `easi/tasks/reverie_ce/prompts.py` | `ReverieCEPromptBuilder` — high-level instruction prompt |
| Create | `easi/tasks/reverie_ce/actions.py` | Re-exports `get_action_space` from `vlnce_r2r.actions` |
| Create | `easi/tasks/reverie_ce/_base.yaml` | Task config pointing to REVERIE-CE HuggingFace repo |
| Create | `easi/tasks/reverie_ce/reverie_ce_val_unseen.yaml` | Val unseen split |
| Create | `easi/tasks/reverie_ce/reverie_ce_test.yaml` | Test split |
| Create | `tests/test_reverie_ce_task.py` | Unit tests for task, prompt builder, action space |
| None | `easi/tasks/vlnce_r2r/` | Imported, not modified |
| None | `easi/simulators/habitat_sim/` | Unchanged |

---

## Chunk 1: Data Preparation

### Task 1: Reformat Dynam3D Data into EASI HuggingFace Repo

This is a one-time manual/scripted step done outside the EASI codebase. It prepares the dataset that the task will consume.

**Files:**
- External: HuggingFace repo `oscarqjh/REVERIE-CE_easi`

- [ ] **Step 1: Download Dynam3D REVERIE-CE data**

Download from HuggingFace `MrZihanWang/Dynam3D`:
```bash
# Download the REVERIE-CE specific files
huggingface-cli download MrZihanWang/Dynam3D \
  data/datasets/reverie_training_data \
  data/datasets/reverie_val_unseen_data.json \
  data/datasets/reverie_test_data.json \
  data/datasets/reverie_val_unseen_gt.json \
  data/datasets/reverie_test_gt.json \
  --local-dir ./dynam3d_download
```

- [ ] **Step 2: Write a conversion script to reshape into JSONL**

Create a temporary script (not committed to EASI) that:
1. Reads the per-scene JSON training files from `reverie_training_data/`
2. Reads the val/test single JSON files
3. Normalises each episode into the VLN-CE R2R JSONL format:

```python
# Expected output format per line:
{
    "episode_id": str(item["episode_id"]),
    "scene_id": item["scene_id"].replace("mp3d/", "").replace(".glb", "").split("/")[-1],
    "instruction": item["instruction"]["instruction_text"],
    "start_position": item["start_position"],
    "start_rotation": item["start_rotation"],
    "goal_position": item["goals"][0]["position"],
    "geodesic_distance": item["info"]["geodesic_distance"],
    "gt_locations": item["reference_path"]
}
```

Note: `scene_id` in VLN-CE R2R format is just the scan name (e.g. `cV4RVeZvu5T`), not the full path. The bridge constructs the full path from `data_dir + mp3d/ + scene_id + scene_id.glb`.

4. Writes `train.jsonl`, `val_unseen.jsonl`, `test.jsonl`
5. Copies ground truth files as-is

- [ ] **Step 3: Verify episode counts match Dynam3D source**

```bash
wc -l data/train.jsonl data/val_unseen.jsonl data/test.jsonl
```

- [ ] **Step 4: Upload to HuggingFace**

```bash
huggingface-cli upload oscarqjh/REVERIE-CE_easi ./reverie_ce_easi/ \
  --repo-type dataset
```

Include `mp3d_scenes.zip` (same Matterport3D scenes as R2R — can be copied from the R2R dataset repo).

---

## Chunk 2: Task Implementation

### Task 2: Create Action Space Module

**Files:**
- Create: `easi/tasks/reverie_ce/actions.py`
- Test: `tests/test_reverie_ce_task.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_reverie_ce_task.py
class TestActionSpace:
    def test_has_four_actions(self):
        from easi.tasks.reverie_ce.actions import get_action_space
        actions = get_action_space()
        assert len(actions) == 4
        assert "move_forward" in actions
        assert "turn_left" in actions
        assert "turn_right" in actions
        assert "stop" in actions
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_reverie_ce_task.py::TestActionSpace -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Create the module**

```python
# easi/tasks/reverie_ce/__init__.py
# (empty)

# easi/tasks/reverie_ce/actions.py
"""REVERIE-CE action space — same as VLN-CE R2R."""
from easi.tasks.vlnce_r2r.actions import get_action_space  # noqa: F401
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_reverie_ce_task.py::TestActionSpace -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add easi/tasks/reverie_ce/__init__.py easi/tasks/reverie_ce/actions.py tests/test_reverie_ce_task.py
git commit -m "feat(reverie-ce): add action space module"
```

---

### Task 3: Create Task Class

**Files:**
- Create: `easi/tasks/reverie_ce/task.py`
- Test: `tests/test_reverie_ce_task.py`

- [ ] **Step 1: Write failing tests**

```python
# Add to tests/test_reverie_ce_task.py
import json
import pytest
from unittest.mock import MagicMock
from easi.core.episode import EpisodeRecord, Observation, StepResult


class TestReverieCETask:
    @pytest.fixture
    def task(self):
        from easi.tasks.reverie_ce.task import ReverieCETask
        mock_config = {
            "name": "reverie_ce_val_unseen",
            "display_name": "REVERIE-CE Val Unseen",
            "simulator": "habitat_sim:v0_1_7",
            "task_class": "easi.tasks.reverie_ce.task.ReverieCETask",
            "max_steps": 500,
            "dataset": {"source": "huggingface", "repo_id": "oscarqjh/REVERIE-CE_easi", "split": "val_unseen"},
            "simulator_configs": {},
            "agent": {"prompt_builder": "easi.tasks.reverie_ce.prompts.ReverieCEPromptBuilder"},
        }
        task = ReverieCETask.__new__(ReverieCETask)
        task._config = mock_config
        task._yaml_path = None
        task._action_space = None
        return task

    def test_action_space(self, task):
        actions = task._build_action_space()
        assert actions == ["move_forward", "turn_left", "turn_right", "stop"]

    def test_format_reset_config(self, task):
        episode = {
            "episode_id": "50001",
            "scene_id": "cV4RVeZvu5T",
            "instruction": "Go to the laundry room and get the cushion",
            "start_position": [1.0, 0.5, -2.0],
            "start_rotation": [0, 0.707, 0, 0.707],
            "goal_position": [4.5, 0.5, 1.2],
            "geodesic_distance": 10.5,
            "gt_locations": [[1.0, 0.5, -2.0], [2.0, 0.5, -1.0]],
            "_data_dir": "/data/reverie_ce",
        }
        config = task.format_reset_config(episode)
        assert config["scene_id"] == "cV4RVeZvu5T"
        assert config["data_dir"] == "/data/reverie_ce"
        assert json.loads(config["start_position"]) == [1.0, 0.5, -2.0]

    def test_evaluate_episode_success(self, task):
        info = {
            "success": 1.0, "oracle_success": 1.0, "spl": 0.8,
            "navigation_error": 1.5, "ndtw": 0.9, "sdtw": 0.85,
            "path_length": 8.0,
        }
        obs = Observation(rgb_path="/tmp/step.png")
        step = StepResult(observation=obs, done=True, info=info)
        result = task.evaluate_episode({}, [step])
        assert result["success"] == 1.0
        assert result["spl"] == 0.8

    def test_evaluate_episode_empty(self, task):
        result = task.evaluate_episode({}, [])
        assert result["success"] is None
        assert result["path_length"] == 0.0

    def test_aggregate_results(self, task):
        records = [
            EpisodeRecord(episode={}, trajectory=[], episode_results={
                "success": 1.0, "oracle_success": 1.0, "spl": 0.8,
                "navigation_error": 1.5, "ndtw": 0.9, "sdtw": 0.85,
                "path_length": 8.0, "steps_taken": 30.0,
            }),
            EpisodeRecord(episode={}, trajectory=[], episode_results={
                "success": 0.0, "oracle_success": 0.0, "spl": 0.0,
                "navigation_error": 6.0, "ndtw": 0.3, "sdtw": 0.0,
                "path_length": 12.0, "steps_taken": 50.0,
            }),
        ]
        summary = task.aggregate_results(records)
        assert summary["num_episodes"] == 2
        assert summary["SR"] == 0.5
        assert summary["SPL"] == 0.4

    def test_bridge_script_path(self, task):
        path = task.get_bridge_script_path()
        assert path.name == "bridge.py"
        assert "reverie_ce" in str(path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_reverie_ce_task.py::TestReverieCETask -v`
Expected: FAIL (ReverieCETask not found)

- [ ] **Step 3: Implement the task class**

```python
# easi/tasks/reverie_ce/task.py
"""REVERIE-CE task for EASI.

Navigation-only evaluation of REVERIE in continuous environments.
Inherits from VLNCETask — same metrics, same bridge protocol.
"""
from __future__ import annotations

from pathlib import Path

from easi.tasks.vlnce_r2r.task import VLNCETask
from easi.utils.logging import get_logger

logger = get_logger(__name__)


class ReverieCETask(VLNCETask):

    def get_task_yaml_path(self) -> Path:
        return Path(__file__).parent / "_base.yaml"

    def get_bridge_script_path(self) -> Path:
        return Path(__file__).parent / "bridge.py"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_reverie_ce_task.py::TestReverieCETask -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add easi/tasks/reverie_ce/task.py tests/test_reverie_ce_task.py
git commit -m "feat(reverie-ce): add ReverieCETask inheriting VLNCETask"
```

---

### Task 4: Create Bridge

**Files:**
- Create: `easi/tasks/reverie_ce/bridge.py`

The bridge must be a standalone script (runs in Python 3.8 subprocess). It inherits from `VLNCEBridge` but provides its own `__main__` entry point.

- [ ] **Step 1: Create the bridge module**

```python
# easi/tasks/reverie_ce/bridge.py
"""REVERIE-CE bridge — inherits VLN-CE R2R bridge.

Runs inside the easi_habitat_sim_v0_1_7 conda env (Python 3.8).
Inherits all reset/step/extract logic from VLNCEBridge.

Usage:
    python bridge.py --workspace /tmp/easi_xxx [--simulator-kwargs '{}']
"""
from __future__ import annotations

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from easi.tasks.vlnce_r2r.bridge import VLNCEBridge  # noqa: E402


class ReverieCEBridge(VLNCEBridge):
    """Bridge for REVERIE-CE. Identical to VLN-CE R2R for now."""
    pass


if __name__ == "__main__":
    ReverieCEBridge.main()
```

- [ ] **Step 2: Verify bridge script path resolves correctly**

The test from Task 3 (`test_bridge_script_path`) already validates this.

Run: `.venv/bin/pytest tests/test_reverie_ce_task.py::TestReverieCETask::test_bridge_script_path -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add easi/tasks/reverie_ce/bridge.py
git commit -m "feat(reverie-ce): add bridge inheriting VLNCEBridge"
```

---

### Task 5: Create Prompt Builder

**Files:**
- Create: `easi/tasks/reverie_ce/prompts.py`
- Test: `tests/test_reverie_ce_task.py`

- [ ] **Step 1: Write failing tests**

```python
# Add to tests/test_reverie_ce_task.py
class TestReverieCEPromptBuilder:
    @pytest.fixture
    def mock_encode(self):
        # Must patch in vlnce_r2r.prompts where the function is actually
        # called (super().build_messages() resolves it there, not in
        # reverie_ce.prompts).
        import easi.tasks.vlnce_r2r.prompts as prompts_mod
        original = prompts_mod._encode_image_base64
        prompts_mod._encode_image_base64 = lambda x: "data:image/png;base64,AAAA"
        yield
        prompts_mod._encode_image_base64 = original

    def _make_memory(self, action_history=None):
        memory = MagicMock()
        memory.task_description = "Go to the laundry room and bring me the blue cushion"
        memory.action_space = ["move_forward", "turn_left", "turn_right", "stop"]
        memory.current_observation = Observation(
            rgb_path="/tmp/test.png",
            metadata={"geo_distance": "5.3"},
        )
        memory.action_history = action_history or []
        memory.steps = []
        return memory

    def test_system_prompt_mentions_high_level(self, mock_encode):
        from easi.tasks.reverie_ce.prompts import ReverieCEPromptBuilder
        builder = ReverieCEPromptBuilder()
        memory = self._make_memory()
        messages = builder.build_messages(memory)
        system_msg = messages[0]["content"]
        assert "high-level" in system_msg.lower() or "described location" in system_msg.lower()

    def test_build_messages_has_image(self, mock_encode):
        from easi.tasks.reverie_ce.prompts import ReverieCEPromptBuilder
        builder = ReverieCEPromptBuilder()
        memory = self._make_memory()
        messages = builder.build_messages(memory)
        user_content = messages[1]["content"]
        image_blocks = [b for b in user_content if b.get("type") == "image_url"]
        assert len(image_blocks) == 1

    def test_build_messages_has_instruction(self, mock_encode):
        from easi.tasks.reverie_ce.prompts import ReverieCEPromptBuilder
        builder = ReverieCEPromptBuilder()
        memory = self._make_memory()
        messages = builder.build_messages(memory)
        text_blocks = [b["text"] for b in messages[1]["content"] if b.get("type") == "text"]
        full_text = "\n".join(text_blocks)
        assert "laundry room" in full_text

    def test_build_messages_has_distance(self, mock_encode):
        from easi.tasks.reverie_ce.prompts import ReverieCEPromptBuilder
        builder = ReverieCEPromptBuilder()
        memory = self._make_memory()
        messages = builder.build_messages(memory)
        text_blocks = [b["text"] for b in messages[1]["content"] if b.get("type") == "text"]
        full_text = "\n".join(text_blocks)
        assert "5.3" in full_text

    def test_parse_response_valid(self):
        from easi.tasks.reverie_ce.prompts import ReverieCEPromptBuilder
        builder = ReverieCEPromptBuilder()
        memory = MagicMock()
        memory.action_space = ["move_forward", "turn_left", "turn_right", "stop"]
        response = json.dumps({
            "visual_state_description": "I see a hallway",
            "reasoning_and_reflection": "Need to find the laundry room",
            "language_plan": "Move forward",
            "executable_plan": [{"action": "move_forward"}],
        })
        actions = builder.parse_response(response, memory)
        assert len(actions) == 1
        assert actions[0].action_name == "move_forward"

    def test_parse_response_invalid_json(self):
        from easi.tasks.reverie_ce.prompts import ReverieCEPromptBuilder
        builder = ReverieCEPromptBuilder()
        memory = MagicMock()
        memory.action_space = ["move_forward", "turn_left", "turn_right", "stop"]
        actions = builder.parse_response("not json", memory)
        assert actions == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_reverie_ce_task.py::TestReverieCEPromptBuilder -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement the prompt builder**

```python
# easi/tasks/reverie_ce/prompts.py
"""REVERIE-CE prompt builder.

Adapted from VLN-CE R2R for REVERIE's high-level instruction style.
REVERIE instructions describe a target location/object rather than
step-by-step route directions.
"""
from __future__ import annotations

from easi.tasks.vlnce_r2r.prompts import VLNCEPromptBuilder
from easi.utils.logging import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """\
## Role and Environment
You are a robot navigating in a 3D indoor environment. You observe the \
environment through a front-facing camera and must navigate to the location \
described in a high-level natural language instruction.

## Observation Description
- **Distance to goal**: Geodesic (shortest walkable path) distance in meters \
to the described location. Decreases as you get closer.

## Available Actions
- move_forward: Move forward by 0.25 meters
- turn_left: Turn left by 15 degrees
- turn_right: Turn right by 15 degrees
- stop: Stop and end navigation (use ONLY when you believe you have reached \
the described location)

## Strategy
1. Read the instruction carefully — it describes a target location or object \
in the environment, not a step-by-step route
2. Observe your surroundings in the image
3. Reason about which direction the described location is likely in
4. Navigate room by room, using landmarks and room types to orient yourself
5. Use stop ONLY when you are confident you have reached the described location

## Guidelines
1. Always output at least one action in executable_plan.
2. Only use actions from the Available Actions list.
3. If previous actions failed, reason about why and try a different approach.
4. Do not repeatedly execute the same action sequence.
5. Keep your plan efficient and concise.

## Response Format
Output a JSON object with exactly these 4 fields:
{
    "visual_state_description": "Describe what you see in the current image",
    "reasoning_and_reflection": "Reason about your situation, reflect on \
history and feedback",
    "language_plan": "Describe your next navigation plan in natural language",
    "executable_plan": [{"action": "<action_name>"}]
}

You may include multiple actions in executable_plan. Actions execute \
sequentially."""


class ReverieCEPromptBuilder(VLNCEPromptBuilder):
    """Prompt builder for REVERIE-CE benchmark.

    Inherits message construction and response parsing from VLNCEPromptBuilder.
    Overrides only the system prompt to frame the task around high-level
    instructions rather than step-by-step route following.
    """

    def build_messages(self, memory):
        # Use parent's build_messages but swap the system prompt
        messages = super().build_messages(memory)
        messages[0]["content"] = SYSTEM_PROMPT
        return messages
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_reverie_ce_task.py::TestReverieCEPromptBuilder -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add easi/tasks/reverie_ce/prompts.py tests/test_reverie_ce_task.py
git commit -m "feat(reverie-ce): add prompt builder for high-level instructions"
```

---

## Chunk 3: YAML Configs and Integration

### Task 6: Create YAML Configs

**Files:**
- Create: `easi/tasks/reverie_ce/_base.yaml`
- Create: `easi/tasks/reverie_ce/reverie_ce_val_unseen.yaml`
- Create: `easi/tasks/reverie_ce/reverie_ce_test.yaml`

- [ ] **Step 1: Create _base.yaml**

```yaml
# easi/tasks/reverie_ce/_base.yaml
display_name: "REVERIE-CE"
description: "REVERIE in Continuous Environments (navigation-only)"
simulator: "habitat_sim:v0_1_7"
task_class: "easi.tasks.reverie_ce.task.ReverieCETask"
max_steps: 500

dataset:
  source: huggingface
  repo_id: "oscarqjh/REVERIE-CE_easi"
  subset: null
  hf_data_dir: "data"
  zip_files:
    - "mp3d_scenes.zip"

simulator_configs:
  render_platform: auto
  screen_height: 480
  screen_width: 480
  hfov: 90
  sensor_height: 1.25
  gpu_device_id: 0
  success_distance: 3.0
  forward_step_size: 0.25
  turn_angle: 15
  allow_sliding: true
  additional_deps:
    - "fastdtw>=0.3.4"

agent:
  prompt_builder: "easi.tasks.reverie_ce.prompts.ReverieCEPromptBuilder"
  prompt_builder_kwargs:
    use_feedback: true
    use_geo_distance: true
    action_history_len: 20
    chat_history: false
    message_window_len: 5
  generation_kwargs:
    temperature: 0
    max_tokens: 4096
    top_p: 0.95
```

- [ ] **Step 2: Create split configs**

```yaml
# easi/tasks/reverie_ce/reverie_ce_val_unseen.yaml
extends: _base.yaml
name: reverie_ce_val_unseen
display_name: "REVERIE-CE Val Unseen"
description: "REVERIE-CE validation split (unseen environments)"
dataset:
  split: "val_unseen"
```

```yaml
# easi/tasks/reverie_ce/reverie_ce_test.yaml
extends: _base.yaml
name: reverie_ce_test
display_name: "REVERIE-CE Test"
description: "REVERIE-CE test split"
dataset:
  split: "test"
```

- [ ] **Step 3: Verify task discovery**

Run: `.venv/bin/easi task list 2>&1 | grep -i reverie`
Expected: Should show `reverie_ce_val_unseen` and `reverie_ce_test`

- [ ] **Step 4: Commit**

```bash
git add easi/tasks/reverie_ce/_base.yaml easi/tasks/reverie_ce/reverie_ce_val_unseen.yaml easi/tasks/reverie_ce/reverie_ce_test.yaml
git commit -m "feat(reverie-ce): add YAML task configs for val_unseen and test splits"
```

---

### Task 7: Run Full Test Suite

- [ ] **Step 1: Run all existing tests to verify no regressions**

Run: `.venv/bin/pytest tests/ -v --timeout=60`
Expected: All tests pass (946+)

- [ ] **Step 2: Run REVERIE-CE specific tests**

Run: `.venv/bin/pytest tests/test_reverie_ce_task.py -v`
Expected: All REVERIE-CE tests pass

- [ ] **Step 3: Final commit if any test fixes needed**

```bash
git add -A
git commit -m "fix(reverie-ce): address test failures"
```

---

## Post-Implementation

After all tasks are complete:

1. **Data preparation** (Task 1) must be done separately — download Dynam3D data, reformat to JSONL, upload to `oscarqjh/REVERIE-CE_easi`
2. **End-to-end smoke test** once the HuggingFace repo is ready:
   ```bash
   easi task download reverie_ce_val_unseen
   easi sim test habitat_sim:v0_1_7
   easi start reverie_ce_val_unseen --agent dummy --max-episodes 1
   ```
