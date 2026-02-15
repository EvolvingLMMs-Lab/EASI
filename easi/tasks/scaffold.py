"""Task scaffolding: generates boilerplate for new benchmark integrations."""

from __future__ import annotations

from pathlib import Path

from easi.utils.logging import get_logger

logger = get_logger(__name__)


def _to_class_name(snake_name: str) -> str:
    """Convert snake_case to PascalCase. e.g. 'my_benchmark' -> 'MyBenchmark'."""
    return "".join(word.capitalize() for word in snake_name.split("_"))


def scaffold_task(
    name: str,
    simulator: str,
    output_dir: Path | None = None,
    max_steps: int = 50,
    tests_dir: Path | None = None,
) -> Path:
    """Generate boilerplate files for a new benchmark task.

    Creates:
        <output_dir>/<name>/
            __init__.py
            task.py         — BaseTask subclass with TODOs
            bridge.py       — BaseBridge subclass with TODOs
            <name>.yaml     — Task config

    Optionally creates:
        <tests_dir>/test_<name>.py — Test file with import and basic checks

    Args:
        name: Task name in snake_case (e.g., 'my_benchmark').
        simulator: Simulator key (e.g., 'ai2thor:v2_1_0', 'dummy:v1').
        output_dir: Parent directory for task folder. Defaults to easi/tasks/.
        max_steps: Default max steps per episode.
        tests_dir: Directory for generated test file. If None, no test generated.

    Returns:
        Path to the created task directory.

    Raises:
        FileExistsError: If the task directory already exists.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent

    task_dir = output_dir / name
    if task_dir.exists():
        raise FileExistsError(f"Task directory already exists: {task_dir}")

    task_dir.mkdir(parents=True)

    class_name = _to_class_name(name)

    # __init__.py
    (task_dir / "__init__.py").write_text("")

    # task.py
    (task_dir / "task.py").write_text(_TASK_TEMPLATE.format(
        name=name,
        class_name=class_name,
    ))

    # bridge.py
    (task_dir / "bridge.py").write_text(_BRIDGE_TEMPLATE.format(
        name=name,
        class_name=class_name,
    ))

    # <name>.yaml
    (task_dir / f"{name}.yaml").write_text(_YAML_TEMPLATE.format(
        name=name,
        class_name=class_name,
        simulator=simulator,
        max_steps=max_steps,
    ))

    # test file
    if tests_dir is not None:
        tests_dir.mkdir(parents=True, exist_ok=True)
        (tests_dir / f"test_{name}.py").write_text(_TEST_TEMPLATE.format(
            name=name,
            class_name=class_name,
        ))

    logger.info("Scaffolded task '%s' at %s", name, task_dir)
    return task_dir


_TASK_TEMPLATE = '''\
"""{class_name} task for EASI.

TODO: Implement format_reset_config() and evaluate_episode().
"""
from __future__ import annotations

from pathlib import Path

from easi.core.base_task import BaseTask
from easi.core.episode import StepResult
from easi.utils.logging import get_logger

logger = get_logger(__name__)


class {class_name}Task(BaseTask):

    def get_task_yaml_path(self) -> Path:
        return Path(__file__).parent / "{name}.yaml"

    def get_bridge_script_path(self) -> Path:
        return Path(__file__).parent / "bridge.py"

    def format_reset_config(self, episode: dict) -> dict:
        """Map dataset episode to simulator reset config.

        TODO: Return the fields your bridge needs to initialize an episode.
        """
        return episode

    def evaluate_episode(
        self, episode: dict, trajectory: list[StepResult]
    ) -> dict[str, float]:
        """Extract metrics from a completed episode trajectory.

        TODO: Implement your evaluation logic.
        """
        if not trajectory:
            return {{"task_success": 0.0, "num_steps": 0.0}}

        last_step = trajectory[-1]
        return {{
            "task_success": last_step.info.get("task_success", 0.0),
            "num_steps": float(len(trajectory)),
            "total_reward": sum(s.reward for s in trajectory),
        }}
'''

_BRIDGE_TEMPLATE = '''\
"""{class_name} bridge for EASI.

Wraps the external benchmark env in EASI's IPC protocol.

TODO: Implement _create_env() and _extract_image().

Usage:
    python bridge.py --workspace /tmp/easi_xxx --data-dir /path/to/data
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add repo root to path for easi imports (bridge runs in subprocess)
_repo_root = Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from easi.simulators.base_bridge import BaseBridge


class {class_name}Bridge(BaseBridge):
    """Wraps the external {class_name} env.

    TODO: Install the external package in the bridge conda env:
        pip install --no-deps -e ./ExternalRepo

    Then import their env class in _create_env().
    """

    def _create_env(self, reset_config, simulator_kwargs):
        """Create the benchmark env instance.

        TODO: Import and instantiate the external env class.
        Example:
            from external_package import BenchmarkEnv
            return BenchmarkEnv(**simulator_kwargs)
        """
        raise NotImplementedError("TODO: implement _create_env()")

    def _extract_image(self, obs):
        """Extract RGB numpy array (H, W, 3) from env observation.

        TODO: Return the image from your env's observation dict.
        Example:
            return obs["head_rgb"]
        """
        raise NotImplementedError("TODO: implement _extract_image()")


if __name__ == "__main__":
    {class_name}Bridge.main()
'''

_YAML_TEMPLATE = '''\
name: {name}
display_name: "{class_name}"
description: "TODO: Add description"
simulator: "{simulator}"
task_class: "easi.tasks.{name}.task.{class_name}Task"
max_steps: {max_steps}
dataset:
  source: local
  path: null
simulator_configs:
  additional_deps: []
'''

_TEST_TEMPLATE = '''\
"""Tests for {name} task and bridge."""
from pathlib import Path

from easi.tasks.{name}.task import {class_name}Task


class Test{class_name}Task:

    def test_task_is_importable(self):
        """Verify the task class can be imported."""
        assert {class_name}Task is not None

    def test_bridge_script_exists(self):
        """Verify the bridge script file exists."""
        from easi.tasks.{name}.bridge import {class_name}Bridge
        assert {class_name}Bridge is not None

    def test_get_bridge_script_path(self, tmp_path):
        """Verify get_bridge_script_path() returns the correct path."""
        yaml_path = tmp_path / "{name}.yaml"
        yaml_path.write_text(
            "name: {name}\\n"
            "display_name: {class_name}\\n"
            "description: test\\n"
            "simulator: dummy:v1\\n"
            "task_class: easi.tasks.{name}.task.{class_name}Task\\n"
            "max_steps: 10\\n"
            "dataset:\\n"
            "  source: local\\n"
            "  path: null\\n"
        )
        task = {class_name}Task(split_yaml_path=yaml_path)
        bridge_path = task.get_bridge_script_path()
        assert bridge_path is not None
        assert bridge_path.name == "bridge.py"
'''
