"""Tests for easi task scaffold command."""
from pathlib import Path

import pytest
import yaml

from easi.tasks.scaffold import scaffold_task


class TestScaffold:

    def test_creates_task_directory(self, tmp_path):
        scaffold_task("my_benchmark", "ai2thor:v2_1_0", output_dir=tmp_path / "easi" / "tasks")
        task_dir = tmp_path / "easi" / "tasks" / "my_benchmark"
        assert task_dir.is_dir()

    def test_creates_required_files(self, tmp_path):
        tasks_dir = tmp_path / "easi" / "tasks"
        scaffold_task("my_benchmark", "ai2thor:v2_1_0", output_dir=tasks_dir)
        task_dir = tasks_dir / "my_benchmark"
        assert (task_dir / "__init__.py").exists()
        assert (task_dir / "task.py").exists()
        assert (task_dir / "bridge.py").exists()
        assert (task_dir / "my_benchmark.yaml").exists()

    def test_creates_test_file(self, tmp_path):
        tasks_dir = tmp_path / "easi" / "tasks"
        scaffold_task("my_benchmark", "dummy:v1", output_dir=tasks_dir, tests_dir=tmp_path / "tests")
        assert (tmp_path / "tests" / "test_my_benchmark.py").exists()

    def test_test_file_has_class_name(self, tmp_path):
        tasks_dir = tmp_path / "easi" / "tasks"
        scaffold_task("my_benchmark", "dummy:v1", output_dir=tasks_dir, tests_dir=tmp_path / "tests")
        content = (tmp_path / "tests" / "test_my_benchmark.py").read_text()
        assert "MyBenchmarkTask" in content
        assert "MyBenchmarkBridge" in content

    def test_yaml_has_correct_simulator(self, tmp_path):
        tasks_dir = tmp_path / "easi" / "tasks"
        scaffold_task("my_benchmark", "dummy:v1", output_dir=tasks_dir)
        config = yaml.safe_load((tasks_dir / "my_benchmark" / "my_benchmark.yaml").read_text())
        assert config["simulator"] == "dummy:v1"

    def test_yaml_has_simulator_configs(self, tmp_path):
        tasks_dir = tmp_path / "easi" / "tasks"
        scaffold_task("my_benchmark", "dummy:v1", output_dir=tasks_dir)
        config = yaml.safe_load((tasks_dir / "my_benchmark" / "my_benchmark.yaml").read_text())
        assert "simulator_configs" in config
        assert "additional_deps" in config["simulator_configs"]

    def test_yaml_no_legacy_fields(self, tmp_path):
        tasks_dir = tmp_path / "easi" / "tasks"
        scaffold_task("my_benchmark", "dummy:v1", output_dir=tasks_dir)
        config = yaml.safe_load((tasks_dir / "my_benchmark" / "my_benchmark.yaml").read_text())
        assert "simulator_kwargs" not in config
        assert "external_packages" not in config

    def test_task_py_has_class_name(self, tmp_path):
        tasks_dir = tmp_path / "easi" / "tasks"
        scaffold_task("my_benchmark", "dummy:v1", output_dir=tasks_dir)
        content = (tasks_dir / "my_benchmark" / "task.py").read_text()
        assert "class MyBenchmarkTask(BaseTask):" in content

    def test_bridge_py_has_class_name(self, tmp_path):
        tasks_dir = tmp_path / "easi" / "tasks"
        scaffold_task("my_benchmark", "dummy:v1", output_dir=tasks_dir)
        content = (tasks_dir / "my_benchmark" / "bridge.py").read_text()
        assert "class MyBenchmarkBridge(BaseBridge):" in content

    def test_refuses_existing_directory(self, tmp_path):
        tasks_dir = tmp_path / "easi" / "tasks"
        scaffold_task("my_benchmark", "dummy:v1", output_dir=tasks_dir)
        with pytest.raises(FileExistsError):
            scaffold_task("my_benchmark", "dummy:v1", output_dir=tasks_dir)

    def test_bridge_imports_base_bridge(self, tmp_path):
        tasks_dir = tmp_path / "easi" / "tasks"
        scaffold_task("my_benchmark", "dummy:v1", output_dir=tasks_dir)
        content = (tasks_dir / "my_benchmark" / "bridge.py").read_text()
        assert "from easi.simulators.base_bridge import BaseBridge" in content

    def test_task_has_get_bridge_script_path(self, tmp_path):
        tasks_dir = tmp_path / "easi" / "tasks"
        scaffold_task("my_benchmark", "dummy:v1", output_dir=tasks_dir)
        content = (tasks_dir / "my_benchmark" / "task.py").read_text()
        assert "get_bridge_script_path" in content
