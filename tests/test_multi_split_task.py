"""Tests for multi-split task support."""
import json
import zipfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from easi.tasks.registry import list_tasks, get_task_entry, refresh, _discover_tasks


class TestMultiSplitRegistry:
    def test_discovers_multiple_yaml_per_folder(self, tmp_path):
        """Registry should find all .yaml files in a task folder."""
        task_dir = tmp_path / "my_bench"
        task_dir.mkdir()
        (task_dir / "__init__.py").write_text("")

        for split in ["base", "hard"]:
            yaml_content = f"""
name: my_bench_{split}
display_name: "My Bench ({split})"
description: "Test split"
simulator: "dummy:v1"
task_class: "easi.tasks.dummy_task.task.DummyTask"
action_space: ["move", "turn"]
max_steps: 100
dataset:
  source: huggingface
  repo_id: "oscarqjh/my-bench"
  subset: null
  split: "{split}"
"""
            (task_dir / f"my_bench_{split}.yaml").write_text(yaml_content)

        entries = _discover_tasks(tasks_dir=tmp_path)

        assert "my_bench_base" in entries
        assert "my_bench_hard" in entries
        assert entries["my_bench_base"].config_path.name == "my_bench_base.yaml"

    def test_yaml_with_explicit_subset(self, tmp_path):
        """Task yaml can specify an explicit HF subset."""
        task_dir = tmp_path / "bench"
        task_dir.mkdir()
        (task_dir / "__init__.py").write_text("")
        (task_dir / "bench_en_test.yaml").write_text("""
name: bench_en_test
display_name: "Bench EN Test"
description: "English test subset"
simulator: "dummy:v1"
task_class: "easi.tasks.dummy_task.task.DummyTask"
action_space: []
max_steps: 10
dataset:
  source: huggingface
  repo_id: "oscarqjh/bench"
  subset: "en"
  split: "test"
""")
        entries = _discover_tasks(tasks_dir=tmp_path)
        assert entries["bench_en_test"].name == "bench_en_test"

    def test_skips_dirs_starting_with_underscore(self, tmp_path):
        """Directories starting with _ should be skipped."""
        hidden_dir = tmp_path / "_internal"
        hidden_dir.mkdir()
        (hidden_dir / "secret.yaml").write_text("""
name: secret_task
simulator: "dummy:v1"
task_class: "easi.tasks.dummy_task.task.DummyTask"
""")
        entries = _discover_tasks(tasks_dir=tmp_path)
        assert "secret_task" not in entries

    def test_skips_invalid_yaml(self, tmp_path):
        """Invalid YAML files should be skipped without crashing."""
        task_dir = tmp_path / "broken"
        task_dir.mkdir()
        (task_dir / "broken.yaml").write_text("this is: [not: valid: yaml: {{")

        entries = _discover_tasks(tasks_dir=tmp_path)
        assert len(entries) == 0

    def test_skips_yaml_without_name(self, tmp_path):
        """YAML files without a 'name' key should be skipped."""
        task_dir = tmp_path / "no_name"
        task_dir.mkdir()
        (task_dir / "no_name.yaml").write_text("""
description: "Missing name field"
simulator: "dummy:v1"
""")
        entries = _discover_tasks(tasks_dir=tmp_path)
        assert len(entries) == 0

    def test_existing_dummy_task_still_discovered(self):
        """The existing dummy_task/task.yaml should still be found by *.yaml glob."""
        refresh()
        tasks = list_tasks()
        assert "dummy_task" in tasks
        entry = get_task_entry("dummy_task")
        assert entry.simulator_key == "dummy:v1"
        assert entry.config_path.name == "task.yaml"


class TestZipExtraction:
    def test_extract_zip_files(self, tmp_path):
        """BaseTask._extract_zip_files should extract .zip files."""
        from easi.core.base_task import BaseTask

        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        zip_path = dataset_dir / "tasks.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("tasks/pick_and_place_simple-Mug/trial_001/traj_data.json", '{"test": true}')
            zf.writestr("tasks/pick_clean-Plate/trial_002/traj_data.json", '{"test": true}')

        BaseTask._extract_zip_files(dataset_dir, ["tasks.zip"])

        assert (dataset_dir / "tasks" / "pick_and_place_simple-Mug" / "trial_001" / "traj_data.json").exists()
        assert (dataset_dir / "tasks" / "pick_clean-Plate" / "trial_002" / "traj_data.json").exists()

    def test_extract_idempotent(self, tmp_path):
        """Second extraction should be skipped (marker file)."""
        from easi.core.base_task import BaseTask

        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        zip_path = dataset_dir / "tasks.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("tasks/test.json", '{}')

        BaseTask._extract_zip_files(dataset_dir, ["tasks.zip"])
        BaseTask._extract_zip_files(dataset_dir, ["tasks.zip"])  # should be no-op

        assert (dataset_dir / ".tasks.zip.extracted").exists()

    def test_missing_zip_does_not_crash(self, tmp_path):
        """Missing zip file should warn but not crash."""
        from easi.core.base_task import BaseTask

        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()

        # Should not raise
        BaseTask._extract_zip_files(dataset_dir, ["nonexistent.zip"])


class TestEpisodeFromHFRow:
    def test_hf_row_to_episode(self):
        """Each HF dataset row should map to an episode dict."""
        from easi.core.base_task import hf_row_to_episode

        # Real EB-Alfred_easi column structure
        row = {
            "id": 0,
            "task": "pick_and_place_simple-Mug-None-Shelf-1/trial_T20190001",
            "repeat_idx": 0,
            "instruction": "Put a mug on the shelf.",
            "task_type": "pick_and_place_simple",
            "trial_id": "trial_T20190001",
        }
        episode = hf_row_to_episode(row)
        assert episode["id"] == 0
        assert episode["task"] == "pick_and_place_simple-Mug-None-Shelf-1/trial_T20190001"
        assert episode["instruction"] == "Put a mug on the shelf."
        assert episode["task_type"] == "pick_and_place_simple"
        assert episode["trial_id"] == "trial_T20190001"

    def test_hf_row_to_episode_is_copy(self):
        """hf_row_to_episode should return a copy, not modify the original."""
        from easi.core.base_task import hf_row_to_episode

        row = {"id": 1, "instruction": "test"}
        episode = hf_row_to_episode(row)
        episode["new_field"] = "added"
        assert "new_field" not in row
