"""Tests for YAML template inheritance (deep_merge + resolve_task_yaml)."""
import pytest
from pathlib import Path


class TestDeepMerge:
    """Tests for deep_merge() utility."""

    def test_flat_override(self):
        from easi.tasks.yaml_utils import deep_merge
        base = {"a": 1, "b": 2}
        override = {"b": 99}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 99}

    def test_nested_dict_merge(self):
        from easi.tasks.yaml_utils import deep_merge
        base = {"agent": {"prompt_builder": "X", "kwargs": {"n_shot": 10, "split": "base"}}}
        override = {"agent": {"kwargs": {"n_shot": 7}}}
        result = deep_merge(base, override)
        assert result["agent"]["prompt_builder"] == "X"
        assert result["agent"]["kwargs"]["n_shot"] == 7
        assert result["agent"]["kwargs"]["split"] == "base"

    def test_list_replaced_not_appended(self):
        from easi.tasks.yaml_utils import deep_merge
        base = {"deps": ["gym", "networkx"]}
        override = {"deps": ["torch"]}
        result = deep_merge(base, override)
        assert result["deps"] == ["torch"]

    def test_new_keys_added(self):
        from easi.tasks.yaml_utils import deep_merge
        base = {"a": 1}
        override = {"b": 2}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 2}

    def test_does_not_mutate_inputs(self):
        from easi.tasks.yaml_utils import deep_merge
        base = {"nested": {"a": 1}}
        override = {"nested": {"b": 2}}
        deep_merge(base, override)
        assert "b" not in base["nested"]

    def test_deeply_nested_three_levels(self):
        from easi.tasks.yaml_utils import deep_merge
        base = {"l1": {"l2": {"l3": "base_val", "keep": True}}}
        override = {"l1": {"l2": {"l3": "override_val"}}}
        result = deep_merge(base, override)
        assert result["l1"]["l2"]["l3"] == "override_val"
        assert result["l1"]["l2"]["keep"] is True

    def test_override_dict_with_scalar(self):
        from easi.tasks.yaml_utils import deep_merge
        base = {"x": {"nested": True}}
        override = {"x": "flat_now"}
        result = deep_merge(base, override)
        assert result["x"] == "flat_now"

    def test_override_scalar_with_dict(self):
        from easi.tasks.yaml_utils import deep_merge
        base = {"x": "flat"}
        override = {"x": {"nested": True}}
        result = deep_merge(base, override)
        assert result["x"] == {"nested": True}

    def test_empty_override(self):
        from easi.tasks.yaml_utils import deep_merge
        base = {"a": 1, "b": 2}
        result = deep_merge(base, {})
        assert result == {"a": 1, "b": 2}

    def test_empty_base(self):
        from easi.tasks.yaml_utils import deep_merge
        result = deep_merge({}, {"a": 1})
        assert result == {"a": 1}


class TestResolveTaskYaml:
    """Tests for resolve_task_yaml() with extends inheritance."""

    def test_no_extends_returns_raw(self, tmp_path):
        from easi.tasks.yaml_utils import resolve_task_yaml
        yaml_file = tmp_path / "task.yaml"
        yaml_file.write_text("name: my_task\nsimulator: dummy:v1\n")
        config = resolve_task_yaml(yaml_file)
        assert config["name"] == "my_task"
        assert "extends" not in config

    def test_extends_merges_base(self, tmp_path):
        from easi.tasks.yaml_utils import resolve_task_yaml
        (tmp_path / "_base.yaml").write_text(
            "simulator: dummy:v1\nmax_steps: 100\n"
            "agent:\n  prompt_builder: MyBuilder\n  kwargs:\n    n_shot: 10\n"
        )
        (tmp_path / "split.yaml").write_text(
            "extends: _base.yaml\nname: my_split\n"
            "agent:\n  kwargs:\n    n_shot: 5\n"
        )
        config = resolve_task_yaml(tmp_path / "split.yaml")
        assert config["name"] == "my_split"
        assert config["simulator"] == "dummy:v1"
        assert config["max_steps"] == 100
        assert config["agent"]["prompt_builder"] == "MyBuilder"
        assert config["agent"]["kwargs"]["n_shot"] == 5

    def test_extends_strips_extends_key(self, tmp_path):
        from easi.tasks.yaml_utils import resolve_task_yaml
        (tmp_path / "_base.yaml").write_text("simulator: dummy:v1\n")
        (tmp_path / "split.yaml").write_text("extends: _base.yaml\nname: test\n")
        config = resolve_task_yaml(tmp_path / "split.yaml")
        assert "extends" not in config

    def test_extends_missing_base_raises(self, tmp_path):
        from easi.tasks.yaml_utils import resolve_task_yaml
        (tmp_path / "split.yaml").write_text("extends: nonexistent.yaml\nname: test\n")
        with pytest.raises(FileNotFoundError, match="nonexistent.yaml"):
            resolve_task_yaml(tmp_path / "split.yaml")

    def test_chained_extends_raises(self, tmp_path):
        from easi.tasks.yaml_utils import resolve_task_yaml
        (tmp_path / "grandparent.yaml").write_text("x: 1\n")
        (tmp_path / "_base.yaml").write_text("extends: grandparent.yaml\ny: 2\n")
        (tmp_path / "split.yaml").write_text("extends: _base.yaml\nname: test\n")
        with pytest.raises(ValueError, match="Chained extends not supported"):
            resolve_task_yaml(tmp_path / "split.yaml")

    def test_base_without_name_not_a_task(self, tmp_path):
        from easi.tasks.yaml_utils import resolve_task_yaml
        (tmp_path / "_base.yaml").write_text(
            "simulator: dummy:v1\nmax_steps: 50\n"
        )
        config = resolve_task_yaml(tmp_path / "_base.yaml")
        assert "name" not in config

    def test_realistic_ebmanipulation_merge(self, tmp_path):
        from easi.tasks.yaml_utils import resolve_task_yaml
        (tmp_path / "_base.yaml").write_text(
            "simulator: coppeliasim:v4_1_0\n"
            "task_class: easi.tasks.ebmanipulation.task.EBManipulationTask\n"
            "max_steps: 15\n"
            "dataset:\n"
            "  source: huggingface\n"
            "  repo_id: oscarqjh/EB-Manipulation_easi\n"
            "  subset: null\n"
            "  zip_files:\n"
            "    - simulator_data.zip\n"
            "simulator_configs:\n"
            "  screen_height: 500\n"
            "  voxel_size: 100\n"
            "agent:\n"
            "  prompt_builder: easi.tasks.ebmanipulation.prompts.EBManipulationPromptBuilder\n"
            "  prompt_builder_kwargs:\n"
            "    n_shot: 10\n"
            "    use_feedback: true\n"
            "    chat_history: false\n"
        )
        (tmp_path / "ebmanipulation_visual.yaml").write_text(
            "extends: _base.yaml\n"
            "name: ebmanipulation_visual\n"
            "display_name: EB-Manipulation Visual Split\n"
            "description: Visual split\n"
            "dataset:\n"
            "  split: visual\n"
            "agent:\n"
            "  prompt_builder_kwargs:\n"
            "    split: visual\n"
        )
        config = resolve_task_yaml(tmp_path / "ebmanipulation_visual.yaml")
        assert config["name"] == "ebmanipulation_visual"
        assert config["simulator"] == "coppeliasim:v4_1_0"
        assert config["max_steps"] == 15
        assert config["dataset"]["source"] == "huggingface"
        assert config["dataset"]["split"] == "visual"
        assert config["dataset"]["zip_files"] == ["simulator_data.zip"]
        assert config["simulator_configs"]["screen_height"] == 500
        assert config["agent"]["prompt_builder_kwargs"]["n_shot"] == 10
        assert config["agent"]["prompt_builder_kwargs"]["split"] == "visual"
        assert config["agent"]["prompt_builder_kwargs"]["use_feedback"] is True


class TestRegistryExtendsIntegration:
    """Test that the registry resolves extends during discovery."""

    def test_registry_discovers_split_with_extends(self, tmp_path):
        from easi.tasks.registry import _discover_tasks
        task_dir = tmp_path / "my_bench"
        task_dir.mkdir()
        (task_dir / "_base.yaml").write_text(
            "simulator: dummy:v1\n"
            "task_class: easi.tasks.dummy_task.task.DummyTask\n"
            "max_steps: 50\n"
        )
        (task_dir / "my_bench_base.yaml").write_text(
            "extends: _base.yaml\n"
            "name: my_bench_base\n"
            "display_name: My Bench Base\n"
            "description: Base split\n"
        )
        entries = _discover_tasks(tasks_dir=tmp_path)
        assert "my_bench_base" in entries
        assert entries["my_bench_base"].simulator_key == "dummy:v1"
        assert entries["my_bench_base"].max_steps == 50

    def test_registry_skips_base_yaml_without_name(self, tmp_path):
        from easi.tasks.registry import _discover_tasks
        task_dir = tmp_path / "my_bench"
        task_dir.mkdir()
        (task_dir / "_base.yaml").write_text(
            "simulator: dummy:v1\nmax_steps: 50\n"
        )
        entries = _discover_tasks(tasks_dir=tmp_path)
        assert len(entries) == 0
