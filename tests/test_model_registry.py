"""Tests for model registry with YAML manifest auto-discovery."""

from __future__ import annotations

import pytest

from easi.llm.models.registry import (
    ModelEntry,
    get_model_entry,
    list_models,
    load_model_class,
    refresh,
)


# ---------------------------------------------------------------------------
# Fake model class used by load_model_class test
# ---------------------------------------------------------------------------

class FakeModel:
    """Stub model class for testing dynamic import."""
    pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_registry():
    """Ensure a clean registry before and after every test."""
    refresh()
    yield
    refresh()


@pytest.fixture()
def fake_model_dir(tmp_path):
    """Create a temporary model directory with a valid manifest.yaml."""
    model_dir = tmp_path / "test_model"
    model_dir.mkdir()
    manifest = model_dir / "manifest.yaml"
    manifest.write_text(
        "name: test_model\n"
        'display_name: "Test Model"\n'
        'description: "A model for testing"\n'
        'model_class: "tests.test_model_registry.FakeModel"\n'
        "default_kwargs:\n"
        '  torch_dtype: "float32"\n'
    )
    return tmp_path


# ---------------------------------------------------------------------------
# TestModelEntry
# ---------------------------------------------------------------------------

class TestModelEntry:
    """Basic dataclass construction tests."""

    def test_create_with_defaults(self):
        entry = ModelEntry(
            name="m1",
            display_name="Model One",
            description="desc",
            model_class="some.module.Class",
        )
        assert entry.name == "m1"
        assert entry.default_kwargs == {}

    def test_create_with_kwargs(self):
        entry = ModelEntry(
            name="m2",
            display_name="Model Two",
            description="desc",
            model_class="some.module.Class",
            default_kwargs={"tp": 2},
        )
        assert entry.default_kwargs == {"tp": 2}


# ---------------------------------------------------------------------------
# TestDiscovery
# ---------------------------------------------------------------------------

class TestDiscovery:
    """Discovery and lookup tests using a temporary model directory."""

    def test_list_models_discovers_from_manifest(self, fake_model_dir, monkeypatch):
        monkeypatch.setattr(
            "easi.llm.models.registry._get_models_dir", lambda: fake_model_dir
        )
        names = list_models()
        assert "test_model" in names

    def test_get_model_entry(self, fake_model_dir, monkeypatch):
        monkeypatch.setattr(
            "easi.llm.models.registry._get_models_dir", lambda: fake_model_dir
        )
        entry = get_model_entry("test_model")
        assert entry.display_name == "Test Model"
        assert entry.description == "A model for testing"
        assert entry.model_class == "tests.test_model_registry.FakeModel"
        assert entry.default_kwargs == {"torch_dtype": "float32"}

    def test_get_model_entry_not_found(self, fake_model_dir, monkeypatch):
        monkeypatch.setattr(
            "easi.llm.models.registry._get_models_dir", lambda: fake_model_dir
        )
        with pytest.raises(KeyError, match="no_such_model"):
            get_model_entry("no_such_model")

    def test_load_model_class(self, fake_model_dir, monkeypatch):
        monkeypatch.setattr(
            "easi.llm.models.registry._get_models_dir", lambda: fake_model_dir
        )
        cls = load_model_class("test_model")
        assert cls is FakeModel


# ---------------------------------------------------------------------------
# TestBuiltInEchoModel
# ---------------------------------------------------------------------------

class TestBuiltInEchoModel:
    def test_echo_model_discovered(self):
        refresh()
        assert "echo" in list_models()

    def test_echo_model_loadable(self):
        refresh()
        cls = load_model_class("echo")
        instance = cls()
        instance.load("dummy", "cpu")
        result = instance.generate([{"role": "user", "content": "hello"}])
        assert "hello" in result.lower()
