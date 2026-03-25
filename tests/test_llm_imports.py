"""Tests for easi.llm public API imports."""


class TestLlmImports:
    def test_import_llm_client(self):
        from easi.llm.client import LLMClient
        assert LLMClient is not None

    def test_import_server_manager(self):
        from easi.llm.server_manager import ServerManager
        assert ServerManager is not None

    def test_import_utils(self):
        from easi.llm.utils import parse_llm_kwargs, split_kwargs, build_litellm_model, validate_backend
        assert parse_llm_kwargs is not None

    def test_legacy_api_client_still_importable(self):
        from easi.llm.api_client import LLMApiClient
        assert LLMApiClient is not None

    def test_legacy_dummy_server_still_importable(self):
        from easi.llm.dummy_server import run_server
        assert run_server is not None
