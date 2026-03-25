"""Unit tests for easi.llm.models.http_server."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

from easi.llm.models.base_model_server import BaseModelServer
from easi.llm.models.http_server import create_app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class EchoModel(BaseModelServer):
    """Test model that echoes back the last user message."""

    def load(self, model_path: str, device: str, **kwargs) -> None:
        self._loaded = True

    def generate(self, messages: list[dict], **kwargs) -> str:
        # Return last user message content, or empty string.
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""


class ErrorModel(BaseModelServer):
    """Test model that always raises an error."""

    def load(self, model_path: str, device: str, **kwargs) -> None:
        pass

    def generate(self, messages: list[dict], **kwargs) -> str:
        raise RuntimeError("generation failed")


def _make_echo_client() -> TestClient:
    model = EchoModel()
    model.load("test", "cpu")
    app = create_app(model)
    return TestClient(app)


# ---------------------------------------------------------------------------
# TestCreateAppUnit
# ---------------------------------------------------------------------------

class TestCreateAppUnit:
    """Verify create_app returns a FastAPI app with the expected routes."""

    def test_returns_fastapi_app(self):
        from fastapi import FastAPI

        model = EchoModel()
        model.load("test", "cpu")
        app = create_app(model)
        assert isinstance(app, FastAPI)

    def test_app_has_health_route(self):
        model = EchoModel()
        model.load("test", "cpu")
        app = create_app(model)
        paths = [route.path for route in app.routes]
        assert "/health" in paths

    def test_app_has_chat_completions_route(self):
        model = EchoModel()
        model.load("test", "cpu")
        app = create_app(model)
        paths = [route.path for route in app.routes]
        assert "/v1/chat/completions" in paths


# ---------------------------------------------------------------------------
# TestEndpoints
# ---------------------------------------------------------------------------

class TestEndpoints:
    """Test endpoints via FastAPI TestClient (no real HTTP server)."""

    def test_health(self):
        client = _make_echo_client()
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_chat_completions_echo(self):
        client = _make_echo_client()
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "echo",
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello!"},
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["object"] == "chat.completion"
        assert data["model"] == "echo"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"] == "Hello!"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert "usage" in data
        assert data["id"].startswith("chatcmpl-")

    def test_chat_completions_passes_gen_kwargs(self):
        """Verify generation kwargs from the request body are forwarded."""
        model = MagicMock(spec=BaseModelServer)
        model.generate.return_value = "ok"
        app = create_app(model)
        client = TestClient(app)

        client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.5,
                "max_tokens": 100,
                "top_p": 0.9,
                "top_k": 50,
                "seed": 42,
            },
        )

        model.generate.assert_called_once()
        _, kwargs = model.generate.call_args
        assert kwargs["temperature"] == 0.5
        assert kwargs["max_tokens"] == 100
        assert kwargs["top_p"] == 0.9
        assert kwargs["top_k"] == 50
        assert kwargs["seed"] == 42

    def test_chat_completions_ignores_unknown_kwargs(self):
        """Unknown keys in the request body are not passed to generate."""
        model = MagicMock(spec=BaseModelServer)
        model.generate.return_value = "ok"
        app = create_app(model)
        client = TestClient(app)

        client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "unknown_param": True,
            },
        )

        model.generate.assert_called_once()
        _, kwargs = model.generate.call_args
        assert "unknown_param" not in kwargs

    def test_chat_completions_error_returns_500(self):
        model = ErrorModel()
        model.load("test", "cpu")
        app = create_app(model)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 500
        data = resp.json()
        assert "error" in data
        assert "generation failed" in data["error"]["message"]

    def test_chat_completions_empty_messages(self):
        client = _make_echo_client()
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "echo", "messages": []},
        )
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["content"] == ""


# ---------------------------------------------------------------------------
# TestHTTPServerMain
# ---------------------------------------------------------------------------

class TestHTTPServerMain:
    """Verify main() argument parsing with everything mocked."""

    @patch("uvicorn.run")
    @patch("easi.llm.models.http_server.create_app")
    def test_main_parses_args(self, mock_create_app, mock_uvicorn_run):
        from easi.llm.models.http_server import main

        mock_cls = MagicMock()
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        mock_create_app.return_value = MagicMock()

        mock_entry = MagicMock()
        mock_entry.default_kwargs = {"default_key": "default_val"}

        with patch(
            "easi.llm.models.registry.load_model_class", return_value=mock_cls
        ), patch(
            "easi.llm.models.registry.get_model_entry", return_value=mock_entry
        ), patch(
            "sys.argv",
            [
                "http_server",
                "--model-name",
                "test_model",
                "--model-path",
                "/tmp/weights",
                "--device",
                "cpu",
                "--port",
                "9000",
                "--kwargs",
                '{"key": "val"}',
            ],
        ):
            main()

        mock_cls.assert_called_once()
        # CLI kwargs override manifest defaults
        mock_instance.load.assert_called_once_with(
            "/tmp/weights", "cpu", default_key="default_val", key="val"
        )
        mock_create_app.assert_called_once_with(mock_instance)
        mock_uvicorn_run.assert_called_once()
        call_kwargs = mock_uvicorn_run.call_args
        assert call_kwargs[1]["port"] == 9000

    @patch("uvicorn.run")
    @patch("easi.llm.models.http_server.create_app")
    def test_main_defaults(self, mock_create_app, mock_uvicorn_run):
        from easi.llm.models.http_server import main

        mock_cls = MagicMock()
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        mock_create_app.return_value = MagicMock()

        mock_entry = MagicMock()
        mock_entry.default_kwargs = {}

        with patch(
            "easi.llm.models.registry.load_model_class", return_value=mock_cls
        ), patch(
            "easi.llm.models.registry.get_model_entry", return_value=mock_entry
        ), patch(
            "sys.argv",
            [
                "http_server",
                "--model-name",
                "m",
                "--model-path",
                "/p",
            ],
        ):
            main()

        # Default device is cuda:0, default port is 8000
        mock_instance.load.assert_called_once_with("/p", "cuda:0")
        call_kwargs = mock_uvicorn_run.call_args
        assert call_kwargs[1]["port"] == 8000
