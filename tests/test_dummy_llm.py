"""Tests for the dummy LLM server and API client."""

import json
import threading
import time

import pytest
import requests

from easi.llm.api_client import LLMApiClient
from easi.llm.dummy_server import create_handler, run_server

# Use a non-standard port to avoid conflicts
TEST_PORT = 18765


@pytest.fixture(scope="module")
def llm_server():
    """Start a dummy LLM server in a background thread."""
    from http.server import HTTPServer

    handler = create_handler("random", ["MoveAhead", "TurnLeft", "TurnRight", "Stop"])
    server = HTTPServer(("127.0.0.1", TEST_PORT), handler)

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    # Wait for server to be ready
    for _ in range(20):
        try:
            requests.get(f"http://127.0.0.1:{TEST_PORT}/health", timeout=0.5)
            break
        except requests.ConnectionError:
            time.sleep(0.1)

    yield server
    server.shutdown()


class TestDummyLLMServer:
    def test_health(self, llm_server):
        resp = requests.get(f"http://127.0.0.1:{TEST_PORT}/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_chat_completions(self, llm_server):
        payload = {
            "model": "dummy",
            "messages": [
                {"role": "user", "content": "Choose an action."},
            ],
        }
        resp = requests.post(
            f"http://127.0.0.1:{TEST_PORT}/v1/chat/completions",
            json=payload,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "choices" in data
        assert len(data["choices"]) == 1
        content = data["choices"][0]["message"]["content"]
        assert "Action:" in content


class TestLLMApiClient:
    def test_generate(self, llm_server):
        client = LLMApiClient(base_url=f"http://127.0.0.1:{TEST_PORT}")
        response = client.generate(
            messages=[{"role": "user", "content": "Choose an action."}]
        )
        assert "Action:" in response

    def test_connection_error(self):
        client = LLMApiClient(base_url="http://127.0.0.1:19999")
        with pytest.raises(ConnectionError):
            client.generate(messages=[{"role": "user", "content": "test"}])
