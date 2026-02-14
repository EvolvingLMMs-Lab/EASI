"""Minimal dummy LLM server for testing.

Implements OpenAI-compatible /v1/chat/completions endpoint using stdlib
http.server. Returns fixed or random actions.

Usage:
    python -m easi.llm.dummy_server --port 8000 --mode random
    # or via CLI:
    easi llm-server --port 8000 --mode random
"""

from __future__ import annotations

import argparse
import json
import random
from http.server import BaseHTTPRequestHandler, HTTPServer

from easi.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_ACTION_SPACE = ["MoveAhead", "TurnLeft", "TurnRight", "Stop"]


class DummyLLMHandler(BaseHTTPRequestHandler):
    """HTTP handler for the dummy LLM server."""

    # Set by the server factory
    mode: str = "random"
    action_space: list[str] = DEFAULT_ACTION_SPACE

    def do_POST(self) -> None:
        if self.path == "/v1/chat/completions":
            self._handle_chat_completions()
        else:
            self.send_error(404, f"Not found: {self.path}")

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json({"status": "ok"})
        else:
            self.send_error(404, f"Not found: {self.path}")

    def _handle_chat_completions(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            request = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        # Generate response based on mode
        if self.mode == "fixed":
            action = self.action_space[0] if self.action_space else "MoveAhead"
        else:  # random
            action = random.choice(self.action_space)

        response_text = f"I will take the following action.\nAction: {action}"

        response = {
            "id": "dummy-001",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

        self._send_json(response)

    def _send_json(self, data: dict) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        logger.trace(format, *args)


def create_handler(mode: str, action_space: list[str]) -> type:
    """Create a handler class with the given configuration."""

    class ConfiguredHandler(DummyLLMHandler):
        pass

    ConfiguredHandler.mode = mode
    ConfiguredHandler.action_space = action_space
    return ConfiguredHandler


def run_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    mode: str = "random",
    action_space: list[str] | None = None,
) -> None:
    """Start the dummy LLM server."""
    action_space = action_space or DEFAULT_ACTION_SPACE
    handler_class = create_handler(mode, action_space)

    server = HTTPServer((host, port), handler_class)
    logger.info("Dummy LLM server running on http://%s:%d", host, port)
    logger.info("Mode: %s, Actions: %s", mode, action_space)
    logger.info("Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Dummy LLM server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mode", choices=["fixed", "random"], default="random")
    parser.add_argument(
        "--action-space",
        type=str,
        nargs="+",
        default=DEFAULT_ACTION_SPACE,
    )
    args = parser.parse_args()

    from easi.utils.logging import setup_logging
    setup_logging("INFO")
    run_server(
        host=args.host,
        port=args.port,
        mode=args.mode,
        action_space=args.action_space,
    )


if __name__ == "__main__":
    main()
