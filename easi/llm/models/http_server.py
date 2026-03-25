"""FastAPI HTTP server wrapping a BaseModelServer in OpenAI-compatible endpoints.

Provides ``create_app`` to build a FastAPI application and ``main`` for
subprocess launch via ``python -m easi.llm.models.http_server``.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from functools import partial
from typing import Any

from easi.utils.logging import get_logger

logger = get_logger(__name__)

# Generation kwargs recognised from the request body.
_GENERATION_KWARGS = frozenset(
    {
        "temperature",
        "max_tokens",
        "top_p",
        "top_k",
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "seed",
    }
)


def create_app(model: Any) -> Any:
    """Create a FastAPI application that serves *model* over HTTP.

    Parameters
    ----------
    model:
        A loaded :class:`BaseModelServer` instance.

    Returns
    -------
    FastAPI
        The application, ready to be passed to ``uvicorn.run``.
    """
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    app = FastAPI(title="EASI Model Server")

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: dict) -> JSONResponse:  # type: ignore[arg-type]
        messages = request.get("messages", [])
        req_model = request.get("model", "custom")

        # Extract recognised generation kwargs.
        gen_kwargs: dict[str, Any] = {}
        for key in _GENERATION_KWARGS:
            if key in request:
                gen_kwargs[key] = request[key]

        try:
            loop = asyncio.get_running_loop()
            content = await loop.run_in_executor(
                None, partial(model.generate, messages, **gen_kwargs)
            )
        except Exception as e:
            logger.error("Generation failed: %s", e, exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"error": {"message": str(e), "type": "server_error"}},
            )

        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
        return JSONResponse(content=response)

    return app


def main() -> None:
    """Entry point for subprocess launch.

    Usage::

        python -m easi.llm.models.http_server \
            --model-name my_model \
            --model-path /path/to/weights \
            --device cuda:0 \
            --port 8000 \
            --kwargs '{"key": "value"}'
    """
    import argparse

    import uvicorn

    from easi.llm.models.registry import get_model_entry, load_model_class

    parser = argparse.ArgumentParser(description="EASI custom model HTTP server")
    parser.add_argument("--model-name", required=True, help="Registered model name")
    parser.add_argument("--model-path", required=True, help="Path to model weights")
    parser.add_argument("--device", default="cuda:0", help="Device (default: cuda:0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument(
        "--kwargs",
        default="{}",
        help="Extra kwargs as JSON string (default: '{}')",
    )
    args = parser.parse_args()

    extra_kwargs: dict[str, Any] = json.loads(args.kwargs)

    # Merge manifest default_kwargs with CLI overrides (CLI wins)
    entry = get_model_entry(args.model_name)
    merged_kwargs = {**entry.default_kwargs, **extra_kwargs}

    logger.info(
        "Loading model '%s' from %s on %s (kwargs=%s)",
        args.model_name,
        args.model_path,
        args.device,
        merged_kwargs,
    )

    cls = load_model_class(args.model_name)
    model_instance = cls()
    model_instance.load(args.model_path, args.device, **merged_kwargs)

    app = create_app(model_instance)

    logger.info("Starting HTTP server on port %d", args.port)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
