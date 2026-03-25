from __future__ import annotations

import base64
import io
from typing import Any

from easi.utils.logging import get_logger

logger = get_logger(__name__)


def _extract_text_from_content(content: str | list[dict[str, Any]]) -> str:
    """Extract concatenated text from an OpenAI-format content field.

    Handles both plain string content and the list-of-parts format.
    Image parts are silently skipped.
    """
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        if part.get("type") == "text":
            parts.append(part.get("text", ""))
    return "\n".join(parts)


def extract_images(messages: list[dict[str, Any]]) -> list[Any]:
    """Extract PIL Images from base64-encoded image_url parts in messages.

    Returns a list of ``PIL.Image.Image`` objects.  PIL is imported lazily
    so the function can be defined even when Pillow is not installed.
    """
    from PIL import Image  # lazy import

    images: list[Any] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") != "image_url":
                continue
            image_url = part.get("image_url", {})
            url = image_url.get("url", "")
            if url.startswith("data:") and ";base64," in url:
                # Format: data:<media_type>;base64,<data>
                _, encoded = url.split(",", 1)
                raw = base64.b64decode(encoded)
                images.append(Image.open(io.BytesIO(raw)))
            elif url.startswith("data:"):
                # Non-base64 data URI (e.g. data:text/plain,...) — skip
                logger.debug("Skipping non-base64 data URI")
            else:
                # HTTP/HTTPS URLs — not yet supported for extraction
                logger.debug("Skipping non-data image URL: %s", url[:80])
    return images


def extract_text_only(messages: list[dict[str, Any]]) -> str:
    """Concatenate all text content from messages, ignoring roles and images."""
    parts: list[str] = []
    for msg in messages:
        content = msg.get("content", "")
        text = _extract_text_from_content(content)
        if text:
            parts.append(text)
    return "\n".join(parts)


def extract_by_role(messages: list[dict[str, Any]]) -> dict[str, str]:
    """Group text content by role.

    Returns a mapping from role name to the concatenated text for that role.
    If a role appears multiple times its texts are joined with newlines.
    """
    grouped: dict[str, list[str]] = {}
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        text = _extract_text_from_content(content)
        if text:
            grouped.setdefault(role, []).append(text)
    return {role: "\n".join(texts) for role, texts in grouped.items()}
