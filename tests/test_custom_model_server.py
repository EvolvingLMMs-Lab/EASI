from __future__ import annotations

import base64

import pytest

from easi.llm.models.base_model_server import BaseModelServer
from easi.llm.models.helpers import extract_by_role, extract_images, extract_text_only

# Minimal valid 1x1 RGB PNG
_PIXEL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
    b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)
_PIXEL_B64 = base64.b64encode(_PIXEL_PNG).decode()


# ---------------------------------------------------------------------------
# TestBaseModelServer
# ---------------------------------------------------------------------------

class TestBaseModelServer:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseModelServer()  # type: ignore[abstract]

    def test_must_implement_load_and_generate(self):
        class OnlyLoad(BaseModelServer):
            def load(self, model_path, device, **kwargs):
                pass

        with pytest.raises(TypeError):
            OnlyLoad()  # type: ignore[abstract]

        class OnlyGenerate(BaseModelServer):
            def generate(self, messages, **kwargs):
                return ""

        with pytest.raises(TypeError):
            OnlyGenerate()  # type: ignore[abstract]

    def test_valid_subclass(self):
        class MyServer(BaseModelServer):
            def load(self, model_path, device, **kwargs):
                pass

            def generate(self, messages, **kwargs):
                return "hello"

        server = MyServer()
        server.load("path", "cpu")
        assert server.generate([]) == "hello"

    def test_default_unload_is_noop(self):
        class MyServer(BaseModelServer):
            def load(self, model_path, device, **kwargs):
                pass

            def generate(self, messages, **kwargs):
                return ""

        server = MyServer()
        assert server.unload() is None


# ---------------------------------------------------------------------------
# TestExtractImages
# ---------------------------------------------------------------------------

class TestExtractImages:
    def test_empty_messages(self):
        assert extract_images([]) == []

    def test_text_only_messages(self):
        msgs = [{"role": "user", "content": "hello"}]
        assert extract_images(msgs) == []

    def test_extracts_base64_image(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{_PIXEL_B64}",
                        },
                    },
                ],
            }
        ]
        images = extract_images(msgs)
        assert len(images) == 1
        assert images[0].size == (1, 1)

    def test_multiple_images(self):
        part = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{_PIXEL_B64}"},
        }
        msgs = [
            {"role": "user", "content": [part, part]},
            {"role": "user", "content": [part]},
        ]
        assert len(extract_images(msgs)) == 3


# ---------------------------------------------------------------------------
# TestExtractTextOnly
# ---------------------------------------------------------------------------

class TestExtractTextOnly:
    def test_empty_messages(self):
        assert extract_text_only([]) == ""

    def test_string_content(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        assert extract_text_only(msgs) == "You are helpful.\nHi"

    def test_list_content_with_images_filtered(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look at this"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{_PIXEL_B64}"},
                    },
                    {"type": "text", "text": "What is it?"},
                ],
            }
        ]
        result = extract_text_only(msgs)
        assert "Look at this" in result
        assert "What is it?" in result
        assert "base64" not in result


# ---------------------------------------------------------------------------
# TestExtractByRole
# ---------------------------------------------------------------------------

class TestExtractByRole:
    def test_basic_role_grouping(self):
        msgs = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "Thanks"},
        ]
        result = extract_by_role(msgs)
        assert result["system"] == "Be helpful."
        assert result["user"] == "Hello\nThanks"
        assert result["assistant"] == "Hi there"

    def test_list_content(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{_PIXEL_B64}"},
                    },
                ],
            }
        ]
        result = extract_by_role(msgs)
        assert result["user"] == "Describe this"
