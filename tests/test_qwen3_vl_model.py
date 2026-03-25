"""Tests for Qwen3-VL custom model server.

Tests model discovery, message format conversion, and interface compliance
without requiring actual model weights or GPU.
"""
import base64

import pytest

from easi.llm.models.registry import get_model_entry, list_models, load_model_class, refresh
from easi.llm.models.qwen3_vl.model import _openai_to_qwen_messages


@pytest.fixture(autouse=True)
def _clear_registry():
    refresh()
    yield
    refresh()


# -- Registry discovery --

class TestQwen3VLDiscovery:
    def test_discovered_in_registry(self):
        assert "qwen3_vl" in list_models()

    def test_model_entry_fields(self):
        entry = get_model_entry("qwen3_vl")
        assert entry.display_name == "Qwen3-VL"
        assert entry.model_class == "easi.llm.models.qwen3_vl.model.Qwen3VLModel"
        assert entry.default_kwargs.get("dtype") == "bfloat16"

    def test_class_importable(self):
        cls = load_model_class("qwen3_vl")
        assert cls.__name__ == "Qwen3VLModel"

    def test_is_base_model_server_subclass(self):
        from easi.llm.models.base_model_server import BaseModelServer
        cls = load_model_class("qwen3_vl")
        assert issubclass(cls, BaseModelServer)


# -- Message format conversion --

class TestOpenAIToQwenMessages:
    """Test _openai_to_qwen_messages conversion."""

    def test_text_only_passthrough(self):
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        result = _openai_to_qwen_messages(messages, images=[])
        assert result[0] == {"role": "system", "content": "You are helpful"}
        assert result[1] == {"role": "user", "content": "Hello"}

    def test_image_url_converted_to_qwen_format(self):
        """OpenAI image_url entries should become Qwen image entries."""
        from PIL import Image
        fake_img = Image.new("RGB", (1, 1), color="red")

        messages = [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                {"type": "text", "text": "What is this?"},
            ]},
        ]
        result = _openai_to_qwen_messages(messages, images=[fake_img])

        user_content = result[0]["content"]
        assert len(user_content) == 2
        assert user_content[0]["type"] == "image"
        assert user_content[0]["image"] is fake_img
        assert user_content[1]["type"] == "text"
        assert user_content[1]["text"] == "What is this?"

    def test_multiple_images(self):
        from PIL import Image
        img1 = Image.new("RGB", (1, 1), color="red")
        img2 = Image.new("RGB", (1, 1), color="blue")

        messages = [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aaa"}},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,bbb"}},
                {"type": "text", "text": "Compare these"},
            ]},
        ]
        result = _openai_to_qwen_messages(messages, images=[img1, img2])

        user_content = result[0]["content"]
        assert user_content[0]["image"] is img1
        assert user_content[1]["image"] is img2

    def test_preserves_roles(self):
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": [
                {"type": "text", "text": "Hello"},
            ]},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = _openai_to_qwen_messages(messages, images=[])
        assert [m["role"] for m in result] == ["system", "user", "assistant"]

    def test_no_images_when_none_extracted(self):
        """image_url entries without matching images are skipped."""
        messages = [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                {"type": "text", "text": "Hello"},
            ]},
        ]
        result = _openai_to_qwen_messages(messages, images=[])
        user_content = result[0]["content"]
        # image_url entry should be skipped since no images provided
        assert len(user_content) == 1
        assert user_content[0]["type"] == "text"

    def test_mixed_text_and_images_across_messages(self):
        from PIL import Image
        img = Image.new("RGB", (1, 1))

        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                {"type": "text", "text": "Describe"},
            ]},
        ]
        result = _openai_to_qwen_messages(messages, images=[img])
        assert result[0]["content"] == "Be helpful"
        assert result[1]["content"][0]["type"] == "image"
