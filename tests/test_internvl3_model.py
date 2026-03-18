"""Tests for InternVL3 custom model server."""
from __future__ import annotations

import base64
import io
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from easi.llm.models.base_model_server import BaseModelServer


# Minimal 1x1 red PNG for testing
def _make_test_image_b64() -> str:
    img = Image.new("RGB", (1, 1), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


_PIXEL_B64 = _make_test_image_b64()


class TestInternVL3ModelSubclass:
    def test_is_base_model_server_subclass(self):
        from easi.llm.models.internvl3.model import InternVL3Model
        assert issubclass(InternVL3Model, BaseModelServer)

    def test_can_instantiate(self):
        from easi.llm.models.internvl3.model import InternVL3Model
        model = InternVL3Model()
        assert model is not None


class TestInternVL3MessageConversion:
    """Test _openai_to_internvl_messages helper."""

    def test_text_only_message(self):
        from easi.llm.models.internvl3.model import _openai_to_internvl_messages
        msgs = [{"role": "user", "content": "hello"}]
        result = _openai_to_internvl_messages(msgs)
        assert result == [{"role": "user", "content": "hello"}]

    def test_single_image_message(self):
        from easi.llm.models.internvl3.model import _openai_to_internvl_messages
        msgs = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_PIXEL_B64}"}},
                {"type": "text", "text": "describe this"},
            ],
        }]
        result = _openai_to_internvl_messages(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        # Should have <image> placeholder followed by text
        assert "<image>" in result[0]["content"]
        assert "describe this" in result[0]["content"]

    def test_multiple_images(self):
        from easi.llm.models.internvl3.model import _openai_to_internvl_messages
        msgs = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_PIXEL_B64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_PIXEL_B64}"}},
                {"type": "text", "text": "compare"},
            ],
        }]
        result = _openai_to_internvl_messages(msgs)
        content = result[0]["content"]
        assert content.count("<image>") == 2

    def test_system_message_preserved(self):
        from easi.llm.models.internvl3.model import _openai_to_internvl_messages
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ]
        result = _openai_to_internvl_messages(msgs)
        assert result[0] == {"role": "system", "content": "You are helpful."}
        assert result[1] == {"role": "user", "content": "hi"}


class TestInternVL3Load:
    """Test load() with mocked transformers."""

    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_load_basic(self, mock_tokenizer_cls, mock_model_cls):
        from easi.llm.models.internvl3.model import InternVL3Model

        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = "cpu"
        mock_model.parameters.return_value = iter([mock_param])
        mock_model_cls.from_pretrained.return_value = mock_model

        model = InternVL3Model()
        model.load("OpenGVLab/InternVL3-8B", "cpu", torch_dtype="float32")

        mock_model_cls.from_pretrained.assert_called_once()
        call_kwargs = mock_model_cls.from_pretrained.call_args
        assert call_kwargs[0][0] == "OpenGVLab/InternVL3-8B"
        assert call_kwargs[1]["trust_remote_code"] is True

    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_load_stores_tokenizer(self, mock_tokenizer_cls, mock_model_cls):
        from easi.llm.models.internvl3.model import InternVL3Model

        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = "cpu"
        mock_model.parameters.return_value = iter([mock_param])
        mock_model_cls.from_pretrained.return_value = mock_model

        model = InternVL3Model()
        model.load("OpenGVLab/InternVL3-8B", "cpu")

        mock_tokenizer_cls.from_pretrained.assert_called_once()
        assert model.tokenizer is not None


class TestInternVL3Generate:
    """Test generate() with mocked model."""

    def _make_loaded_model(self):
        """Create an InternVL3Model with mocked internals."""
        import torch
        from easi.llm.models.internvl3.model import InternVL3Model

        model = InternVL3Model()
        model.model = MagicMock()
        model.tokenizer = MagicMock()
        model.device = "cpu"

        # Mock model.parameters() to return a param with real dtype
        mock_param = MagicMock()
        mock_param.dtype = torch.float32
        model.model.parameters.return_value = iter([mock_param])

        # Mock model.chat() to return a string
        model.model.chat.return_value = "This is a red pixel."
        return model

    def test_generate_text_only(self):
        model = self._make_loaded_model()
        result = model.generate([
            {"role": "user", "content": "hello"},
        ])
        assert isinstance(result, str)
        assert result == "This is a red pixel."

    def test_generate_with_image(self):
        model = self._make_loaded_model()
        result = model.generate([{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_PIXEL_B64}"}},
                {"type": "text", "text": "describe"},
            ],
        }])
        assert isinstance(result, str)
        # model.chat should have been called with pixel_values
        model.model.chat.assert_called_once()

    def test_generate_passes_max_tokens(self):
        model = self._make_loaded_model()
        model.generate(
            [{"role": "user", "content": "hi"}],
            max_tokens=512,
        )
        call_args = model.model.chat.call_args
        gen_config = call_args[0][3]  # 4th positional arg
        assert gen_config["max_new_tokens"] == 512

    def test_generate_prepends_system_message(self):
        model = self._make_loaded_model()
        model.generate([
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "hello"},
        ])
        call_args = model.model.chat.call_args
        question = call_args[0][2]  # 3rd positional arg
        assert "Be concise." in question


class TestInternVL3Unload:
    def test_unload_cleans_up(self):
        from easi.llm.models.internvl3.model import InternVL3Model
        model = InternVL3Model()
        model.model = MagicMock()
        model.tokenizer = MagicMock()
        model.unload()
        assert not hasattr(model, "model") or model.model is None


class TestInternVL3Registry:
    @pytest.fixture(autouse=True)
    def _clear_registry(self):
        from easi.llm.models.registry import refresh
        refresh()
        yield
        refresh()

    def test_manifest_discovered(self):
        from easi.llm.models.registry import list_models
        assert "internvl3" in list_models()

    def test_model_entry_has_correct_class(self):
        from easi.llm.models.registry import get_model_entry
        entry = get_model_entry("internvl3")
        assert entry.model_class == "easi.llm.models.internvl3.model.InternVL3Model"
