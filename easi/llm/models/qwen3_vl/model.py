"""Qwen3-VL custom model server for EASI.

Loads Qwen3-VL models (8B, 72B, etc.) via HuggingFace Transformers and
serves them through the custom model server pipeline.

Requires:
    pip install transformers torch torchvision pillow

Usage:
    easi start <task> --backend custom --model qwen3_vl \
        --llm-kwargs '{"model_path": "Qwen/Qwen3-VL-8B-Instruct"}'
"""
from __future__ import annotations

from easi.llm.models.base_model_server import BaseModelServer
from easi.llm.models.helpers import extract_images
from easi.utils.logging import get_logger

logger = get_logger(__name__)

# Recognised dtype strings for from_pretrained
_ALLOWED_DTYPES = {"bfloat16", "float16", "float32", "auto"}


def _openai_to_qwen_messages(messages: list[dict], images: list) -> list[dict]:
    """Convert OpenAI-format messages to Qwen3-VL format.

    OpenAI format:
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    Qwen format:
        {"type": "image", "image": <PIL.Image>}
    """
    image_idx = 0
    converted = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            converted.append({"role": role, "content": content})
            continue

        new_content = []
        for part in content:
            ptype = part.get("type", "")
            if ptype == "image_url" and image_idx < len(images):
                new_content.append({"type": "image", "image": images[image_idx]})
                image_idx += 1
            elif ptype == "text":
                new_content.append({"type": "text", "text": part.get("text", "")})

        converted.append({"role": role, "content": new_content})

    return converted


class Qwen3VLModel(BaseModelServer):
    """Qwen3-VL vision-language model server."""

    def load(self, model_path: str, device: str, **kwargs) -> None:
        """Load Qwen3-VL model and processor.

        Args:
            model_path: HuggingFace model ID (e.g., "Qwen/Qwen3-VL-8B-Instruct")
                        or local path to model weights.
            device: Device string (e.g., "cuda:0"). When using device_map="auto",
                    this is used as fallback.
            **kwargs: Extra kwargs passed to from_pretrained.
                      Supported: torch_dtype, attn_implementation.
        """
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        # Resolve torch dtype — newer transformers uses "dtype" instead of "torch_dtype"
        dtype_str = kwargs.pop("torch_dtype", None) or kwargs.pop("dtype", "auto")
        if dtype_str not in _ALLOWED_DTYPES:
            logger.warning("Unrecognised dtype '%s', falling back to 'auto'", dtype_str)
            dtype_str = "auto"
        torch_dtype = getattr(torch, dtype_str, "auto") if dtype_str != "auto" else "auto"

        attn_impl = kwargs.pop("attn_implementation", None)

        # Use device_map="auto" only if accelerate is available; otherwise
        # fall back to loading on the specified device directly.
        try:
            import accelerate  # noqa: F401
            load_kwargs = {"dtype": torch_dtype, "device_map": "auto"}
        except ImportError:
            logger.info("accelerate not installed, loading model on %s without device_map", device)
            load_kwargs = {"dtype": torch_dtype}

        if attn_impl:
            # Validate flash_attention_2 availability before requesting it
            if attn_impl == "flash_attention_2":
                try:
                    import flash_attn  # noqa: F401
                    load_kwargs["attn_implementation"] = attn_impl
                except ImportError:
                    logger.warning(
                        "flash_attn not installed, falling back to sdpa attention. "
                        "Install with: pip install flash-attn --no-build-isolation"
                    )
                    load_kwargs["attn_implementation"] = "sdpa"
            else:
                load_kwargs["attn_implementation"] = attn_impl

        logger.info("Loading Qwen3-VL from %s (dtype=%s, attn=%s)", model_path, dtype_str, attn_impl)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, **load_kwargs
        )
        # Move to device if device_map was not used
        if "device_map" not in load_kwargs:
            self.model = self.model.to(device)

        self.processor = AutoProcessor.from_pretrained(model_path)
        # device_map="auto" shards across GPUs; .device would raise RuntimeError.
        # Use the device of the first parameter instead.
        self.device = next(self.model.parameters()).device
        logger.info("Qwen3-VL loaded on %s", self.device)

    def generate(self, messages: list[dict], **kwargs) -> str:
        """Generate response from OpenAI-format messages.

        Converts OpenAI message format to Qwen3-VL format, processes
        images via the Qwen processor, and runs generation.
        """
        import torch

        # Extract images from OpenAI-format base64 entries
        images = extract_images(messages)

        # Convert message format
        qwen_messages = _openai_to_qwen_messages(messages, images)

        # Process with Qwen processor (handles tokenization + image processing)
        inputs = self.processor.apply_chat_template(
            qwen_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        # Generation kwargs
        max_new_tokens = kwargs.get("max_tokens", 4096)
        temperature = kwargs.get("temperature", 0.0)
        top_p = kwargs.get("top_p", 0.95)

        gen_kwargs = {"max_new_tokens": max_new_tokens}
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            gen_kwargs["do_sample"] = True
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        # Trim input tokens from output
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text[0] if output_text else ""

    def unload(self) -> None:
        """Release GPU memory."""
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor

        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Qwen3-VL model unloaded")
