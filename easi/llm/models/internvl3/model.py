"""InternVL3 custom model server for EASI.

Loads InternVL3 models via transformers and serves them through the
EASI custom backend HTTP server.  Uses the model's built-in ``.chat()``
method which handles chat template formatting and image preprocessing
internally.

Usage::

    easi start <task> --backend custom --model internvl3 \\
        --llm-kwargs '{"model_path": "OpenGVLab/InternVL3-8B"}'
"""
from __future__ import annotations

from typing import Any

import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from easi.llm.models.base_model_server import BaseModelServer
from easi.llm.models.helpers import extract_images
from easi.utils.logging import get_logger

logger = get_logger(__name__)

_ALLOWED_DTYPES = {"bfloat16", "float16", "float32", "auto"}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Image preprocessing (InternVL3 dynamic resolution tiling)
# ---------------------------------------------------------------------------

def _build_transform(input_size: int) -> T.Compose:
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best = (1, 1)
    best_diff = float("inf")
    area = width * height
    for ratio in target_ratios:
        target_ar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_ar)
        if diff < best_diff or (diff == best_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]):
            best_diff = diff
            best = ratio
    return best


def _dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> list[Image.Image]:
    """Split image into tiles using InternVL3's dynamic resolution strategy."""
    width, height = image.size
    aspect_ratio = width / height

    target_ratios = set()
    for n in range(min_num, max_num + 1):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i * j <= max_num and i * j >= min_num:
                    target_ratios.add((i, j))
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    best = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, width, height, image_size
    )

    target_w = best[0] * image_size
    target_h = best[1] * image_size
    blocks = best[0] * best[1]

    resized = image.resize((target_w, target_h))
    processed = []
    for i in range(blocks):
        box = (
            (i % best[0]) * image_size,
            (i // best[0]) * image_size,
            ((i % best[0]) + 1) * image_size,
            ((i // best[0]) + 1) * image_size,
        )
        processed.append(resized.crop(box))

    if use_thumbnail and blocks > 1:
        thumbnail = image.resize((image_size, image_size))
        processed.append(thumbnail)

    return processed


def _load_image(image: Image.Image, max_num: int = 12) -> Any:
    """Preprocess a PIL image into a pixel_values tensor for InternVL3.

    Returns float32 tensors; the caller should cast to the model's dtype
    via ``.to(dtype=model.dtype, device=model.device)``.
    """
    import torch

    transform = _build_transform(448)
    tiles = _dynamic_preprocess(image, image_size=448, max_num=max_num)
    pixel_values = torch.stack([transform(tile) for tile in tiles])
    return pixel_values


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------

def _openai_to_internvl_messages(
    messages: list[dict],
) -> list[dict]:
    """Convert OpenAI-format messages to InternVL3 format.

    OpenAI format uses ``image_url`` content parts with base64 data URIs.
    InternVL3 expects ``<image>`` placeholder tokens in the text content,
    with actual pixel tensors passed separately.

    Returns a new message list with image_url parts replaced by ``<image>\\n``
    text tokens.  The caller is responsible for extracting and preprocessing
    the actual PIL images via ``extract_images()`` + ``_load_image()``.
    """
    converted = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            converted.append({"role": role, "content": content})
            continue

        # Multimodal content list
        text_parts: list[str] = []
        for part in content:
            ptype = part.get("type", "")
            if ptype == "image_url":
                text_parts.append("<image>\n")
            elif ptype == "text":
                text_parts.append(part.get("text", ""))

        converted.append({"role": role, "content": "".join(text_parts)})

    return converted


# ---------------------------------------------------------------------------
# Model server
# ---------------------------------------------------------------------------

class InternVL3Model(BaseModelServer):
    """InternVL3 vision-language model server."""

    def load(self, model_path: str, device: str, **kwargs: Any) -> None:
        """Load InternVL3 model and tokenizer.

        Args:
            model_path: HuggingFace model ID or local path.
            device: Device string (e.g. ``"cuda:0"``).
            **kwargs: ``torch_dtype``, ``attn_implementation``.
        """
        import torch
        from transformers import AutoModel, AutoTokenizer

        # Resolve dtype
        dtype_str = kwargs.pop("torch_dtype", None) or kwargs.pop("dtype", "auto")
        if dtype_str not in _ALLOWED_DTYPES:
            logger.warning("Unrecognised dtype '%s', falling back to 'auto'", dtype_str)
            dtype_str = "auto"
        torch_dtype = getattr(torch, dtype_str, "auto") if dtype_str != "auto" else "auto"

        attn_impl = kwargs.pop("attn_implementation", None)

        # Device mapping
        try:
            import accelerate  # noqa: F401
            load_kwargs: dict[str, Any] = {"torch_dtype": torch_dtype, "device_map": "auto"}
        except ImportError:
            logger.info("accelerate not installed, loading on %s", device)
            load_kwargs = {"torch_dtype": torch_dtype}

        # Attention implementation with fallback
        if attn_impl:
            if attn_impl == "flash_attention_2":
                try:
                    import flash_attn  # noqa: F401
                    load_kwargs["attn_implementation"] = attn_impl
                except ImportError:
                    logger.warning("flash_attn not installed, falling back to sdpa")
                    load_kwargs["attn_implementation"] = "sdpa"
            else:
                load_kwargs["attn_implementation"] = attn_impl

        load_kwargs["trust_remote_code"] = True

        logger.info(
            "Loading InternVL3 from %s (dtype=%s, attn=%s)",
            model_path, dtype_str, attn_impl,
        )
        self.model = AutoModel.from_pretrained(model_path, **load_kwargs).eval()

        if "device_map" not in load_kwargs:
            self.model = self.model.to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.device = next(self.model.parameters()).device
        logger.info("InternVL3 loaded on %s", self.device)

    def generate(self, messages: list[dict], **kwargs: Any) -> str:
        """Generate response from OpenAI-format messages."""
        import torch

        # Extract and preprocess images, cast to model dtype
        pil_images = extract_images(messages)
        pixel_values = None
        if pil_images:
            model_dtype = next(self.model.parameters()).dtype
            tensors = [
                _load_image(img).to(dtype=model_dtype, device=self.device)
                for img in pil_images
            ]
            pixel_values = torch.cat(tensors, dim=0)

        # Convert messages
        internvl_messages = _openai_to_internvl_messages(messages)

        # Extract system message (prepend to first user question if present)
        system_prefix = ""
        start = 0
        if internvl_messages and internvl_messages[0]["role"] == "system":
            system_prefix = internvl_messages[0]["content"] + "\n"
            start = 1

        # Build question from last user message (model.chat expects this)
        question = internvl_messages[-1]["content"] if internvl_messages else ""

        # Build history from prior messages (pairs of user/assistant)
        history: list[tuple[str, str]] = []
        i = start
        while i < len(internvl_messages) - 1:
            if (
                internvl_messages[i]["role"] == "user"
                and i + 1 < len(internvl_messages)
                and internvl_messages[i + 1]["role"] == "assistant"
            ):
                user_content = internvl_messages[i]["content"]
                # Prepend system message to the first user turn
                if i == start and system_prefix:
                    user_content = system_prefix + user_content
                history.append((
                    user_content,
                    internvl_messages[i + 1]["content"],
                ))
                i += 2
            else:
                i += 1

        # If no history, prepend system to the question directly
        if not history and system_prefix:
            question = system_prefix + question

        # Generation config
        max_new_tokens = kwargs.get("max_tokens", 4096)
        temperature = kwargs.get("temperature", 0.0)
        top_p = kwargs.get("top_p", 0.95)

        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generation_config["temperature"] = temperature
            generation_config["top_p"] = top_p

        with torch.no_grad():
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                generation_config,
                history=history,
            )

        return response

    def unload(self) -> None:
        """Release GPU memory."""
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer

        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("InternVL3 model unloaded")
