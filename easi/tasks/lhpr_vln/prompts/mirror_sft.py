"""Mirror-experiment variant of the LHPR-VLN SFT prompt builder.

Horizontally flips every observation image and swaps the model's
``turn_left`` / ``turn_right`` outputs before they are executed. Used
to probe whether the SFT checkpoint has learned left/right-symmetric
navigation (vs. a dataset-biased shortcut).

Paired yamls: ``lhpr_vln_*_filtered_sft_mirror.yaml``.

Trajectory logging is unchanged: ``llm_response`` records the raw LLM
tokens (pre-remap), ``action`` records the executed (post-remap) name.
A mirror run is identifiable by ``<|left|>`` tokens co-occurring with
``turn_right`` executed actions.
"""
from __future__ import annotations

import base64
import io
import os
from pathlib import Path

from easi.agents.prompt_builder import _encode_image_base64 as _base_encode
from easi.core.episode import Action
from easi.core.memory import AgentMemory
from easi.tasks.lhpr_vln.prompts.sft import LHPRVLNSFTPromptBuilder
from easi.utils.logging import get_logger

logger = get_logger(__name__)

_ACTION_REMAP = {"turn_left": "turn_right", "turn_right": "turn_left"}

# Debug env: when set to a directory, the builder writes every flipped PNG it
# emits per call under <dir>/step_NNNN_<slot>.png (current views) and
# <dir>/step_NNNN_history_MMM.png (sampled historical front-views). Used to
# verify the mirror transformation matches the sim-rendered frames.
_DEBUG_DIR_ENV = "MIRROR_DEBUG_DIR"


def _encode_image_flipped(path: str) -> str | None:
    """Encode ``path`` as a horizontally-flipped base64 data URL.

    Delegates the NFS-safe read (with retries on truncation) to the
    parent helper, then decodes, flips, and re-encodes via PIL.
    """
    from PIL import Image

    data_url = _base_encode(path)
    if data_url is None:
        return None
    _, b64 = data_url.split(",", 1)
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw))
    fmt = (img.format or "PNG").upper()
    flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
    buf = io.BytesIO()
    flipped.save(buf, format=fmt)
    mime = "image/jpeg" if fmt in {"JPEG", "JPG"} else "image/png"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


def _resolve_debug_dir() -> Path | None:
    raw = os.environ.get(_DEBUG_DIR_ENV)
    if not raw:
        return None
    path = Path(raw).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _dump_data_url(data_url: str | None, dest: Path) -> None:
    if data_url is None:
        return
    try:
        _, b64 = data_url.split(",", 1)
        dest.write_bytes(base64.b64decode(b64))
    except Exception as e:  # noqa: BLE001 — debug-only path, never fatal
        logger.warning("failed to write debug frame %s: %s", dest, e)


class LHPRVLNMirrorSFTPromptBuilder(LHPRVLNSFTPromptBuilder):
    """SFT builder that mirrors observations and swaps left/right actions."""

    def build_messages(self, memory: AgentMemory) -> list[dict]:
        # Record the step index so ``_build_content`` can label debug dumps.
        # ``memory.steps`` holds completed steps; the NEXT emission maps to
        # index ``len(memory.steps)``.
        self._debug_step_idx = len(memory.steps)
        return super().build_messages(memory)

    def _build_content(
        self,
        instruction: str,
        history_paths: list[str],
        current_views: list[str],
    ) -> list[dict]:
        # Mirroring the scene swaps the semantic meaning of the side
        # cameras: what was captured on the right now lies on the agent's
        # left (flipped). Feed flip(right) into the "left side" slot and
        # flip(left) into the "right side" slot; front stays centered.
        swapped_views = list(current_views)
        if len(swapped_views) == 3:
            swapped_views[0], swapped_views[2] = swapped_views[2], swapped_views[0]

        content: list[dict] = []

        content.append({"type": "text", "text": (
            "You are an autonomous navigation robot. You will get a task "
            "with historical pictures and current pictures you see.\n"
            f"Based on this information, you need to decide your next "
            f"{self.window_size} actions, which could involve "
            "<|left|>,<|right|>,<|forward|>. "
            "If you finish your mission, output <|stop|>. "
            "Here are some examples: "
            "<|left|><|forward|><|forward|><|stop|>, "
            "<|forward|><|forward|><|forward|><|left|><|forward|> or <|stop|>"
        )})

        debug_dir = _resolve_debug_dir()
        step_idx = getattr(self, "_debug_step_idx", 0)

        if history_paths:
            content.append({"type": "text", "text": "\n# Your historical pictures are: "})
            for h_idx, path in enumerate(history_paths):
                img_url = _encode_image_flipped(path)
                if img_url:
                    content.append({"type": "image_url", "image_url": {"url": img_url}})
                if debug_dir is not None:
                    _dump_data_url(
                        img_url,
                        debug_dir / f"step_{step_idx:04d}_history_{h_idx:03d}.png",
                    )

        view_labels = ["left side", "front side", "right side"]
        # Slot name used for the filename (`left_side` -> `left` etc.) so it's
        # easy to diff against the sim-rendered ``step_NNNN_<cam>.png`` files.
        slot_filenames = ["left", "front", "right"]
        content.append({"type": "text", "text": "\n# Your current observations are "})
        for i, path in enumerate(swapped_views):
            if i > 0:
                content.append({"type": "text", "text": ", "})
            content.append({"type": "text", "text": f"{view_labels[i]}: "})
            img_url = _encode_image_flipped(path)
            if img_url:
                content.append({"type": "image_url", "image_url": {"url": img_url}})
            if debug_dir is not None and i < len(slot_filenames):
                _dump_data_url(
                    img_url,
                    debug_dir / f"step_{step_idx:04d}_{slot_filenames[i]}.png",
                )

        content.append({"type": "text", "text": (
            f"\n# Your mission is: {instruction}\n"
            "PS: The mission is complex. You may infer several sub-tasks "
            "within the mission, and output <|stop|> when a sub-task is "
            f"achieved. So far, you have output <|stop|> {self._stop_count} "
            "times. Historical information reflects progress up to the "
            "current subgoal. <|NAV|>"
        )})

        return content

    def parse_response(self, llm_response: str, memory: AgentMemory) -> list[Action]:
        actions = super().parse_response(llm_response, memory)
        remapped: list[Action] = []
        for act in actions:
            new_name = _ACTION_REMAP.get(act.action_name, act.action_name)
            if new_name == act.action_name:
                remapped.append(act)
            else:
                remapped.append(Action(action_name=new_name))
        return remapped
