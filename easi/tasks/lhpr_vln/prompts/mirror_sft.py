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

from easi.agents.prompt_builder import _encode_image_base64 as _base_encode
from easi.core.episode import Action
from easi.core.memory import AgentMemory
from easi.tasks.lhpr_vln.prompts.sft import LHPRVLNSFTPromptBuilder
from easi.utils.logging import get_logger

logger = get_logger(__name__)

_ACTION_REMAP = {"turn_left": "turn_right", "turn_right": "turn_left"}


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


class LHPRVLNMirrorSFTPromptBuilder(LHPRVLNSFTPromptBuilder):
    """SFT builder that mirrors observations and swaps left/right actions."""

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

        if history_paths:
            content.append({"type": "text", "text": "\n# Your historical pictures are: "})
            for path in history_paths:
                img_url = _encode_image_flipped(path)
                if img_url:
                    content.append({"type": "image_url", "image_url": {"url": img_url}})

        view_labels = ["left side", "front side", "right side"]
        content.append({"type": "text", "text": "\n# Your current observations are "})
        for i, path in enumerate(swapped_views):
            if i > 0:
                content.append({"type": "text", "text": ", "})
            content.append({"type": "text", "text": f"{view_labels[i]}: "})
            img_url = _encode_image_flipped(path)
            if img_url:
                content.append({"type": "image_url", "image_url": {"url": img_url}})

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
