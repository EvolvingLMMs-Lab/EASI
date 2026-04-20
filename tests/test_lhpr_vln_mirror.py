"""Tests for LHPRVLNMirrorSFTPromptBuilder: flip + action remap."""
import base64
import io

import pytest
from PIL import Image

from easi.core.episode import Action, Observation
from easi.core.memory import AgentMemory
from easi.tasks.lhpr_vln.prompts.mirror_sft import (
    LHPRVLNMirrorSFTPromptBuilder,
    _encode_image_flipped,
)


ACTION_SPACE = ["move_forward", "turn_left", "turn_right", "stop"]


@pytest.fixture
def builder():
    b = LHPRVLNMirrorSFTPromptBuilder(window_size=5, max_history_images=20)
    b.set_action_space(ACTION_SPACE)
    return b


@pytest.fixture
def memory():
    return AgentMemory(
        task_description="go to the kitchen",
        action_space=ACTION_SPACE,
        current_observation=Observation(rgb_path="/dev/null"),
    )


def _make_image(path, color):
    img = Image.new("RGB", (4, 2), color=(0, 0, 0))
    # Left column red, rest black so flip is detectable pixel-wise
    img.putpixel((0, 0), color)
    img.putpixel((0, 1), color)
    img.save(path, format="PNG")


def test_flip_helper_returns_data_url(tmp_path):
    p = tmp_path / "img.png"
    _make_image(p, (255, 0, 0))

    data_url = _encode_image_flipped(str(p))
    assert data_url is not None
    assert data_url.startswith("data:image/png;base64,")

    _, b64 = data_url.split(",", 1)
    decoded = base64.b64decode(b64)
    out = Image.open(io.BytesIO(decoded)).convert("RGB")

    # Original red pixels at (0, *); after horizontal flip they're at (w-1, *)
    assert out.size == (4, 2)
    assert out.getpixel((3, 0)) == (255, 0, 0)
    assert out.getpixel((3, 1)) == (255, 0, 0)
    assert out.getpixel((0, 0)) == (0, 0, 0)


def test_flip_helper_missing_file_returns_none(tmp_path):
    assert _encode_image_flipped(str(tmp_path / "nope.png")) is None


def test_parse_response_swaps_left_right(builder, memory):
    actions = builder.parse_response("<action><|left|></action>", memory)
    assert len(actions) == 1
    assert actions[0].action_name == "turn_right"


def test_parse_response_swaps_right_to_left(builder, memory):
    actions = builder.parse_response("<action><|right|></action>", memory)
    assert len(actions) == 1
    assert actions[0].action_name == "turn_left"


def test_parse_response_passes_forward_and_stop(builder, memory):
    actions = builder.parse_response(
        "<action><|forward|><|stop|></action>", memory
    )
    assert [a.action_name for a in actions] == ["move_forward", "stop"]


def test_parse_response_multi_action_mixed(builder, memory):
    actions = builder.parse_response(
        "<action><|left|><|forward|><|right|><|stop|></action>", memory
    )
    assert [a.action_name for a in actions] == [
        "turn_right", "move_forward", "turn_left", "stop",
    ]


def test_build_content_swaps_side_views(builder, tmp_path):
    left = tmp_path / "left.png"
    front = tmp_path / "front.png"
    right = tmp_path / "right.png"
    _make_image(left, (255, 0, 0))   # red
    _make_image(front, (0, 255, 0))  # green
    _make_image(right, (0, 0, 255))  # blue

    content = builder._build_content(
        instruction="test",
        history_paths=[],
        current_views=[str(left), str(front), str(right)],
    )

    # Extract image_url blocks in order
    urls = [b["image_url"]["url"] for b in content if b.get("type") == "image_url"]
    assert len(urls) == 3

    def _first_pixel(data_url, xy):
        _, b64 = data_url.split(",", 1)
        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        return img.getpixel(xy)

    # Slot 0 = "left side" should contain flipped right-camera (blue)
    # Slot 1 = "front side" should contain flipped front-camera (green)
    # Slot 2 = "right side" should contain flipped left-camera  (red)
    # Post-flip, the color-carrying column moved from x=0 to x=3.
    assert _first_pixel(urls[0], (3, 0)) == (0, 0, 255)
    assert _first_pixel(urls[1], (3, 0)) == (0, 255, 0)
    assert _first_pixel(urls[2], (3, 0)) == (255, 0, 0)
