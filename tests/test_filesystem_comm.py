"""Tests for filesystem-based IPC communication."""

import json
import tempfile
from pathlib import Path

import pytest

from easi.communication.filesystem import (
    atomic_write_json,
    cleanup_workspace,
    create_workspace,
    delete_file,
    read_json,
    write_command,
)
from easi.communication.schemas import (
    make_close_command,
    make_observation_response,
    make_reset_command,
    make_step_command,
    parse_action_from_command,
    parse_observation,
    parse_step_result,
)
from easi.core.episode import Action


class TestAtomicWriteJson:
    def test_write_and_read(self, tmp_path):
        path = tmp_path / "test.json"
        data = {"key": "value", "num": 42}
        atomic_write_json(path, data)
        assert read_json(path) == data

    def test_no_tmp_file_left(self, tmp_path):
        path = tmp_path / "test.json"
        atomic_write_json(path, {"a": 1})
        assert not (tmp_path / "test.tmp").exists()

    def test_read_nonexistent(self, tmp_path):
        assert read_json(tmp_path / "nope.json") is None


class TestDeleteFile:
    def test_delete_existing(self, tmp_path):
        path = tmp_path / "test.txt"
        path.write_text("hello")
        delete_file(path)
        assert not path.exists()

    def test_delete_nonexistent(self, tmp_path):
        delete_file(tmp_path / "nope.txt")  # should not raise


class TestWorkspace:
    def test_create_and_cleanup(self):
        ws = create_workspace()
        assert ws.exists()
        cleanup_workspace(ws)
        assert not ws.exists()


class TestSchemas:
    def test_reset_command(self):
        cmd = make_reset_command("ep_001", {"scene": "A"})
        assert cmd["type"] == "reset"
        assert cmd["episode_id"] == "ep_001"
        assert cmd["reset_config"] == {"scene": "A"}

    def test_step_command(self):
        action = Action(action_name="MoveAhead", params={"distance": 0.25})
        cmd = make_step_command(action)
        assert cmd["type"] == "step"
        assert cmd["action"]["action_name"] == "MoveAhead"
        assert cmd["action"]["params"]["distance"] == 0.25

    def test_close_command(self):
        cmd = make_close_command()
        assert cmd["type"] == "close"

    def test_observation_response(self):
        resp = make_observation_response(
            rgb_path="/tmp/rgb.png",
            agent_pose=[1.0, 2.0, 3.0],
            reward=0.5,
            done=True,
        )
        assert resp["status"] == "ok"
        assert resp["observation"]["rgb_path"] == "/tmp/rgb.png"
        assert resp["reward"] == 0.5
        assert resp["done"] is True

    def test_parse_observation(self):
        resp = make_observation_response(rgb_path="/tmp/rgb.png", depth_path="/tmp/depth.png")
        obs = parse_observation(resp)
        assert obs.rgb_path == "/tmp/rgb.png"
        assert obs.depth_path == "/tmp/depth.png"

    def test_parse_step_result(self):
        resp = make_observation_response(rgb_path="/tmp/rgb.png", reward=1.0, done=True)
        result = parse_step_result(resp)
        assert result.reward == 1.0
        assert result.done is True
        assert result.observation.rgb_path == "/tmp/rgb.png"

    def test_parse_action_from_command(self):
        action = Action(action_name="TurnLeft", params={"angle": 90.0})
        cmd = make_step_command(action)
        parsed = parse_action_from_command(cmd)
        assert parsed.action_name == "TurnLeft"
        assert parsed.params["angle"] == 90.0


    def test_step_command_with_skill_text(self):
        """action_name carries full skill text for EB-Alfred (e.g., 'find a Cabinet').

        The IPC protocol supports this natively — no schema changes needed.
        """
        action = Action(action_name="find a Cabinet", params={})
        cmd = make_step_command(action)
        assert cmd["action"]["action_name"] == "find a Cabinet"

        # Round-trip: serialize and parse back
        parsed = parse_action_from_command(cmd)
        assert parsed.action_name == "find a Cabinet"

    def test_observation_response_with_task_metrics(self):
        """info field carries task_success and task_progress for EB-Alfred."""
        resp = make_observation_response(
            rgb_path="/tmp/rgb.png",
            reward=0.5,
            done=False,
            info={"task_success": 0.0, "task_progress": 0.33, "last_action_success": 1.0},
        )
        result = parse_step_result(resp)
        assert result.info["task_success"] == 0.0
        assert result.info["task_progress"] == 0.33
        assert result.info["last_action_success"] == 1.0


class TestWriteCommand:
    def test_clears_old_response(self, tmp_path):
        # Write a response file
        resp_path = tmp_path / "response.json"
        resp_path.write_text('{"old": true}')

        # write_command should delete it
        write_command(tmp_path, make_reset_command("ep_001"))
        assert not resp_path.exists()
        assert (tmp_path / "command.json").exists()
