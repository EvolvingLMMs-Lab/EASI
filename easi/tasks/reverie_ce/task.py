"""REVERIE-CE task for EASI.

Navigation-only evaluation of REVERIE in continuous environments.
Inherits from VLNCETask — same metrics, same bridge protocol.
"""
from __future__ import annotations

from pathlib import Path

from easi.tasks.vlnce_r2r.task import VLNCETask
from easi.utils.logging import get_logger

logger = get_logger(__name__)


class ReverieCETask(VLNCETask):

    def get_task_yaml_path(self) -> Path:
        return Path(__file__).parent / "_base.yaml"

    def get_bridge_script_path(self) -> Path:
        return Path(__file__).parent / "bridge.py"
