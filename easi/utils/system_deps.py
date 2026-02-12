"""System dependency checker for simulators.

Verifies that required system-level packages (xvfb, EGL, conda, etc.) are
installed before attempting environment setup.
"""

from __future__ import annotations

import subprocess

from easi.core.exceptions import EnvironmentSetupError
from easi.utils.logging import get_logger

logger = get_logger(__name__)


# Known system dependencies and how to check/install them
KNOWN_DEPS: dict[str, dict[str, str | list[str]]] = {
    "xvfb": {
        "check_command": ["which", "Xvfb"],
        "install_hint": "sudo apt-get install -y xvfb",
    },
    "egl": {
        "check_command": ["ldconfig", "-p"],
        "check_grep": "libEGL",
        "install_hint": "sudo apt-get install -y libegl1-mesa-dev",
    },
    "osmesa": {
        "check_command": ["ldconfig", "-p"],
        "check_grep": "libOSMesa",
        "install_hint": "sudo apt-get install -y libosmesa6-dev",
    },
    "conda": {
        "check_command": ["which", "conda"],
        "install_hint": "Install Miniconda: https://docs.conda.io/en/latest/miniconda.html",
    },
}


class SystemDependencyChecker:
    """Checks for system-level dependencies."""

    def check(self, dep_name: str) -> bool:
        """Check if a single system dependency is available.

        Returns True if the dependency is found, False otherwise.
        """
        dep_info = KNOWN_DEPS.get(dep_name)
        if dep_info is None:
            logger.warning("Unknown system dependency: %s (skipping check)", dep_name)
            return True

        try:
            result = subprocess.run(
                dep_info["check_command"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            # If there's a grep pattern, check stdout for it
            check_grep = dep_info.get("check_grep")
            if check_grep:
                return check_grep in result.stdout
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def check_all(self, deps: list[str]) -> list[str]:
        """Check multiple dependencies, returning list of missing ones."""
        missing = []
        for dep in deps:
            if not self.check(dep):
                missing.append(dep)
        return missing

    def assert_all(self, deps: list[str]) -> None:
        """Assert all dependencies are present, raising with install hints if not."""
        missing = self.check_all(deps)
        if not missing:
            return

        hints = []
        for dep in missing:
            dep_info = KNOWN_DEPS.get(dep, {})
            hint = dep_info.get("install_hint", f"Install '{dep}' manually")
            hints.append(f"  - {dep}: {hint}")

        raise EnvironmentSetupError(
            f"Missing system dependencies:\n" + "\n".join(hints),
            missing_deps=missing,
        )
