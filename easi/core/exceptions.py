"""Custom exception hierarchy for EASI."""

from __future__ import annotations


class EASIError(Exception):
    """Base exception for all EASI errors."""


class EnvironmentSetupError(EASIError):
    """Raised when a simulator environment fails to install or validate."""

    def __init__(self, message: str, missing_deps: list[str] | None = None):
        super().__init__(message)
        self.missing_deps = missing_deps or []


class SimulatorError(EASIError):
    """Raised when a simulator encounters an error during operation."""


class SimulatorTimeoutError(SimulatorError):
    """Raised when a simulator operation exceeds the configured timeout."""

    def __init__(self, message: str, timeout: float):
        super().__init__(message)
        self.timeout = timeout


class ActionParseError(EASIError):
    """Raised when an LLM response cannot be parsed into a valid Action."""

    def __init__(self, message: str, raw_response: str):
        super().__init__(message)
        self.raw_response = raw_response


class DatasetError(EASIError):
    """Raised when a dataset cannot be loaded or downloaded."""
