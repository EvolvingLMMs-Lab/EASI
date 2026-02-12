"""Dynamic class import utility."""

from __future__ import annotations

import importlib


def import_class(dotted_path: str) -> type:
    """Import a class from a dotted path.

    Args:
        dotted_path: Fully qualified class name
            (e.g., "easi.tasks.ebalfred.prompts.EBAlfredPromptBuilder").

    Returns:
        The class object.

    Raises:
        ImportError: If the module cannot be found.
        AttributeError: If the class does not exist in the module.
    """
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
