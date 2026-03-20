"""Runtime configuration applied before importing LiteLLM."""

from __future__ import annotations

import importlib
import os
from typing import Any


def configure_litellm_runtime_env() -> None:
    """Force LiteLLM to use its bundled model cost map instead of fetching remotely."""
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")


configure_litellm_runtime_env()
litellm: Any = importlib.import_module("litellm")
