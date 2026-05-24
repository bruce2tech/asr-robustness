"""ASR model adapters and registry.

Importing this package registers only the lightweight stub model. Heavy
adapters (torch / transformers / espnet) are imported lazily via
:func:`ensure_loaded` so that merely listing or scoring does not pull in
multi-gigabyte dependencies.
"""

from __future__ import annotations

import importlib

from asr_robustness.models import stub  # noqa: F401  -- registers the "stub" model
from asr_robustness.models.base import ASRModel
from asr_robustness.models.registry import available, create, register

# Map model key -> adapter module that registers it.
_ADAPTER_MODULES = {
    "whisper": "asr_robustness.models.whisper_hf",
    "wav2vec2": "asr_robustness.models.wav2vec2_hf",
    "espnet": "asr_robustness.models.espnet_pretrained",
}

__all__ = ["ASRModel", "available", "create", "register", "ensure_loaded"]


def ensure_loaded(key: str) -> None:
    """Import the adapter module that registers model ``key`` (if it is heavy)."""
    module = _ADAPTER_MODULES.get(key)
    if module:
        importlib.import_module(module)
