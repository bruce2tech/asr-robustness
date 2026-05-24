"""A dependency-free stub ASR model.

Exists so the registry, scoring, and decode-runner logic can be unit-tested
without torch, transformers, or any model download. The stub returns a
caller-supplied transcript (optionally per-utterance), which lets tests drive
exact, predictable WER outcomes.
"""

from __future__ import annotations

import numpy as np

from asr_robustness.models.base import ASRModel
from asr_robustness.models.registry import register


@register("stub")
class StubModel(ASRModel):
    """Returns a fixed transcript, or one looked up per call index."""

    def __init__(self, transcript: str = "", responses: list[str] | None = None):
        self.name = "stub"
        self._transcript = transcript
        self._responses = responses
        self._calls = 0

    def transcribe(self, audio: np.ndarray, sr: int) -> str:
        idx = self._calls
        self._calls += 1
        if self._responses is not None:
            return self._responses[idx % len(self._responses)]
        return self._transcript
