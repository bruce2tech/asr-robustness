"""The uniform ASR model interface.

Every benchmarked system -- Whisper, wav2vec 2.0, ESPnet -- is wrapped in an
adapter implementing :class:`ASRModel`. The evaluation harness then treats them
identically: hand over a mono float32 waveform, receive a transcript string.
That uniformity is what makes a fair multi-model benchmark possible.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ASRModel(ABC):
    """Base class for an ASR system under evaluation."""

    #: Short identifier used in results and plots (set by the adapter).
    name: str = "asr-model"

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sr: int) -> str:
        """Transcribe one mono float32 waveform to a transcript string."""
        raise NotImplementedError

    def transcribe_batch(self, audios: list[np.ndarray], sr: int) -> list[str]:
        """Transcribe several waveforms; adapters may override for true batching."""
        return [self.transcribe(a, sr) for a in audios]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
