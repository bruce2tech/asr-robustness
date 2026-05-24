"""Sample banks for noise clips and room impulse responses.

A bank is a named collection of audio clips. The degradation pipeline draws
from a bank using the per-utterance random seed, so the *same* noise clip /
RIR is chosen every time a given utterance is degraded -- reproducibility
without baking the audio into the repo.

Entries may be in-memory arrays (convenient for tests) or file paths (loaded
lazily and cached). IDs are stable strings; for directory-loaded banks the ID
is the path relative to the bank root, so subset filtering by prefix works
(e.g. MUSAN's ``noise/``, ``music/``, ``speech/`` subtrees).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from asr_robustness.audio import TARGET_SR, load_audio


class AudioBank:
    """A named, lazily-loaded collection of audio clips."""

    def __init__(self, entries: dict[str, np.ndarray | str | Path], sr: int = TARGET_SR):
        self._raw: dict[str, np.ndarray | str | Path] = dict(entries)
        self._cache: dict[str, np.ndarray] = {}
        self._ids: list[str] = sorted(self._raw)
        self.sr = sr

    @classmethod
    def from_dir(cls, root: str | Path, sr: int = TARGET_SR, pattern: str = "*.wav") -> "AudioBank":
        """Build a bank from every file matching ``pattern`` under ``root``."""
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"bank directory not found: {root}")
        files = sorted(root.rglob(pattern))
        if not files:
            raise FileNotFoundError(f"no files matching {pattern!r} under {root}")
        entries = {str(f.relative_to(root)): f for f in files}
        return cls(entries, sr=sr)

    def __len__(self) -> int:
        return len(self._ids)

    def ids(self, prefix: str | None = None) -> list[str]:
        """All clip IDs, optionally filtered to those starting with ``prefix``."""
        if prefix is None:
            return list(self._ids)
        return [i for i in self._ids if i.startswith(prefix)]

    def get(self, clip_id: str) -> np.ndarray:
        """Return the clip for ``clip_id`` (loading + caching paths on demand)."""
        if clip_id in self._cache:
            return self._cache[clip_id]
        entry = self._raw[clip_id]
        if isinstance(entry, (str, Path)):
            clip, _ = load_audio(entry, target_sr=self.sr)
        else:
            clip = np.asarray(entry, dtype=np.float32)
        self._cache[clip_id] = clip
        return clip

    def sample(
        self, rng: np.random.Generator, prefix: str | None = None
    ) -> tuple[str, np.ndarray]:
        """Pick a clip reproducibly from ``rng``; returns ``(clip_id, signal)``."""
        pool = self.ids(prefix)
        if not pool:
            raise ValueError(f"bank has no clips for prefix={prefix!r}")
        clip_id = pool[int(rng.integers(0, len(pool)))]
        return clip_id, self.get(clip_id)


class NoiseBank(AudioBank):
    """A bank of background-noise clips (e.g. MUSAN)."""


class RIRBank(AudioBank):
    """A bank of room impulse responses (e.g. the OpenSLR RIR corpus)."""
