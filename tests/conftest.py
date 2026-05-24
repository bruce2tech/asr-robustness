"""Shared synthetic-signal fixtures.

The degradation harness is tested entirely on synthetic signals so the suite
runs with zero downloaded data -- a property the real pipeline relies on too.
"""

import numpy as np
import pytest

SR = 16_000


@pytest.fixture
def sr() -> int:
    return SR


@pytest.fixture
def speech_like() -> np.ndarray:
    """A 1 s voiced-speech-like signal: a few harmonics under an envelope."""
    t = np.arange(SR) / SR
    sig = sum(np.sin(2 * np.pi * f * t) / (i + 1) for i, f in enumerate([200, 700, 1500]))
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)  # 3 Hz syllable-rate envelope
    sig = sig * envelope
    return (0.5 * sig / np.max(np.abs(sig))).astype(np.float32)


@pytest.fixture
def white_noise() -> np.ndarray:
    """1.5 s of white noise (longer than `speech_like`, so it must be cropped)."""
    rng = np.random.default_rng(0)
    return (0.3 * rng.standard_normal(int(1.5 * SR))).astype(np.float32)


@pytest.fixture
def synthetic_rir() -> np.ndarray:
    """A 0.3 s RIR: a direct-path impulse followed by an exponentially decaying tail."""
    rng = np.random.default_rng(1)
    n = int(0.3 * SR)
    rir = np.zeros(n, dtype=np.float64)
    direct = 60
    rir[direct] = 1.0
    tail_idx = np.arange(direct, n)
    decay = np.exp(-(tail_idx - direct) / (0.05 * SR))
    rir[direct:] += 0.4 * rng.standard_normal(tail_idx.size) * decay
    return rir.astype(np.float32)


def tone(freq: float, seconds: float = 1.0, amp: float = 0.5) -> np.ndarray:
    """A pure sine tone -- used to probe frequency-selective effects."""
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
