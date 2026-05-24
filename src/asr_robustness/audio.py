"""Audio I/O and resampling helpers.

Everything downstream assumes **mono float32** signals at a single working
sample rate (16 kHz by default -- the standard for ASR front-ends).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16_000
EPS = 1e-10


def to_mono(x: np.ndarray) -> np.ndarray:
    """Collapse a (samples, channels) array to mono by averaging channels."""
    if x.ndim == 1:
        return x
    return x.mean(axis=1)


def resample(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample with a polyphase filter (good quality, no external deps)."""
    if orig_sr == target_sr:
        return x.astype(np.float32)
    g = np.gcd(int(orig_sr), int(target_sr))
    y = resample_poly(x, target_sr // g, orig_sr // g)
    return y.astype(np.float32)


def load_audio(path: str | Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Load an audio file as a mono float32 signal at ``target_sr``.

    Returns ``(signal, sample_rate)``.
    """
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    x = to_mono(np.asarray(x, dtype=np.float32))
    x = resample(x, sr, target_sr)
    return x, target_sr


def save_audio(path: str | Path, x: np.ndarray, sr: int = TARGET_SR) -> None:
    """Write a float32 signal to ``path`` (parent directories are created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(x, dtype=np.float32), sr)


def rms(x: np.ndarray) -> float:
    """Root-mean-square level of a signal (float64 accumulation)."""
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(x)) + EPS))


def peak_normalize(x: np.ndarray, peak: float = 0.99) -> np.ndarray:
    """Scale a signal so its largest absolute sample equals ``peak``.

    Used before writing degraded audio so that additive noise / convolution
    gain does not clip the int conversion in the WAV writer.
    """
    m = float(np.max(np.abs(x))) if x.size else 0.0
    if m < EPS:
        return x.astype(np.float32)
    return (x * (peak / m)).astype(np.float32)
