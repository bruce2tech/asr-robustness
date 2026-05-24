"""Acoustic degradation primitives.

Each effect takes a clean mono float32 signal and returns ``(degraded, info)``,
where ``info`` is a dict recording the *actual* parameters applied. That dict is
written into the evaluation manifest so every degraded utterance is fully
reproducible and every result can be sliced (e.g. WER-vs-SNR breakdowns).

Conventions
-----------
* Signals are 1-D float32 in roughly [-1, 1]; computation is done in float64.
* SNR mixing is RMS-based: ``snr_db`` is the ratio of clean RMS to noise RMS.
* Effects never normalize loudness silently except where physically motivated
  (reverb energy compensation); callers peak-normalize before writing to disk.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal as sps

EPS = 1e-10


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(x, dtype=np.float64))) + EPS))


def _match_length(noise: np.ndarray, n: int, rng: np.random.Generator | None) -> np.ndarray:
    """Crop or tile ``noise`` to exactly ``n`` samples.

    If the noise clip is longer than the target a random offset is chosen (so
    repeated runs over the same utterance can see different noise segments);
    if shorter it is tiled. ``rng`` makes the offset reproducible.
    """
    noise = np.asarray(noise, dtype=np.float64)
    if noise.size == 0:
        raise ValueError("noise signal is empty")
    if noise.size >= n:
        start = 0 if rng is None else int(rng.integers(0, noise.size - n + 1))
        return noise[start : start + n]
    reps = int(np.ceil(n / noise.size))
    return np.tile(noise, reps)[:n]


def add_noise(
    clean: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    """Add background noise scaled to a target signal-to-noise ratio.

    The noise is rescaled so that ``20*log10(rms(clean)/rms(noise)) == snr_db``,
    then added sample-wise. Because ``mixed - clean`` is exactly the scaled
    noise, the realized SNR equals ``snr_db`` by construction.
    """
    clean = np.asarray(clean, dtype=np.float64)
    noise = _match_length(noise, clean.size, rng)
    clean_rms = _rms(clean)
    target_noise_rms = clean_rms / (10.0 ** (snr_db / 20.0))
    scaled = noise * (target_noise_rms / (_rms(noise) + EPS))
    mixed = clean + scaled
    info = {
        "effect": "add_noise",
        "snr_db": float(snr_db),
        "clean_rms": clean_rms,
        "noise_rms": _rms(scaled),
    }
    return mixed.astype(np.float32), info


def add_babble(
    clean: np.ndarray,
    talkers: list[np.ndarray],
    snr_db: float,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    """Mix in multi-talker babble -- a sum of competing speakers -- at a target SNR.

    Babble differs from stationary noise in kind, not just degree: it contains
    *intelligible competing speech*, which hijacks attention and defeats simple
    spectral noise models. At the same SNR it is far more disruptive to ASR (and
    to human listeners) than broadband noise, and it is a closer model of
    crowded-room operational audio. The summed talkers are treated as a single
    noise source and scaled by :func:`add_noise` to hit ``snr_db``.
    """
    clean = np.asarray(clean, dtype=np.float64)
    if not talkers:
        raise ValueError("babble requires at least one talker")
    summed = np.zeros(clean.size, dtype=np.float64)
    for talker in talkers:
        summed += _match_length(np.asarray(talker, dtype=np.float64), clean.size, rng)
    mixed, noise_info = add_noise(clean, summed, snr_db, rng=None)
    info = {
        "effect": "add_babble",
        "snr_db": float(snr_db),
        "n_talkers": len(talkers),
        "clean_rms": noise_info["clean_rms"],
        "noise_rms": noise_info["noise_rms"],
    }
    return mixed, info


def add_reverb(
    clean: np.ndarray,
    rir: np.ndarray,
    preserve_energy: bool = True,
) -> tuple[np.ndarray, dict]:
    """Convolve a signal with a room impulse response.

    The RIR is peak-normalized and time-aligned to its direct-path peak so the
    reverberant output stays sample-aligned with the clean reference (important
    for fair WER scoring). With ``preserve_energy`` the output RMS is matched to
    the input so reverb does not also act as a gain change.
    """
    clean = np.asarray(clean, dtype=np.float64)
    rir = np.asarray(rir, dtype=np.float64)
    if rir.size == 0:
        raise ValueError("RIR is empty")
    rir = rir / (np.max(np.abs(rir)) + EPS)
    peak = int(np.argmax(np.abs(rir)))
    wet_full = sps.fftconvolve(clean, rir)
    wet = wet_full[peak : peak + clean.size]
    if preserve_energy and _rms(wet) > EPS:
        wet = wet * (_rms(clean) / _rms(wet))
    info = {"effect": "add_reverb", "rir_len": int(rir.size), "rir_peak_idx": peak}
    return wet.astype(np.float32), info


def narrowband(
    clean: np.ndarray,
    sr: int,
    low_hz: float = 300.0,
    high_hz: float = 3400.0,
) -> tuple[np.ndarray, dict]:
    """Band-limit a signal to the telephone band (300-3400 Hz by default).

    Models the bandwidth restriction of a narrowband voice channel, which
    discards high-frequency cues many ASR front-ends rely on.
    """
    clean = np.asarray(clean, dtype=np.float64)
    nyq = sr / 2.0
    high_hz = min(high_hz, nyq - 1.0)
    sos = sps.butter(4, [low_hz / nyq, high_hz / nyq], btype="band", output="sos")
    y = sps.sosfilt(sos, clean)
    info = {"effect": "narrowband", "low_hz": float(low_hz), "high_hz": float(high_hz)}
    return y.astype(np.float32), info


def mu_law_codec(clean: np.ndarray, mu: int = 255) -> tuple[np.ndarray, dict]:
    """G.711 mu-law companding round-trip: float -> 8-bit -> float.

    This is the quantization a standard narrowband telephone codec applies.
    The encode/decode round-trip is lossy, injecting codec-grade quantization
    distortion that is concentrated in low-amplitude (quiet) regions.
    """
    x = np.clip(np.asarray(clean, dtype=np.float64), -1.0, 1.0)
    # Encode: mu-law compress, then quantize to 8 bits (256 levels).
    comp = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    levels = np.round((comp + 1.0) / 2.0 * 255.0)
    comp_q = levels / 255.0 * 2.0 - 1.0
    # Decode: mu-law expand.
    y = np.sign(comp_q) * (1.0 / mu) * ((1.0 + mu) ** np.abs(comp_q) - 1.0)
    info = {"effect": "mu_law_codec", "mu": int(mu)}
    return y.astype(np.float32), info


def clip_signal(clean: np.ndarray, percentile: float = 99.0) -> tuple[np.ndarray, dict]:
    """Hard-clip amplitude at a percentile threshold (overdriven-mic distortion)."""
    clean = np.asarray(clean, dtype=np.float64)
    thr = float(np.percentile(np.abs(clean), percentile)) if clean.size else 0.0
    thr = max(thr, EPS)
    y = np.clip(clean, -thr, thr)
    info = {"effect": "clip", "percentile": float(percentile), "threshold": thr}
    return y.astype(np.float32), info


def gain(clean: np.ndarray, gain_db: float) -> tuple[np.ndarray, dict]:
    """Apply a fixed gain in decibels."""
    factor = 10.0 ** (gain_db / 20.0)
    y = np.asarray(clean, dtype=np.float64) * factor
    return y.astype(np.float32), {"effect": "gain", "gain_db": float(gain_db)}


def packet_loss(
    clean: np.ndarray,
    sr: int,
    loss_rate: float = 0.1,
    packet_ms: float = 20.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    """Drop fixed-length packets to model VoIP / radio packet loss.

    The signal is split into ``packet_ms`` frames; each is silenced with
    probability ``loss_rate``. Unlike additive noise this *removes* information
    outright -- a revealing stressor because CTC and generative models handle
    gaps very differently (clean deletion vs. hallucinating across the gap).
    """
    out = np.asarray(clean, dtype=np.float64).copy()
    rng = rng if rng is not None else np.random.default_rng()
    packet = max(1, int(round(sr * packet_ms / 1000.0)))
    n_packets = (out.size + packet - 1) // packet
    dropped = 0
    for i in range(n_packets):
        if rng.random() < loss_rate:
            out[i * packet : (i + 1) * packet] = 0.0
            dropped += 1
    info = {
        "effect": "packet_loss",
        "loss_rate": float(loss_rate),
        "packet_ms": float(packet_ms),
        "n_packets": int(n_packets),
        "dropped": int(dropped),
    }
    return out.astype(np.float32), info


# name -> (intermediate container extension, ffmpeg encode arguments)
_FFMPEG_CODECS = {
    "g726": ("wav", ["-ar", "8000", "-ac", "1", "-c:a", "g726", "-b:a", "16k"]),
    "g722": ("wav", ["-ar", "16000", "-ac", "1", "-c:a", "g722"]),
    # ffmpeg's native opus encoder only accepts 48 kHz input; the decode step
    # then resamples back to the working sample rate.
    "opus": ("ogg", ["-ar", "48000", "-ac", "1", "-c:a", "opus", "-strict", "-2",
                     "-b:a", "{bitrate}"]),
}


def ffmpeg_available() -> bool:
    """True if an ``ffmpeg`` binary is on PATH."""
    return shutil.which("ffmpeg") is not None


def _run_ffmpeg(args: list[str]) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args],
        check=True,
        capture_output=True,
    )


def _fit_length(x: np.ndarray, n: int) -> np.ndarray:
    """Trim or zero-pad ``x`` to exactly ``n`` samples (codecs shift length slightly)."""
    if x.size >= n:
        return x[:n]
    return np.concatenate([x, np.zeros(n - x.size, dtype=x.dtype)])


def apply_codec(
    clean: np.ndarray,
    sr: int,
    codec: str = "g726",
    bitrate: str = "8k",
) -> tuple[np.ndarray, dict]:
    """Round-trip audio through a real telephony / VoIP codec (via ffmpeg).

    A genuine lossy encode+decode injects the codec's own artifacts -- ADPCM
    quantization, low-bitrate transform-coding distortion -- which are exactly
    what transmitted, operational audio carries. ``g726`` (narrowband) and
    ``g722`` (wideband) are standard telephony codecs; ``opus`` is the modern
    VoIP codec. ``bitrate`` applies to ``opus`` only.
    """
    if codec not in _FFMPEG_CODECS:
        raise ValueError(f"unknown codec {codec!r}; known: {sorted(_FFMPEG_CODECS)}")
    if not ffmpeg_available():
        raise RuntimeError(
            "ffmpeg not found -- install it (e.g. `brew install ffmpeg`) for codec degradations"
        )
    ext, encode_args = _FFMPEG_CODECS[codec]
    encode_args = [a.replace("{bitrate}", bitrate) for a in encode_args]
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        src, encoded, out = tmp / "in.wav", tmp / f"enc.{ext}", tmp / "out.wav"
        sf.write(str(src), np.asarray(clean, dtype=np.float32), sr)
        _run_ffmpeg(["-i", str(src), *encode_args, str(encoded)])
        _run_ffmpeg(["-i", str(encoded), "-ar", str(sr), "-ac", "1", str(out)])
        decoded, _ = sf.read(str(out), dtype="float32")
    decoded = _fit_length(np.asarray(decoded, dtype=np.float32), len(clean))
    return decoded, {"effect": "apply_codec", "codec": codec, "bitrate": bitrate}


def measure_snr(clean: np.ndarray, degraded: np.ndarray) -> float:
    """Estimate SNR (dB) treating ``degraded - clean`` as the noise component.

    Valid when ``degraded`` is a sample-aligned, additive corruption of
    ``clean`` -- used in tests to confirm :func:`add_noise` hits its target.
    """
    clean = np.asarray(clean, dtype=np.float64)
    degraded = np.asarray(degraded, dtype=np.float64)
    n = min(clean.size, degraded.size)
    noise = degraded[:n] - clean[:n]
    return 20.0 * np.log10((_rms(clean[:n]) + EPS) / (_rms(noise) + EPS))
