"""Unit tests for the audio audition tool."""

from pathlib import Path

import numpy as np
import soundfile as sf

from asr_robustness.degrade.audition import audition

CONFIG = Path(__file__).parents[1] / "configs" / "degradation.yaml"


def _audio_file(tmp_path) -> Path:
    path = tmp_path / "utt.wav"
    sig = (0.2 * np.random.default_rng(0).standard_normal(16_000)).astype(np.float32)
    sf.write(str(path), sig, 16_000)
    return path


def test_audition_writes_one_wav_per_condition(tmp_path):
    written = audition(
        str(_audio_file(tmp_path)),
        conditions=["clean", "telephone"],  # neither needs a bank
        out_dir=tmp_path / "demo",
        degradation_config=str(CONFIG),
        noise_dir=None,
        rir_dir=None,
    )
    assert len(written) == 2
    assert all(p.exists() and p.suffix == ".wav" for p in written)
    assert {p.name.split("__")[1] for p in written} == {"clean.wav", "telephone.wav"}


def test_audition_skips_conditions_with_missing_bank(tmp_path):
    written = audition(
        str(_audio_file(tmp_path)),
        conditions=["clean", "noise_0db"],  # noise_0db needs a noise bank
        out_dir=tmp_path / "demo",
        degradation_config=str(CONFIG),
        noise_dir=None,
        rir_dir=None,
    )
    assert len(written) == 1
    assert written[0].name.endswith("__clean.wav")


def test_audition_uses_noise_bank_when_available(tmp_path):
    # MUSAN-style layout: the bank root holds a `noise/` category subdirectory,
    # so clip IDs are `noise/...` and the `noise_type: noise` filter matches.
    musan = tmp_path / "musan"
    noise_cat = musan / "noise"
    noise_cat.mkdir(parents=True)
    sf.write(
        str(noise_cat / "n.wav"),
        np.random.default_rng(1).standard_normal(16_000).astype(np.float32),
        16_000,
    )
    written = audition(
        str(_audio_file(tmp_path)),
        conditions=["noise_0db"],
        out_dir=tmp_path / "demo",
        degradation_config=str(CONFIG),
        noise_dir=str(musan),
        rir_dir=None,
    )
    assert len(written) == 1
    assert "noise_0db" in written[0].name
