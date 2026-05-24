"""Unit tests for the degradation primitives."""

import numpy as np
import pytest

from asr_robustness.audio import rms
from asr_robustness.degrade import effects
from tests.conftest import tone


@pytest.mark.parametrize("snr_db", [20, 10, 5, 0, -5])
def test_add_noise_hits_target_snr(speech_like, white_noise, snr_db):
    rng = np.random.default_rng(42)
    mixed, info = effects.add_noise(speech_like, white_noise, snr_db, rng)
    realized = effects.measure_snr(speech_like, mixed)
    assert abs(realized - snr_db) < 0.1, f"target {snr_db} dB, got {realized:.3f} dB"
    assert info["snr_db"] == snr_db
    assert mixed.shape == speech_like.shape


def test_add_noise_tiles_short_noise(speech_like):
    short = speech_like[: len(speech_like) // 3]  # shorter than target
    mixed, _ = effects.add_noise(speech_like, short, snr_db=10)
    assert mixed.shape == speech_like.shape


def test_add_noise_empty_noise_raises(speech_like):
    with pytest.raises(ValueError):
        effects.add_noise(speech_like, np.array([], dtype=np.float32), snr_db=10)


@pytest.mark.parametrize("snr_db", [5, 0, -5])
def test_add_babble_hits_target_snr(speech_like, white_noise, snr_db):
    talkers = [white_noise * g for g in (0.3, 0.5, 0.8, 0.4, 0.6, 0.7)]
    mixed, info = effects.add_babble(speech_like, talkers, snr_db, np.random.default_rng(0))
    assert info["effect"] == "add_babble"
    assert info["n_talkers"] == 6
    assert mixed.shape == speech_like.shape
    assert effects.measure_snr(speech_like, mixed) == pytest.approx(snr_db, abs=0.1)


def test_add_babble_requires_at_least_one_talker(speech_like):
    with pytest.raises(ValueError):
        effects.add_babble(speech_like, [], snr_db=0)


def test_reverb_preserves_length(speech_like, synthetic_rir):
    wet, info = effects.add_reverb(speech_like, synthetic_rir)
    assert wet.shape == speech_like.shape
    assert info["rir_len"] == len(synthetic_rir)


def test_reverb_preserves_energy(speech_like, synthetic_rir):
    wet, _ = effects.add_reverb(speech_like, synthetic_rir, preserve_energy=True)
    assert rms(wet) == pytest.approx(rms(speech_like), rel=1e-4)


def test_reverb_actually_smears_signal(speech_like, synthetic_rir):
    wet, _ = effects.add_reverb(speech_like, synthetic_rir)
    # A reverberant signal must differ from the dry one.
    assert np.max(np.abs(wet - speech_like)) > 1e-3


def test_mu_law_roundtrip_is_lossy_but_faithful(speech_like):
    y, info = effects.mu_law_codec(speech_like)
    assert info["mu"] == 255
    assert y.shape == speech_like.shape
    # Highly correlated with the original...
    corr = np.corrcoef(speech_like, y)[0, 1]
    assert corr > 0.999
    # ...but not identical -- quantization distortion was injected.
    assert np.max(np.abs(y - speech_like)) > 1e-4


def test_narrowband_attenuates_below_band(sr):
    low = tone(100)  # well below the 300 Hz lower edge
    y, _ = effects.narrowband(low, sr)
    assert rms(y) / rms(low) < 0.3


def test_narrowband_passes_in_band(sr):
    mid = tone(1000)  # squarely inside the 300-3400 Hz band
    y, _ = effects.narrowband(mid, sr)
    assert rms(y) / rms(mid) > 0.7


def test_clip_reduces_peak(speech_like):
    y, info = effects.clip_signal(speech_like, percentile=90.0)
    assert np.max(np.abs(y)) <= info["threshold"] + 1e-6
    assert np.max(np.abs(y)) < np.max(np.abs(speech_like))


def test_gain_scales_rms(speech_like):
    y, info = effects.gain(speech_like, gain_db=6.0)
    assert rms(y) / rms(speech_like) == pytest.approx(10 ** (6.0 / 20.0), rel=1e-4)
    assert info["gain_db"] == 6.0


@pytest.mark.parametrize("loss_rate", [0.2, 0.5])
def test_packet_loss_drops_packets_and_preserves_length(speech_like, sr, loss_rate):
    out, info = effects.packet_loss(
        speech_like, sr, loss_rate=loss_rate, rng=np.random.default_rng(0)
    )
    assert out.shape == speech_like.shape
    assert info["effect"] == "packet_loss"
    assert 0 < info["dropped"] <= info["n_packets"]
    assert np.any(out == 0.0)  # dropped packets are silenced


def test_packet_loss_zero_rate_is_identity(speech_like, sr):
    out, info = effects.packet_loss(
        speech_like, sr, loss_rate=0.0, rng=np.random.default_rng(0)
    )
    assert info["dropped"] == 0
    assert np.array_equal(out, speech_like)


@pytest.mark.skipif(not effects.ffmpeg_available(), reason="ffmpeg not installed")
@pytest.mark.parametrize("codec", ["g726", "g722", "opus"])
def test_apply_codec_roundtrip(speech_like, sr, codec):
    out, info = effects.apply_codec(speech_like, sr, codec=codec, bitrate="6k")
    assert out.shape == speech_like.shape  # length is fitted back to original
    assert info["codec"] == codec
    assert not np.allclose(out, speech_like)  # lossy encode changed the signal


def test_apply_codec_unknown_raises(speech_like, sr):
    with pytest.raises(ValueError):
        effects.apply_codec(speech_like, sr, codec="not-a-codec")


def test_measure_snr_roundtrip(speech_like, white_noise):
    mixed, _ = effects.add_noise(speech_like, white_noise, snr_db=7.5)
    assert effects.measure_snr(speech_like, mixed) == pytest.approx(7.5, abs=0.1)
