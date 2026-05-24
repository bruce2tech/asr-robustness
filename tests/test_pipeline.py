"""Unit tests for condition loading and the degradation pipeline."""

from pathlib import Path

import numpy as np
import pytest

from asr_robustness.degrade import effects
from asr_robustness.degrade.banks import NoiseBank, RIRBank
from asr_robustness.degrade.pipeline import DegradationPipeline, load_conditions

CONFIG = Path(__file__).parents[1] / "configs" / "degradation.yaml"


@pytest.fixture
def pipeline(white_noise, synthetic_rir) -> DegradationPipeline:
    noise_bank = NoiseBank(
        {
            "noise/white": white_noise,
            "noise/pink": white_noise * 0.5,
            "speech/talker_a": white_noise * 0.7,
            "speech/talker_b": white_noise * 0.3,
        }
    )
    rir_bank = RIRBank({"room/small": synthetic_rir})
    return DegradationPipeline.from_config(CONFIG, noise_bank=noise_bank, rir_bank=rir_bank)


def test_load_conditions_includes_named_and_swept():
    conditions = load_conditions(CONFIG)
    assert "clean" in conditions
    assert "telephone" in conditions
    # snr_sweep_db and babble_sweep_db each expand to one condition per SNR.
    assert "noise_20db" in conditions
    assert "noise_-20db" in conditions
    assert "babble_20db" in conditions
    assert "babble_-5db" in conditions
    assert "babble_-20db" in conditions
    # Both sweeps expand correctly into matching add_noise / add_babble stages.
    assert conditions["noise_0db"][0]["effect"] == "add_noise"
    assert conditions["babble_0db"][0]["effect"] == "add_babble"


def test_babble_condition_samples_competing_speakers(pipeline, speech_like, sr):
    out, meta = pipeline.apply(speech_like, sr, "babble_-5db", seed=0)
    assert out.shape == speech_like.shape
    stage = meta["stages"][0]
    assert stage["effect"] == "add_babble"
    assert stage["n_talkers"] == 6
    # Babble draws competing speakers from the speech/ category, not noise/.
    assert len(stage["talker_ids"]) == 6
    assert all(tid.startswith("speech/") for tid in stage["talker_ids"])


def test_load_conditions_rejects_empty(tmp_path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("conditions: {}\n")
    with pytest.raises(ValueError):
        load_conditions(empty)


def test_clean_condition_is_identity(pipeline, speech_like, sr):
    out, meta = pipeline.apply(speech_like, sr, "clean", seed=0)
    assert np.array_equal(out, speech_like)
    assert meta["condition"] == "clean"
    assert meta["stages"] == []


def test_pipeline_is_deterministic(pipeline, speech_like, sr):
    a, meta_a = pipeline.apply(speech_like, sr, "noise_10db", seed=7)
    b, meta_b = pipeline.apply(speech_like, sr, "noise_10db", seed=7)
    assert np.array_equal(a, b)
    assert meta_a == meta_b


def test_seed_changes_the_result(pipeline, speech_like, sr):
    a, _ = pipeline.apply(speech_like, sr, "noise_10db", seed=1)
    b, _ = pipeline.apply(speech_like, sr, "noise_10db", seed=2)
    assert not np.array_equal(a, b)


def test_compound_condition_runs_and_preserves_length(pipeline, speech_like, sr):
    out, meta = pipeline.apply(speech_like, sr, "reverb_noise_5db", seed=3)
    assert out.shape == speech_like.shape
    effects_applied = [s["effect"] for s in meta["stages"]]
    assert effects_applied == ["add_reverb", "add_noise"]


def test_telephone_condition_records_codec_stages(pipeline, speech_like, sr):
    out, meta = pipeline.apply(speech_like, sr, "telephone", seed=0)
    assert out.shape == speech_like.shape
    assert [s["effect"] for s in meta["stages"]] == ["narrowband", "mu_law_codec"]


def test_metadata_records_sampled_noise_id(pipeline, speech_like, sr):
    _, meta = pipeline.apply(speech_like, sr, "noise_0db", seed=0)
    noise_stage = meta["stages"][0]
    assert noise_stage["effect"] == "add_noise"
    assert noise_stage["noise_id"].startswith("noise/")


def test_unknown_condition_raises(pipeline, speech_like, sr):
    with pytest.raises(KeyError):
        pipeline.apply(speech_like, sr, "does_not_exist", seed=0)


def test_packet_loss_condition_runs(pipeline, speech_like, sr):
    out, meta = pipeline.apply(speech_like, sr, "packet_loss_30pct", seed=0)
    assert out.shape == speech_like.shape
    assert meta["stages"][0]["effect"] == "packet_loss"
    assert meta["stages"][0]["dropped"] > 0


@pytest.mark.skipif(not effects.ffmpeg_available(), reason="ffmpeg not installed")
def test_codec_condition_runs(pipeline, speech_like, sr):
    out, meta = pipeline.apply(speech_like, sr, "codec_g726", seed=0)
    assert out.shape == speech_like.shape
    assert meta["stages"][0] == {"effect": "apply_codec", "codec": "g726", "bitrate": "8k"}


def test_missing_noise_bank_raises(speech_like, sr):
    bare = DegradationPipeline.from_config(CONFIG)  # no banks
    with pytest.raises(ValueError, match="noise bank"):
        bare.apply(speech_like, sr, "noise_10db", seed=0)
