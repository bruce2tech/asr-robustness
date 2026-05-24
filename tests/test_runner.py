"""Unit tests for the decode runner (uses the stub model -- no torch needed)."""

import numpy as np
import soundfile as sf

from asr_robustness.degrade.banks import NoiseBank
from asr_robustness.degrade.pipeline import DegradationPipeline
from asr_robustness.eval.runner import read_results, run_decode, write_results
from asr_robustness.models.stub import StubModel


def _tiny_manifest(tmp_path):
    """Two real (synthetic) audio files plus their manifest records."""
    records = []
    for i, text in enumerate(["hello world", "the quick brown fox"]):
        path = tmp_path / f"u{i}.wav"
        sig = (0.1 * np.random.default_rng(i).standard_normal(16_000)).astype(np.float32)
        sf.write(str(path), sig, 16_000)
        records.append(
            {
                "utt_id": f"u{i}",
                "audio_path": str(path),
                "text": text,
                "dataset": "synthetic",
                "split": "test",
                "speaker": f"spk{i}",
                "accent": None,
                "domain": "read",
            }
        )
    return records


def _pipeline():
    noise = NoiseBank({"noise/white": np.random.default_rng(0).standard_normal(16_000)})
    conditions = {"clean": [], "noise_0db": [{"effect": "add_noise", "snr_db": 0}]}
    return DegradationPipeline(conditions, noise_bank=noise)


def test_run_decode_produces_one_row_per_pair(tmp_path):
    records = _tiny_manifest(tmp_path)
    model = StubModel(transcript="hello world")
    results = run_decode(records, model, _pipeline(), ["clean", "noise_0db"], progress=False)

    assert len(results) == 2 * 2  # 2 utterances x 2 conditions
    assert {r["condition"] for r in results} == {"clean", "noise_0db"}
    assert {r["model"] for r in results} == {"stub"}


def test_run_decode_scores_and_carries_metadata(tmp_path):
    records = _tiny_manifest(tmp_path)
    model = StubModel(transcript="hello world")
    results = run_decode(records, model, _pipeline(), ["clean"], progress=False)

    by_utt = {r["utt_id"]: r for r in results}
    # u0 reference is exactly the stub output -> zero WER.
    assert by_utt["u0"]["wer"] == 0.0
    # u1 reference differs -> non-zero WER.
    assert by_utt["u1"]["wer"] > 0.0
    # Carried manifest fields + degradation metadata are present.
    assert by_utt["u0"]["speaker"] == "spk0"
    assert by_utt["u0"]["domain"] == "read"
    assert by_utt["u0"]["degradation"]["condition"] == "clean"
    assert "seed" in by_utt["u0"]["degradation"]


def test_run_decode_seed_is_stable_per_utterance(tmp_path):
    records = _tiny_manifest(tmp_path)
    model = StubModel(transcript="x")
    results = run_decode(records, model, _pipeline(), ["clean"], seed_base=100, progress=False)
    seeds = {r["utt_id"]: r["seed"] for r in results}
    assert seeds == {"u0": 100, "u1": 101}


def test_run_decode_limit(tmp_path):
    records = _tiny_manifest(tmp_path)
    model = StubModel(transcript="x")
    results = run_decode(records, model, _pipeline(), ["clean"], limit=1, progress=False)
    assert len(results) == 1


def test_results_jsonl_roundtrip(tmp_path):
    records = _tiny_manifest(tmp_path)
    model = StubModel(transcript="hello world")
    results = run_decode(records, model, _pipeline(), ["clean", "noise_0db"], progress=False)
    path = write_results(results, tmp_path / "results.jsonl")
    assert read_results(path) == results
