"""Unit tests for dataset sources and manifest construction.

The manifest builder is exercised against a tiny synthetic LibriSpeech-shaped
corpus, so no real download is needed.
"""

import numpy as np
import pytest
import soundfile as sf

from asr_robustness.data.manifest import (
    Utterance,
    build_librispeech_manifest,
    read_manifest,
    write_manifest,
)
from asr_robustness.data.sources import DATASETS

SR = 16_000
FAKE_UTTS = {
    "1089-134686-0000": "HELLO WORLD",
    "1089-134686-0001": "THIS IS A TEST",
}


def _make_fake_librispeech(root, split="test-clean"):
    """Create a minimal LibriSpeech directory layout under ``root``."""
    chapter = root / "LibriSpeech" / split / "1089" / "134686"
    chapter.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for utt_id in FAKE_UTTS:
        sig = (0.1 * rng.standard_normal(SR)).astype(np.float32)  # 1 s of audio
        sf.write(str(chapter / f"{utt_id}.flac"), sig, SR)
    lines = [f"{u} {t}" for u, t in FAKE_UTTS.items()]
    (chapter / "1089-134686.trans.txt").write_text("\n".join(lines) + "\n")


def test_dataset_registry_is_sane():
    for name in ("librispeech", "musan", "rirs", "common_voice", "voices"):
        assert name in DATASETS
    for archive in DATASETS["librispeech"].archives.values():
        assert archive.url.startswith("https://")
        assert archive.member_root.startswith("LibriSpeech/")
    assert DATASETS["common_voice"].kind == "hf"
    assert DATASETS["voices"].kind == "manual"


def test_manifest_roundtrip(tmp_path):
    recs = [
        Utterance("u1", "/a.wav", "hello", 1.0, "ds", "test", speaker="s1"),
        Utterance("u2", "/b.wav", "world", 2.0, "ds", "test", accent="us"),
    ]
    path = write_manifest(recs, tmp_path / "m.jsonl")
    back = read_manifest(path)
    assert len(back) == 2
    assert back[0]["utt_id"] == "u1"
    assert back[1]["accent"] == "us"
    assert back[0]["domain"] == "read"


def test_build_librispeech_manifest(tmp_path):
    _make_fake_librispeech(tmp_path)
    manifest = build_librispeech_manifest(tmp_path, "test-clean", out_dir=tmp_path / "manifests")
    records = read_manifest(manifest)

    assert len(records) == len(FAKE_UTTS)
    by_id = {r["utt_id"]: r for r in records}
    assert by_id["1089-134686-0000"]["text"] == "HELLO WORLD"
    for rec in records:
        assert rec["dataset"] == "librispeech"
        assert rec["split"] == "test-clean"
        assert rec["speaker"] == "1089"
        assert rec["duration"] == pytest.approx(1.0, abs=0.05)
        assert rec["audio_path"].endswith(".flac")


def test_build_librispeech_manifest_missing_split(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_librispeech_manifest(tmp_path, "test-clean")
