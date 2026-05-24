"""Tests for the Phase 6 fine-tuning dataset and collator.

We deliberately stub the ``WhisperProcessor`` rather than loading the real one.
The dataset's contract with the processor is narrow -- it calls
``feature_extractor(...).input_features[0]`` and ``tokenizer(...).input_ids`` --
and a fake captures that contract exactly while keeping the suite offline,
CUDA-free, and fast.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from asr_robustness.degrade.banks import NoiseBank, RIRBank
from asr_robustness.degrade.pipeline import DegradationPipeline
from asr_robustness.train.dataset import (
    DataCollatorWhisperFT,
    WhisperFTDataset,
)

CONFIG = Path(__file__).parents[1] / "configs" / "degradation.yaml"


class _Output:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeFeatureExtractor:
    """Returns a (1, n_mels, n_frames) input_features tensor, like Whisper's real one."""

    n_mels = 80
    n_frames = 3000

    def __init__(self):
        self.last_audio: np.ndarray | None = None

    def __call__(self, audio, sampling_rate):
        self.last_audio = np.asarray(audio, dtype=np.float32)
        feat = np.zeros((1, self.n_mels, self.n_frames), dtype=np.float32)
        # Encode a summary of the input audio into the feature tensor so tests
        # can detect whether the dataset passed clean vs degraded audio in.
        feat[0, 0, 0] = float(np.mean(np.abs(self.last_audio)))
        return _Output(input_features=feat)

    def pad(self, items, return_tensors="pt"):
        arr = np.stack([np.asarray(it["input_features"], dtype=np.float32) for it in items])
        return {"input_features": torch.from_numpy(arr)}


class FakeTokenizer:
    """Tokenizes by mapping each character to its ASCII code -- enough for the contract."""

    pad_token_id = 0

    def __call__(self, text):
        ids = [ord(c) for c in text]
        return _Output(input_ids=ids)

    def set_prefix_tokens(self, language, task):  # noqa: ARG002 -- contract method
        pass

    def pad(self, items, return_tensors="pt"):
        max_len = max(len(it["input_ids"]) for it in items)
        ids = torch.zeros((len(items), max_len), dtype=torch.long)
        mask = torch.zeros((len(items), max_len), dtype=torch.long)
        for i, it in enumerate(items):
            n = len(it["input_ids"])
            ids[i, :n] = torch.tensor(it["input_ids"], dtype=torch.long)
            mask[i, :n] = 1
        out = {"input_ids": ids, "attention_mask": mask}
        # Trainer-style: the dict needs `.attention_mask` attribute access too.
        return _DictWithAttrs(out)


class _DictWithAttrs(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class FakeProcessor:
    def __init__(self):
        self.feature_extractor = FakeFeatureExtractor()
        self.tokenizer = FakeTokenizer()


@pytest.fixture
def tiny_manifest(tmp_path: Path) -> Path:
    """Write two synthetic 1 s utterances + a 2-row JSONL manifest, return its path."""
    sr = 16_000
    t = np.arange(sr) / sr
    audio_a = (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    audio_b = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path_a = tmp_path / "a.wav"
    path_b = tmp_path / "b.wav"
    sf.write(str(path_a), audio_a, sr)
    sf.write(str(path_b), audio_b, sr)

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"utt_id": "a", "audio_path": str(path_a), "text": "hello",
                    "duration": 1.0, "dataset": "synthetic", "split": "test"}) + "\n"
        + json.dumps({"utt_id": "b", "audio_path": str(path_b), "text": "world",
                      "duration": 1.0, "dataset": "synthetic", "split": "test"}) + "\n"
    )
    return manifest


def test_dataset_length_and_keys(tiny_manifest):
    ds = WhisperFTDataset(str(tiny_manifest), FakeProcessor())
    assert len(ds) == 2
    item = ds[0]
    assert set(item.keys()) == {"input_features", "labels"}
    assert item["input_features"].shape == (80, 3000)
    assert item["labels"] == [ord(c) for c in "hello"]


def test_clean_dataset_passes_unmodified_audio_to_extractor(tiny_manifest):
    """No pipeline -> the feature extractor sees the *clean* audio."""
    processor = FakeProcessor()
    ds = WhisperFTDataset(str(tiny_manifest), processor)
    ds[0]
    seen = processor.feature_extractor.last_audio
    # Should look like the original 220 Hz sine: bounded amplitude, ~16k samples.
    assert seen.shape == (16_000,)
    assert 0.2 < float(np.max(np.abs(seen))) <= 0.5


def test_pipeline_without_conditions_raises(tiny_manifest, white_noise, synthetic_rir):
    noise_bank = NoiseBank({"noise/white": white_noise})
    pipeline = DegradationPipeline.from_config(CONFIG, noise_bank=noise_bank)
    with pytest.raises(ValueError, match="conditions"):
        WhisperFTDataset(str(tiny_manifest), FakeProcessor(), pipeline=pipeline, conditions=[])


def test_mct_dataset_actually_degrades_audio(tiny_manifest, white_noise, synthetic_rir):
    """With a pipeline + condition list, the extractor sees *degraded* audio."""
    noise_bank = NoiseBank({"noise/white": white_noise})
    rir_bank = RIRBank({"room/synthetic": synthetic_rir})
    pipeline = DegradationPipeline.from_config(CONFIG, noise_bank=noise_bank, rir_bank=rir_bank)

    clean_processor = FakeProcessor()
    clean_ds = WhisperFTDataset(str(tiny_manifest), clean_processor)
    clean_ds[0]
    clean_audio = clean_processor.feature_extractor.last_audio.copy()

    mct_processor = FakeProcessor()
    # Use a deep-noise condition so audio is guaranteed to differ from clean.
    mct_ds = WhisperFTDataset(
        str(tiny_manifest), mct_processor,
        pipeline=pipeline, conditions=["noise_-10db"],
    )
    mct_ds[0]
    degraded = mct_processor.feature_extractor.last_audio

    assert degraded.shape == clean_audio.shape
    assert not np.array_equal(degraded, clean_audio)


def test_collator_pads_labels_and_masks_with_neg100():
    processor = FakeProcessor()
    collator = DataCollatorWhisperFT(processor=processor, decoder_start_token_id=-999)
    features = [
        {"input_features": np.zeros((80, 3000), dtype=np.float32), "labels": [10, 11, 12]},
        {"input_features": np.zeros((80, 3000), dtype=np.float32), "labels": [20, 21]},
    ]
    batch = collator(features)
    assert batch["input_features"].shape == (2, 80, 3000)
    # Shorter label row gets padded then masked with -100 for the loss.
    labels = batch["labels"]
    assert labels.shape == (2, 3)
    assert labels[1, 2].item() == -100
    assert labels[0].tolist() == [10, 11, 12]


def test_collator_strips_redundant_bos():
    processor = FakeProcessor()
    bos = 99
    collator = DataCollatorWhisperFT(processor=processor, decoder_start_token_id=bos)
    features = [
        {"input_features": np.zeros((80, 3000), dtype=np.float32), "labels": [bos, 10, 11]},
        {"input_features": np.zeros((80, 3000), dtype=np.float32), "labels": [bos, 20, 21]},
    ]
    batch = collator(features)
    # All sequences started with the BOS -> the collator should drop it.
    assert batch["labels"].shape == (2, 2)
    assert batch["labels"][0].tolist() == [10, 11]
