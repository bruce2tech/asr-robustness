"""Dataset + collator for wav2vec 2.0 CTC fine-tuning, with online MCT aug.

Parallels :mod:`asr_robustness.train.dataset` (Whisper) but speaks wav2vec 2.0's
input/output convention:

* **inputs** are raw waveforms (``input_values``), not mel features
* **labels** are character-level CTC token IDs (no decoder prompt, no BOS/EOS)
* the collator pads inputs and labels separately and masks pad positions to
  ``-100`` so :class:`transformers.Wav2Vec2ForCTC`'s built-in CTC loss ignores
  them

The augmentation regime is identical to Whisper MCT-FT: when ``pipeline`` and
``conditions`` are provided, every ``__getitem__`` samples one named condition
and applies it to the clean audio with a fresh seed before the processor
extracts features. Sharing the :class:`DegradationPipeline` across both
training drivers means the *same* degradations are used in training across
architectures -- a key precondition for honest cross-architecture comparison.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from asr_robustness.audio import TARGET_SR, load_audio
from asr_robustness.data.manifest import read_manifest
from asr_robustness.degrade.pipeline import DegradationPipeline


class Wav2Vec2FTDataset(Dataset):
    """A manifest-backed wav2vec 2.0 fine-tuning dataset.

    Parameters mirror :class:`asr_robustness.train.dataset.WhisperFTDataset` so
    the same training-time choices (manifest, augmentation conditions, limit)
    transfer across the two drivers with no surprises.
    """

    def __init__(
        self,
        manifest_path: str,
        processor: Any,
        pipeline: DegradationPipeline | None = None,
        conditions: list[str] | None = None,
        limit: int | None = None,
    ):
        if pipeline is not None and not conditions:
            raise ValueError("conditions must be non-empty when a pipeline is supplied")
        self.records = read_manifest(manifest_path)
        if limit:
            self.records = self.records[:limit]
        self.processor = processor
        self.pipeline = pipeline
        self.conditions = list(conditions) if conditions else None

    def __len__(self) -> int:
        return len(self.records)

    def _audio_for(self, rec: dict) -> np.ndarray:
        audio, _ = load_audio(rec["audio_path"], target_sr=TARGET_SR)
        if self.pipeline is not None:
            condition = random.choice(self.conditions)  # type: ignore[arg-type]
            # Independent random seed per call -- augmentation is intentionally
            # stochastic so each epoch sees a different acoustic realization.
            seed = random.randrange(2**31)
            audio, _ = self.pipeline.apply(audio, TARGET_SR, condition, seed)
        return audio.astype(np.float32)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        audio = self._audio_for(rec)
        # wav2vec2 processor returns input_values (raw normalized waveform).
        input_values = self.processor(
            audio, sampling_rate=TARGET_SR
        ).input_values[0]
        # wav2vec2-base-960h tokenizer is character-level over the LibriSpeech
        # vocab; upper-cased text matches the trained tokenizer.
        labels = self.processor.tokenizer(rec["text"].upper()).input_ids
        return {"input_values": input_values, "labels": labels}


@dataclass
class DataCollatorWav2Vec2FT:
    """Pad and stack a batch of :class:`Wav2Vec2FTDataset` items for ``Trainer``.

    Inputs and labels are padded independently. Label pad positions are
    rewritten to ``-100`` so ``Wav2Vec2ForCTC``'s CTC loss ignores them
    (matches the convention used in the HuggingFace fine-tuning recipe).
    """

    processor: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        # Explicitly request attention_mask -- wav2vec2's feature extractor
        # doesn't return one by default, but it's needed so the model can
        # attend over only the real (non-padded) audio frames during training.
        batch = self.processor.pad(
            input_features, padding=True, return_tensors="pt", return_attention_mask=True
        )
        labels_batch = self.processor.pad(
            labels=label_features, padding=True, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch
