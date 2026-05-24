"""Dataset + collator for Whisper fine-tuning, with online multi-condition aug.

The training loop sees one of two regimes depending on whether ``pipeline`` is
supplied:

* **Clean-FT** (no pipeline): the dataset returns Whisper features computed
  from the original clean audio. This is the ablation baseline -- a fine-tune
  that learns nothing about degraded acoustics.
* **Multi-condition training (MCT)** (pipeline + condition list given): for
  each ``__getitem__``, the dataset uniformly samples one named degradation
  condition and applies it to the clean signal with a fresh random seed
  before feature extraction. The model sees a different acoustic realization
  of each utterance on every epoch -- the standard recipe for noise-robust
  ASR fine-tuning.

The collator follows the canonical HuggingFace Whisper recipe: feature tensors
are stacked, label sequences are padded with the tokenizer's pad token, and
those pad positions are then rewritten to ``-100`` so the cross-entropy loss
ignores them. If the tokenizer prepended a redundant BOS that the model will
add back during the forward pass, the collator strips it.
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


class WhisperFTDataset(Dataset):
    """A manifest-backed Whisper fine-tuning dataset.

    Parameters
    ----------
    manifest_path:
        Path to a ``.jsonl`` manifest (see :mod:`asr_robustness.data.manifest`).
    processor:
        A ``WhisperProcessor``. Must already have prefix tokens set for the
        target language/task -- :func:`prepare_processor` handles this.
    pipeline:
        Optional :class:`DegradationPipeline`. When supplied alongside
        ``conditions``, every ``__getitem__`` samples one condition and
        degrades the audio before feature extraction.
    conditions:
        Names of conditions in ``pipeline`` that are eligible for sampling.
        Required iff ``pipeline`` is given. Repeat a name to weight it higher.
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
            # Independent random seed per call -- training augmentation is
            # intentionally stochastic, not reproducible per-utterance.
            seed = random.randrange(2**31)
            audio, _ = self.pipeline.apply(audio, TARGET_SR, condition, seed)
        return audio.astype(np.float32)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        audio = self._audio_for(rec)
        features = self.processor.feature_extractor(
            audio, sampling_rate=TARGET_SR
        ).input_features[0]
        labels = self.processor.tokenizer(rec["text"]).input_ids
        return {"input_features": features, "labels": labels}


def prepare_processor(processor: Any, language: str = "en", task: str = "transcribe") -> Any:
    """Configure a ``WhisperProcessor`` for fine-tuning on one language/task."""
    processor.tokenizer.set_prefix_tokens(language=language, task=task)
    return processor


@dataclass
class DataCollatorWhisperFT:
    """Pad/stack a batch of ``WhisperFTDataset`` items for ``Seq2SeqTrainer``.

    ``decoder_start_token_id`` is read from the model's config; pass it in so
    the collator can drop a redundant prepended BOS that the model itself will
    insert again at decode time.
    """

    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_batch = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features],
            return_tensors="pt",
        )
        labels_batch = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features],
            return_tensors="pt",
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # Strip a redundant prepended BOS: the model re-adds it during forward.
        if (labels[:, 0] == self.decoder_start_token_id).all().item():
            labels = labels[:, 1:]
        input_batch["labels"] = labels
        return input_batch
