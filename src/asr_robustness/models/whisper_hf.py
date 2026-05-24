"""Whisper adapter (HuggingFace Transformers).

Whisper is an attentional encoder-decoder model. Its generative decoder is the
reason it is central to this project's hallucination analysis: on low-SNR audio
it tends to emit fluent but invented text rather than degrade gracefully.

torch / transformers are imported at module load, so this module is imported
lazily (see ``asr_robustness.models.ensure_loaded``).
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from asr_robustness.audio import TARGET_SR, resample
from asr_robustness.models.base import ASRModel
from asr_robustness.models.registry import register


def _pick_device(requested: str | None) -> str:
    """Choose a compute device: explicit request, else MPS (Apple GPU), else CPU."""
    if requested:
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@register("whisper")
class WhisperModel(ASRModel):
    """A Whisper checkpoint served through Transformers."""

    def __init__(
        self,
        model_id: str = "openai/whisper-base",
        name: str | None = None,
        device: str | None = None,
        language: str = "en",
    ):
        self.name = name or model_id.split("/")[-1]
        self.model_id = model_id
        self.language = language
        self.device = _pick_device(device)
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def transcribe(self, audio: np.ndarray, sr: int) -> str:
        if sr != TARGET_SR:
            audio = resample(audio, sr, TARGET_SR)
        features = self.processor(
            audio, sampling_rate=TARGET_SR, return_tensors="pt"
        ).input_features.to(self.device)
        tokens = self.model.generate(
            features, language=self.language, task="transcribe"
        )
        return self.processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
