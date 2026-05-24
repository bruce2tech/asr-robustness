"""wav2vec 2.0 adapter (HuggingFace Transformers).

wav2vec 2.0 is a CTC model: a self-supervised encoder with a frame-wise
character/token classifier on top. Unlike Whisper it has **no generative
decoder**, so it cannot hallucinate fluent text -- under heavy noise it tends to
drop or garble tokens instead. Contrasting the two failure modes is one of the
project's findings, which is why both architectures are benchmarked.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from asr_robustness.audio import TARGET_SR, resample
from asr_robustness.models.base import ASRModel
from asr_robustness.models.registry import register


def _pick_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@register("wav2vec2")
class Wav2Vec2Model(ASRModel):
    """A wav2vec 2.0 CTC checkpoint served through Transformers."""

    def __init__(
        self,
        model_id: str = "facebook/wav2vec2-base-960h",
        name: str | None = None,
        device: str | None = None,
    ):
        self.name = name or model_id.split("/")[-1]
        self.model_id = model_id
        self.device = _pick_device(device)
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def transcribe(self, audio: np.ndarray, sr: int) -> str:
        if sr != TARGET_SR:
            audio = resample(audio, sr, TARGET_SR)
        inputs = self.processor(
            audio, sampling_rate=TARGET_SR, return_tensors="pt"
        ).input_values.to(self.device)
        logits = self.model(inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)  # greedy CTC decoding
        return self.processor.batch_decode(predicted_ids)[0].strip()
