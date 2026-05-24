"""ESPnet adapter for pretrained models from the ESPnet model zoo.

ESPnet is the reference toolkit named in the target role. Here it contributes a
**pretrained** model as a benchmark peer (a model trained properly on full
LibriSpeech, so the comparison is fair). Hands-on ESPnet *training* happens
separately in Phase 6 via fine-tuning.

espnet / espnet_model_zoo are imported at module load, so this module is
imported lazily (see ``asr_robustness.models.ensure_loaded``).
"""

from __future__ import annotations

import numpy as np

from asr_robustness.audio import TARGET_SR, resample
from asr_robustness.models.base import ASRModel
from asr_robustness.models.registry import register


@register("espnet")
class ESPnetModel(ASRModel):
    """A pretrained ESPnet ASR model served via ``espnet2`` Speech2Text.

    ``model_id`` is an ESPnet model-zoo tag, e.g.
    ``espnet/simpleoier_librispeech_asr_train_asr_conformer7_*``.
    """

    def __init__(
        self,
        model_id: str,
        name: str | None = None,
        device: str | None = None,
    ):
        # Imported here (not at module top) so a missing espnet install only
        # fails when this model is actually used.
        from espnet2.bin.asr_inference import Speech2Text
        from espnet_model_zoo.downloader import ModelDownloader

        self.name = name or model_id.split("/")[-1]
        self.model_id = model_id

        downloaded = ModelDownloader().download_and_unpack(model_id)
        self.speech2text = Speech2Text(
            **downloaded,
            device=device or "cpu",  # MPS support in ESPnet is incomplete
            nbest=1,
        )

    def transcribe(self, audio: np.ndarray, sr: int) -> str:
        if sr != TARGET_SR:
            audio = resample(audio, sr, TARGET_SR)
        nbest = self.speech2text(audio.astype(np.float32))
        if not nbest:
            return ""
        text = nbest[0][0]  # (text, tokens, token_ids, hyp) -> first field
        return text.strip()
