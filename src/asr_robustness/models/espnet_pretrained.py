"""ESPnet adapter for pretrained models from the ESPnet model zoo.

ESPnet is the reference toolkit named in the target role. It contributes a
**pretrained** model as a benchmark peer (a model trained properly on full
LibriSpeech, so the comparison is fair) and, via Phase 6's fine-tuning,
locally-trained checkpoints loaded back through the same adapter for the
symmetric ablation.

espnet / espnet_model_zoo are imported lazily (see ``asr_robustness.models.ensure_loaded``).
"""

from __future__ import annotations

import os

import numpy as np

from asr_robustness.audio import TARGET_SR, resample
from asr_robustness.models.base import ASRModel
from asr_robustness.models.registry import register


@register("espnet")
class ESPnetModel(ASRModel):
    """A pretrained or fine-tuned ESPnet ASR model served via ``espnet2`` Speech2Text.

    Two ways to load:

    1. **Pretrained from model zoo**: ``model_id`` is a hub tag like
       ``asapp/e_branchformer_librispeech``. ``ModelDownloader`` fetches (or
       reuses cached) bundle artifacts.
    2. **Local FT checkpoint**: ``model_id`` is a local directory that
       contains a ``valid.acc.best.pth`` symlink (or file). The directory's
       config files reference pod-side BPE/tokenizer paths that won't exist
       on the Mac, so we additionally require ``base_model_id`` (the hub tag
       of the bundle the FT was initialized from). We pull the bundle's
       config + tokenizer through ``ModelDownloader`` (cached) and swap in
       our local FT weights as ``asr_model_file``. The fine-tune doesn't
       change architecture or tokenizer -- only weights -- so this is
       exactly correct.
    """

    def __init__(
        self,
        model_id: str,
        name: str | None = None,
        device: str | None = None,
        beam_size: int = 5,
        base_model_id: str | None = None,
    ):
        # Imported here (not at module top) so a missing espnet install only
        # fails when this model is actually used.
        from espnet2.bin.asr_inference import Speech2Text
        from espnet_model_zoo.downloader import ModelDownloader

        self.name = name or os.path.basename(model_id.rstrip("/"))
        self.model_id = model_id

        if os.path.isdir(model_id):
            if not base_model_id:
                raise ValueError(
                    f"model_id={model_id!r} is a local directory; you must "
                    f"also provide base_model_id (e.g. "
                    f"'asapp/e_branchformer_librispeech') so the adapter can "
                    f"resolve the matching architecture and tokenizer."
                )
            downloaded = dict(ModelDownloader().download_and_unpack(base_model_id))
            local_ckpt = os.path.join(model_id, "valid.acc.best.pth")
            if not os.path.exists(local_ckpt):
                raise FileNotFoundError(
                    f"expected 'valid.acc.best.pth' inside {model_id!r}; "
                    f"got contents: {os.listdir(model_id)}"
                )
            downloaded["asr_model_file"] = local_ckpt
        else:
            downloaded = ModelDownloader().download_and_unpack(model_id)

        # beam_size=5 (vs the ESPnet default of 20) cuts decode time ~3-4x with
        # negligible WER impact on this scale of model; the default is too
        # generous for the model sizes we're benchmarking.
        self.speech2text = Speech2Text(
            **downloaded,
            device=device or "cpu",  # MPS support in ESPnet is incomplete
            nbest=1,
            beam_size=beam_size,
        )

    def transcribe(self, audio: np.ndarray, sr: int) -> str:
        if sr != TARGET_SR:
            audio = resample(audio, sr, TARGET_SR)
        nbest = self.speech2text(audio.astype(np.float32))
        if not nbest:
            return ""
        text = nbest[0][0]  # (text, tokens, token_ids, hyp) -> first field
        return text.strip()
