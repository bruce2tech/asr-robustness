"""Registry of dataset sources used by the project.

Each :class:`Dataset` is one of three kinds:

* ``archive``  -- a set of directly-downloadable archives (handled by
  ``download.py``): LibriSpeech, MUSAN, the OpenSLR RIR corpus.
* ``hf``       -- loaded through the HuggingFace ``datasets`` library
  (Common Voice; needs a HF login and acceptance of the dataset terms).
* ``manual``   -- requires a one-off manual download / access request
  (VOiCES); the local path is then set in ``configs/data.yaml``.

Why these corpora:

* **LibriSpeech** -- clean read speech; the controlled base that synthetic
  degradation is applied to, and the source of fine-tuning audio.
* **MUSAN**      -- diverse noise (music / speech / ambient) for the noise bank.
* **OpenSLR RIRs** -- real + simulated room impulse responses for reverberation.
* **Common Voice** -- crowd-sourced speech with **accent labels**, for the
  accent breakdown.
* **VOiCES**     -- speech recorded in genuinely noisy, reverberant far-field
  rooms; the real-world holdout that validates the synthetic findings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

OPENSLR = "https://www.openslr.org/resources"


@dataclass(frozen=True)
class Archive:
    """A single downloadable archive belonging to a dataset."""

    name: str
    url: str
    member_root: str  # path (relative to data root) created on extraction; used
    md5: str | None = None  # for idempotency and (optional) integrity checks


@dataclass(frozen=True)
class Dataset:
    """A named corpus and how to obtain it."""

    name: str
    kind: str  # "archive" | "hf" | "manual"
    archives: dict[str, Archive] = field(default_factory=dict)
    note: str = ""


_LS_SUBSETS = [
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
    "train-clean-100",
    "train-clean-360",
]

LIBRISPEECH = Dataset(
    name="librispeech",
    kind="archive",
    archives={
        s: Archive(name=s, url=f"{OPENSLR}/12/{s}.tar.gz", member_root=f"LibriSpeech/{s}")
        for s in _LS_SUBSETS
    },
)

MUSAN = Dataset(
    name="musan",
    kind="archive",
    archives={"musan": Archive("musan", f"{OPENSLR}/17/musan.tar.gz", "musan")},
)

RIRS = Dataset(
    name="rirs",
    kind="archive",
    archives={"rirs": Archive("rirs", f"{OPENSLR}/28/rirs_noises.zip", "RIRS_NOISES")},
)

COMMON_VOICE = Dataset(
    name="common_voice",
    kind="hf",
    note=(
        "Load via HuggingFace datasets: 'mozilla-foundation/common_voice_17_0'. "
        "Requires `huggingface-cli login` and accepting the dataset terms on the "
        "Hub. Provides per-utterance accent labels used by the accent breakdown."
    ),
)

VOICES = Dataset(
    name="voices",
    kind="manual",
    note=(
        "VOiCES corpus -- real far-field noisy/reverberant speech. Obtain from "
        "https://iqtlabs.github.io/voices/ (or the 'lab41openaudiocorpus' open "
        "S3 bucket), then set its extracted path under `datasets.voices.path` in "
        "configs/data.yaml. Used as the real-world holdout in Phase 7."
    ),
)

DATASETS: dict[str, Dataset] = {
    d.name: d for d in (LIBRISPEECH, MUSAN, RIRS, COMMON_VOICE, VOICES)
}

# Smallest LibriSpeech subset -- used by `download --minimal` for smoke tests.
MINIMAL_LIBRISPEECH_SUBSETS = ["dev-clean"]
