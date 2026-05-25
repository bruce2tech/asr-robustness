"""Pre-render (and optionally degrade) training audio to disk for ESPnet FT.

ESPnet's training pipeline ("native recipe", Path II in the project plan) wants
audio on disk plus Kaldi-style metadata files; it does not have a HuggingFace-
style in-memory online augmentation hook. So for the multi-condition ESPnet
fine-tune we pre-render one degraded WAV per training utterance, sampling a
condition with a deterministic seed so the rendering is reproducible.

Output layout (under ``--out-dir``):

    <out_dir>/
      wav/<utt_id>.wav    # 16 kHz mono float32 WAV, possibly degraded
      wav.scp             # "<utt_id> /abs/path/to/wav" lines
      text                # "<utt_id> <reference text>" lines
      utt2spk             # "<utt_id> <speaker_id>" lines
      spk2utt             # "<speaker_id> <utt_id1> <utt_id2> ..." lines
      manifest.jsonl      # parallel asr_robustness manifest of the rendered set
      degradation.jsonl   # per-utt degradation metadata (which condition, which
                          # noise clip, which RIR, realized SNR, ...) -- exactly
                          # the metadata our eval harness records, so the *same*
                          # post-hoc slicing is possible on the training set too

Modes:

* ``--mode clean``  copies/converts the original clean audio to ``wav/``;
                    matches ESPnet's normal-FT data prep.
* ``--mode mct``    samples one condition per utt from a list and applies it
                    via :class:`DegradationPipeline` before writing.

The script is idempotent: utterances whose WAV is already present at the right
size are skipped, so a partial run can be resumed.

CLI::

    python -m asr_robustness.train.espnet_render_data \\
        --manifest manifests/librispeech_train-clean-100.jsonl \\
        --out-dir data/espnet/train-clean-100-mct \\
        --mode mct \\
        --degradation-config configs/degradation.yaml \\
        --conditions configs/train/espnet_mct_conditions.yaml \\
        --noise-bank data/musan \\
        --rir-bank data/RIRS_NOISES/simulated_rirs \\
        --seed-base 0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from tqdm import tqdm

from asr_robustness.audio import TARGET_SR, load_audio, peak_normalize
from asr_robustness.data.manifest import read_manifest
from asr_robustness.degrade.banks import NoiseBank, RIRBank
from asr_robustness.degrade.pipeline import DegradationPipeline


def _maybe_bank(cls, path: str | None):
    if path and Path(path).is_dir():
        return cls.from_dir(path)
    return None


def _load_conditions(path: str | None) -> list[str] | None:
    """Read the augmentation condition list from a YAML file.

    The YAML must define a top-level ``conditions:`` key whose value is a list
    of condition names. Repeat a name to weight it higher in random sampling.
    """
    if path is None:
        return None
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    conditions = cfg.get("conditions")
    if not conditions:
        raise ValueError(f"no `conditions:` list found in {path}")
    return list(conditions)


def render(
    manifest_path: str,
    out_dir: str | Path,
    mode: str,
    degradation_config: str = "configs/degradation.yaml",
    conditions_path: str | None = None,
    noise_bank_dir: str | None = None,
    rir_bank_dir: str | None = None,
    seed_base: int = 0,
    limit: int | None = None,
) -> dict:
    """Render the manifest's audio (possibly degraded) into an ESPnet-ready dir.

    Returns a summary dict with utterance counts and totals.
    """
    if mode not in {"clean", "mct"}:
        raise ValueError(f"--mode must be 'clean' or 'mct', got {mode!r}")
    if mode == "mct" and conditions_path is None:
        raise ValueError("--mode mct requires --conditions")

    out_dir = Path(out_dir)
    wav_dir = out_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    records = read_manifest(manifest_path)
    if limit:
        records = records[:limit]

    pipeline = None
    conditions: list[str] | None = None
    if mode == "mct":
        conditions = _load_conditions(conditions_path)
        pipeline = DegradationPipeline.from_config(
            degradation_config,
            noise_bank=_maybe_bank(NoiseBank, noise_bank_dir),
            rir_bank=_maybe_bank(RIRBank, rir_bank_dir),
        )
        # Fail fast if any requested condition isn't defined.
        unknown = set(conditions) - set(pipeline.condition_names())
        if unknown:
            raise ValueError(f"unknown conditions: {sorted(unknown)}")

    # We open the metadata files in append mode and rewrite the Kaldi-style
    # files at the end from the full record list. This way a partial / resumed
    # run can keep adding without corrupting the index files.
    rendered: list[dict] = []
    degradation_log: list[dict] = []

    for idx, rec in enumerate(tqdm(records, desc=f"render({mode})")):
        utt_id = rec["utt_id"]
        out_wav = wav_dir / f"{utt_id}.wav"

        # Idempotency check: if the WAV exists and is non-empty, skip.
        if out_wav.exists() and out_wav.stat().st_size > 0:
            rendered.append(rec)
            continue

        audio, sr = load_audio(rec["audio_path"], target_sr=TARGET_SR)
        if pipeline is not None:
            # Deterministic-per-utterance: same utt_id index ⇒ same condition,
            # same noise clip, same RIR. Re-runs reproduce the same training set.
            local_random = random.Random(seed_base + idx)
            condition = local_random.choice(conditions)  # type: ignore[arg-type]
            degraded, meta = pipeline.apply(audio, sr, condition, seed_base + idx)
            sf.write(str(out_wav), peak_normalize(degraded), sr)
            degradation_log.append({"utt_id": utt_id, "condition": condition, **meta})
        else:
            # Clean mode: write the un-degraded source. Peak-normalize for level
            # parity with the MCT-rendered files.
            sf.write(str(out_wav), peak_normalize(audio), sr)

        rendered.append(rec)

    # ---- Kaldi-style data files ---------------------------------------------
    # ESPnet's training pipeline expects these four files alongside the audio.
    # See https://espnet.github.io/espnet/espnet2_tutorial.html for the format.
    # All four are utt_id-keyed plain-text; ESPnet handles BPE-encoding text on
    # the fly during data loading.
    by_speaker = defaultdict(list)
    with open(out_dir / "wav.scp", "w", encoding="utf-8") as scp, \
         open(out_dir / "text", "w", encoding="utf-8") as text_f, \
         open(out_dir / "utt2spk", "w", encoding="utf-8") as utt2spk_f:
        for rec in rendered:
            utt_id = rec["utt_id"]
            speaker = rec.get("speaker") or utt_id.split("-")[0]
            wav_abs = (wav_dir / f"{utt_id}.wav").resolve()
            scp.write(f"{utt_id} {wav_abs}\n")
            text_f.write(f"{utt_id} {rec['text']}\n")
            utt2spk_f.write(f"{utt_id} {speaker}\n")
            by_speaker[speaker].append(utt_id)

    with open(out_dir / "spk2utt", "w", encoding="utf-8") as spk2utt_f:
        for speaker, utts in sorted(by_speaker.items()):
            spk2utt_f.write(f"{speaker} " + " ".join(sorted(utts)) + "\n")

    # ---- Project-style metadata for slicing later ---------------------------
    # Keep a parallel asr_robustness manifest of the rendered set so the eval
    # harness's existing manifest reader works on it directly. Also persist
    # the degradation metadata per utt (which condition, which noise clip,
    # measured SNR, etc.) for any future analysis of "did the MCT model see
    # condition X enough?" -- exactly the slicing we do on eval results.
    with open(out_dir / "manifest.jsonl", "w", encoding="utf-8") as fh:
        for rec in rendered:
            out_rec = dict(rec)
            out_rec["audio_path"] = str((wav_dir / f"{rec['utt_id']}.wav").resolve())
            fh.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
    if degradation_log:
        with open(out_dir / "degradation.jsonl", "w", encoding="utf-8") as fh:
            for entry in degradation_log:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    summary = {
        "mode": mode,
        "rendered": len(rendered),
        "speakers": len(by_speaker),
        "out_dir": str(out_dir),
        "conditions_used": conditions if mode == "mct" else None,
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, help="source .jsonl manifest")
    ap.add_argument("--out-dir", required=True, help="ESPnet data directory to create")
    ap.add_argument("--mode", required=True, choices=["clean", "mct"])
    ap.add_argument("--degradation-config", default="configs/degradation.yaml")
    ap.add_argument(
        "--conditions",
        help="path to YAML with a `conditions:` list (required for --mode mct)",
    )
    ap.add_argument("--noise-bank", default="data/musan")
    ap.add_argument("--rir-bank", default="data/RIRS_NOISES/simulated_rirs")
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--limit", type=int, help="render only the first N utterances")
    args = ap.parse_args(argv)

    summary = render(
        manifest_path=args.manifest,
        out_dir=args.out_dir,
        mode=args.mode,
        degradation_config=args.degradation_config,
        conditions_path=args.conditions,
        noise_bank_dir=args.noise_bank,
        rir_bank_dir=args.rir_bank,
        seed_base=args.seed_base,
        limit=args.limit,
    )
    print("\n=== ESPnet data prep summary ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
