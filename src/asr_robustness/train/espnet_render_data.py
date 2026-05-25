"""Pre-render (and optionally degrade) training audio to disk for ESPnet FT.

ESPnet's training pipeline ("native recipe", Path II in the project plan) wants
audio on disk plus Kaldi-style metadata files; it does not have a HuggingFace-
style in-memory online augmentation hook. So for the multi-condition ESPnet
fine-tune we pre-render one degraded WAV per training utterance, sampling a
condition with a deterministic seed so the rendering is reproducible.

Output layout (under ``--out-dir``):

    <out_dir>/
      wav/<utt_id>.wav    # 16 kHz mono WAV, possibly degraded (mct mode only)
      wav.scp             # "<utt_id> /abs/path/to/audio" lines
      text                # "<utt_id> <reference text>" lines
      utt2spk             # "<utt_id> <speaker_id>" lines
      spk2utt             # "<speaker_id> <utt_id1> <utt_id2> ..." lines
      manifest.jsonl      # parallel asr_robustness manifest of the rendered set
      degradation.jsonl   # per-utt degradation metadata (mct only)

Modes:

* ``--mode clean``  references the original FLAC files in ``wav.scp`` directly.
                    ESPnet's ``sound`` data type reads FLAC via soundfile, so
                    no conversion is needed and no audio is rewritten -- this
                    makes clean-mode prep nearly instantaneous (~seconds for
                    train-clean-100's 28k utts).
* ``--mode mct``    parallelizes across worker processes: each worker
                    initializes its own DegradationPipeline (with shared
                    noise/RIR banks loaded once per worker), then renders
                    its share of utterances. With 8 workers, render time for
                    train-clean-100 drops from ~3 h single-threaded to ~25 min.

The script is idempotent: utterances whose WAV is already present at the right
size are skipped, so a partial run can be resumed.

CLI::

    python -m asr_robustness.train.espnet_render_data \\
        --manifest manifests/librispeech_train-clean-100.jsonl \\
        --out-dir data/espnet/train-clean-100-mct \\
        --mode mct \\
        --conditions configs/train/espnet_mct_conditions.yaml \\
        --noise-bank data/musan --rir-bank data/RIRS_NOISES/simulated_rirs \\
        --num-workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import soundfile as sf
import yaml
from tqdm import tqdm

from asr_robustness.audio import TARGET_SR, load_audio, peak_normalize
from asr_robustness.data.manifest import read_manifest
from asr_robustness.degrade.banks import NoiseBank, RIRBank
from asr_robustness.degrade.pipeline import DegradationPipeline


# ----- worker state ----------------------------------------------------------
# Populated once per worker process by ``_worker_init``. Keeping the pipeline
# at module scope means each worker pays the bank-loading cost once, not once
# per utterance.
_WORKER_PIPELINE: DegradationPipeline | None = None
_WORKER_CONDITIONS: list[str] | None = None


def _maybe_bank(cls, path: str | None):
    if path and Path(path).is_dir():
        return cls.from_dir(path)
    return None


def _load_conditions(path: str | None) -> list[str] | None:
    if path is None:
        return None
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    conditions = cfg.get("conditions")
    if not conditions:
        raise ValueError(f"no `conditions:` list found in {path}")
    return list(conditions)


def _worker_init(degradation_config: str, conditions_path: str,
                 noise_dir: str | None, rir_dir: str | None) -> None:
    """Per-worker initializer: build the pipeline and load the condition list.

    Called once per worker process when the ProcessPoolExecutor spins up.
    Subsequent ``_render_one`` calls in the same worker reuse this state.
    """
    global _WORKER_PIPELINE, _WORKER_CONDITIONS
    _WORKER_CONDITIONS = _load_conditions(conditions_path)
    _WORKER_PIPELINE = DegradationPipeline.from_config(
        degradation_config,
        noise_bank=_maybe_bank(NoiseBank, noise_dir),
        rir_bank=_maybe_bank(RIRBank, rir_dir),
    )


def _render_one(args: tuple) -> tuple:
    """Render one MCT utterance in a worker process.

    Returns ``(utt_id, wav_path, status, degradation_meta)``, where:
      * ``wav_path`` is the absolute path written to ``wav.scp``
      * ``status`` is one of: ``"rendered"``, ``"skipped"``, ``"failed"``
      * ``degradation_meta`` is the per-utt degradation record (None on skip/fail)
    """
    idx, rec, wav_dir_str, seed_base = args
    wav_dir = Path(wav_dir_str)
    utt_id = rec["utt_id"]
    out_wav = wav_dir / f"{utt_id}.wav"

    # Idempotency: skip utts whose WAV is already present and non-empty.
    if out_wav.exists() and out_wav.stat().st_size > 0:
        return (utt_id, str(out_wav.resolve()), "skipped", None)

    try:
        audio, sr = load_audio(rec["audio_path"], target_sr=TARGET_SR)
        # Deterministic-per-utterance condition choice + degradation seed.
        local_random = random.Random(seed_base + idx)
        condition = local_random.choice(_WORKER_CONDITIONS)
        degraded, meta = _WORKER_PIPELINE.apply(audio, sr, condition, seed_base + idx)
        sf.write(str(out_wav), peak_normalize(degraded), sr)
        return (
            utt_id,
            str(out_wav.resolve()),
            "rendered",
            {"utt_id": utt_id, "condition": condition, **meta},
        )
    except Exception as exc:
        return (utt_id, f"{type(exc).__name__}: {exc}", "failed", None)


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
    num_workers: int | None = None,
) -> dict:
    """Render the manifest's audio into an ESPnet-ready directory.

    See module docstring for the output layout and the clean vs mct modes.
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

    # Validate the requested conditions exist in the degradation config before
    # spinning up workers (fail fast, not after 20 minutes of parallel work).
    if mode == "mct":
        conditions = _load_conditions(conditions_path)
        validation_pipeline = DegradationPipeline.from_config(
            degradation_config,
            noise_bank=_maybe_bank(NoiseBank, noise_bank_dir),
            rir_bank=_maybe_bank(RIRBank, rir_bank_dir),
        )
        unknown = set(conditions) - set(validation_pipeline.condition_names())
        if unknown:
            raise ValueError(f"unknown conditions: {sorted(unknown)}")

    rendered_paths: dict[str, str] = {}
    degradation_log: list[dict] = []
    failed: list[tuple[str, str]] = []

    if mode == "clean":
        # Clean mode optimization: don't write any new audio. ESPnet's `sound`
        # data type uses soundfile, which reads FLAC fine. Pointing wav.scp at
        # the original FLAC paths is equivalent for training and saves an
        # entire FLAC->WAV transcode pass over the corpus.
        for rec in records:
            rendered_paths[rec["utt_id"]] = str(Path(rec["audio_path"]).resolve())
    else:
        # MCT mode: parallelize across CPU cores. Each worker builds its own
        # DegradationPipeline (bank load is lazy so the worker startup cost
        # is small) and then renders its share of utterances.
        #
        # CAP default workers at 16 -- some cloud GPU pods (e.g. RunPod L40S)
        # report 128 vCPUs, but spinning up 128 worker processes each holding
        # DegradationPipeline state (banks + per-task audio buffers) OOMs the
        # container. Sweet spot is ~8-16 workers: still ~10x speedup over
        # single-threaded, but well within memory headroom. The user can
        # override explicitly via --num-workers if their pod has more RAM.
        DEFAULT_WORKER_CAP = 16
        workers = num_workers or min(os.cpu_count() or 1, DEFAULT_WORKER_CAP)
        args_list = [
            (idx, rec, str(wav_dir), seed_base) for idx, rec in enumerate(records)
        ]
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(degradation_config, conditions_path, noise_bank_dir, rir_bank_dir),
        ) as exe:
            for utt_id, wav_path, status, meta in tqdm(
                exe.map(_render_one, args_list, chunksize=4),
                total=len(args_list),
                desc=f"render(mct, {workers}w)",
            ):
                if status == "failed":
                    failed.append((utt_id, wav_path))
                    continue
                rendered_paths[utt_id] = wav_path
                if meta is not None:
                    degradation_log.append(meta)

    if failed:
        print(f"\nWARNING: {len(failed)} utterances failed to render. First 10:")
        for utt_id, err in failed[:10]:
            print(f"  {utt_id}: {err}")

    rendered_records = [r for r in records if r["utt_id"] in rendered_paths]

    # ---- Kaldi-style data files (ESPnet's training pipeline expects these) --
    by_speaker = defaultdict(list)
    with open(out_dir / "wav.scp", "w", encoding="utf-8") as scp, \
         open(out_dir / "text", "w", encoding="utf-8") as text_f, \
         open(out_dir / "utt2spk", "w", encoding="utf-8") as utt2spk_f:
        for rec in rendered_records:
            utt_id = rec["utt_id"]
            speaker = rec.get("speaker") or utt_id.split("-")[0]
            scp.write(f"{utt_id} {rendered_paths[utt_id]}\n")
            text_f.write(f"{utt_id} {rec['text']}\n")
            utt2spk_f.write(f"{utt_id} {speaker}\n")
            by_speaker[speaker].append(utt_id)

    with open(out_dir / "spk2utt", "w", encoding="utf-8") as spk2utt_f:
        for speaker, utts in sorted(by_speaker.items()):
            spk2utt_f.write(f"{speaker} " + " ".join(sorted(utts)) + "\n")

    # ---- Project-style metadata for slicing later ---------------------------
    with open(out_dir / "manifest.jsonl", "w", encoding="utf-8") as fh:
        for rec in rendered_records:
            out_rec = dict(rec)
            out_rec["audio_path"] = rendered_paths[rec["utt_id"]]
            fh.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
    if degradation_log:
        with open(out_dir / "degradation.jsonl", "w", encoding="utf-8") as fh:
            for entry in degradation_log:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    summary = {
        "mode": mode,
        "rendered": len(rendered_records),
        "skipped_or_failed": len(records) - len(rendered_records),
        "speakers": len(by_speaker),
        "out_dir": str(out_dir),
        "workers": (num_workers or os.cpu_count() or 1) if mode == "mct" else None,
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
    ap.add_argument(
        "--num-workers",
        type=int,
        help="MCT mode: parallel worker processes (default: os.cpu_count())",
    )
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
        num_workers=args.num_workers,
    )
    print("\n=== ESPnet data prep summary ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
