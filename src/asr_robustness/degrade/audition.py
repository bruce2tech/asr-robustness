"""Audition tool: render clean vs degraded audio to disk so you can *listen*.

Hearing the degradation is a necessary validation step. A WER-vs-SNR curve is
only trustworthy once you have confirmed by ear that the conditions are what
they claim to be -- that a 0 dB mix really is barely intelligible, that the
"telephone" condition really sounds like a phone line, that reverb sounds like
a room and not a bug. It is also where the qualitative examples for the writeup
come from.

    python -m asr_robustness.degrade.audition \\
        --manifest manifests/librispeech_dev-clean.jsonl --index 0 \\
        --conditions clean noise_0db telephone reverb --out demo/

One WAV is written per condition (peak-normalized so levels are comparable and
nothing clips). On macOS, --play auditions them in order via `afplay`.
Conditions whose noise/RIR bank is not downloaded yet are skipped with a note.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
from pathlib import Path

from asr_robustness.audio import load_audio, peak_normalize, save_audio
from asr_robustness.data.manifest import read_manifest
from asr_robustness.degrade.banks import NoiseBank, RIRBank
from asr_robustness.degrade.effects import measure_snr
from asr_robustness.degrade.pipeline import DegradationPipeline, load_conditions

DEFAULT_CONDITIONS = ["clean", "noise_10db", "noise_0db", "reverb", "telephone"]


def _maybe_bank(cls, path: str | None):
    """Build a bank if its directory exists; else None (condition gets skipped)."""
    return cls.from_dir(path) if path and Path(path).is_dir() else None


def _stage_summary(info: dict) -> str:
    """One-line human description of an applied degradation stage."""
    effect = info.get("effect", "?")
    if effect == "add_noise":
        return f"noise snr={info['snr_db']:.0f}dB clip={info['noise_id']}"
    if effect == "add_babble":
        return f"babble snr={info['snr_db']:.0f}dB, {info['n_talkers']} competing talkers"
    if effect == "add_reverb":
        return f"reverb rir={info['rir_id']}"
    if effect == "narrowband":
        return f"band-limit {info['low_hz']:.0f}-{info['high_hz']:.0f}Hz"
    if effect == "mu_law_codec":
        return f"mu-law codec (mu={info['mu']})"
    if effect == "packet_loss":
        return (f"packet loss {info['loss_rate'] * 100:.0f}% "
                f"({info['dropped']}/{info['n_packets']} pkts dropped)")
    if effect == "apply_codec":
        return f"codec {info['codec']} (bitrate {info['bitrate']})"
    if effect == "clip":
        return f"clip at {info['percentile']:.0f}th pct"
    if effect == "gain":
        return f"gain {info['gain_db']:+.0f}dB"
    return effect


def audition(
    audio_path: str,
    conditions: list[str],
    out_dir: str | Path = "demo",
    degradation_config: str = "configs/degradation.yaml",
    noise_dir: str | None = "data/musan",
    rir_dir: str | None = "data/RIRS_NOISES/simulated_rirs",
    seed: int = 0,
    reference: str | None = None,
    utt_id: str | None = None,
    play: bool = False,
) -> list[Path]:
    """Render ``audio_path`` under each condition; return the written WAV paths."""
    clean, sr = load_audio(audio_path)
    pipeline = DegradationPipeline.from_config(
        degradation_config,
        noise_bank=_maybe_bank(NoiseBank, noise_dir),
        rir_bank=_maybe_bank(RIRBank, rir_dir),
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = utt_id or Path(audio_path).stem

    print(f"utterance: {stem}  ({len(clean) / sr:.1f}s @ {sr} Hz)")
    if reference:
        print(f"reference: {reference}")
    print()

    written: list[Path] = []
    for condition in conditions:
        try:
            degraded, meta = pipeline.apply(clean, sr, condition, seed)
        except (KeyError, ValueError) as exc:
            print(f"  [skip] {condition:16s} {exc}")
            continue
        path = out_dir / f"{stem}__{condition}.wav"
        save_audio(path, peak_normalize(degraded), sr)

        stages = meta["stages"]
        detail = "; ".join(_stage_summary(s) for s in stages) if stages else "(unmodified)"
        # For a pure additive condition (noise or babble), confirm the realized SNR.
        if len(stages) == 1 and stages[0].get("effect") in ("add_noise", "add_babble"):
            detail += f"  [measured SNR {measure_snr(clean, degraded):+.1f} dB]"
        print(f"  {condition:16s} -> {path.name}")
        print(f"  {'':16s}    {detail}")
        written.append(path)

    print(f"\n{len(written)} file(s) in {out_dir}/")
    if written:
        print(f"listen:  afplay {out_dir}/{stem}__<condition>.wav     (or open {out_dir}/ in Finder)")
    if play:
        _play(written)
    return written


def _play(paths: list[Path]) -> None:
    """Play each WAV in order (macOS `afplay`)."""
    if platform.system() != "Darwin":
        print("--play needs macOS (afplay); open the files manually instead.")
        return
    for path in paths:
        print(f"  playing {path.name} ...")
        subprocess.run(["afplay", str(path)], check=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render clean vs degraded audio to listen to.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", help="manifest to pick an utterance from")
    src.add_argument("--audio", help="a single audio file to degrade")
    ap.add_argument("--index", type=int, default=0, help="utterance index in --manifest")
    ap.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    ap.add_argument("--out", default="demo", help="output directory")
    ap.add_argument("--degradation-config", default="configs/degradation.yaml")
    ap.add_argument("--noise-dir", default="data/musan")
    ap.add_argument("--rir-dir", default="data/RIRS_NOISES/simulated_rirs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--play", action="store_true", help="play each file (macOS)")
    ap.add_argument("--list-conditions", action="store_true", help="list conditions and exit")
    args = ap.parse_args(argv)

    if args.list_conditions:
        for name in load_conditions(args.degradation_config):
            print(name)
        return 0

    reference = utt_id = None
    if args.manifest:
        records = read_manifest(args.manifest)
        if not 0 <= args.index < len(records):
            ap.error(f"--index {args.index} out of range (manifest has {len(records)})")
        rec = records[args.index]
        audio_path, reference, utt_id = rec["audio_path"], rec["text"], rec["utt_id"]
    else:
        audio_path = args.audio

    audition(
        audio_path,
        conditions=args.conditions,
        out_dir=args.out,
        degradation_config=args.degradation_config,
        noise_dir=args.noise_dir,
        rir_dir=args.rir_dir,
        seed=args.seed,
        reference=reference,
        utt_id=utt_id,
        play=args.play,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
