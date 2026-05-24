"""Normalized JSON-lines manifests.

Every dataset, however it is stored on disk, is reduced to one **manifest**: a
JSON-lines file with one :class:`Utterance` per line. The evaluation harness,
the degradation pipeline, and the analysis code all consume manifests and
nothing else -- so adding a new corpus only means writing one builder here.

The ``accent`` and ``domain`` fields are what make the Phase 7 breakdowns
possible: WER can be sliced by accent group or by read-vs-conversational
domain straight from the manifest.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf


@dataclass
class Utterance:
    """One utterance: a pointer to audio plus its reference text and metadata."""

    utt_id: str
    audio_path: str
    text: str
    duration: float
    dataset: str
    split: str
    speaker: str | None = None
    accent: str | None = None  # populated for Common Voice; drives the accent breakdown
    domain: str = "read"  # "read" | "conversational" | "meeting" | ...

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), ensure_ascii=False)


def write_manifest(records: list[Utterance], path: str | Path) -> Path:
    """Write utterances to a ``.jsonl`` manifest (parent dirs created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(rec.to_json() + "\n")
    return path


def read_manifest(path: str | Path) -> list[dict]:
    """Read a ``.jsonl`` manifest back into a list of dicts."""
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_librispeech_manifest(
    data_root: str | Path, split: str, out_dir: str | Path = "manifests"
) -> Path:
    """Build a manifest for one LibriSpeech split (e.g. ``test-clean``).

    LibriSpeech stores audio as ``<speaker>/<chapter>/<utt>.flac`` with a
    sidecar ``<speaker>-<chapter>.trans.txt`` mapping utterance IDs to
    upper-cased reference text (text normalization is deferred to WER scoring).
    """
    split_dir = Path(data_root) / "LibriSpeech" / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"LibriSpeech split not found: {split_dir}")

    records: list[Utterance] = []
    for trans in sorted(split_dir.rglob("*.trans.txt")):
        for line in trans.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            utt_id, text = line.split(" ", 1)
            audio = trans.parent / f"{utt_id}.flac"
            if not audio.exists():
                raise FileNotFoundError(f"missing audio for {utt_id}: {audio}")
            records.append(
                Utterance(
                    utt_id=utt_id,
                    audio_path=str(audio),
                    text=text,
                    duration=float(sf.info(str(audio)).duration),
                    dataset="librispeech",
                    split=split,
                    speaker=utt_id.split("-")[0],
                    domain="read",
                )
            )
    if not records:
        raise ValueError(f"no utterances found under {split_dir}")
    out = write_manifest(records, Path(out_dir) / f"librispeech_{split}.jsonl")
    total_h = sum(r.duration for r in records) / 3600.0
    print(f"librispeech/{split}: {len(records)} utterances, {total_h:.2f} h -> {out}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build dataset manifests.")
    ap.add_argument("--dataset", default="librispeech", choices=["librispeech"])
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--split", required=True, help="e.g. test-clean")
    ap.add_argument("--out-dir", default="manifests")
    args = ap.parse_args(argv)

    if args.dataset == "librispeech":
        build_librispeech_manifest(args.data_root, args.split, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
