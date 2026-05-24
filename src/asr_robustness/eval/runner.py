"""The decode runner: drive a model over degraded audio and score every result.

For each utterance in a manifest and each named degradation condition, the
runner: loads the clean audio, applies the (reproducible) degradation, runs the
model, scores the hypothesis against the reference, and records one result row
with the full degradation metadata attached. Those rows are the raw material
for every Phase 5/7 breakdown.
"""

from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from asr_robustness.audio import load_audio
from asr_robustness.degrade.pipeline import DegradationPipeline
from asr_robustness.eval.scoring import score_utterance
from asr_robustness.models.base import ASRModel

# Manifest fields copied verbatim onto each result row (enable later slicing).
_CARRY_FIELDS = ("dataset", "split", "speaker", "accent", "domain")


def run_decode(
    records: list[dict],
    model: ASRModel,
    pipeline: DegradationPipeline,
    conditions: list[str],
    seed_base: int = 0,
    limit: int | None = None,
    progress: bool = True,
) -> list[dict]:
    """Decode ``model`` over every (utterance, condition) pair; return result rows.

    Each utterance gets a stable seed (``seed_base + index``) so its degradation
    is identical across models and across re-runs -- the basis for fair, paired
    model comparison.
    """
    records = records[:limit] if limit else records
    results: list[dict] = []
    iterator = tqdm(records, desc=model.name, disable=not progress)

    for index, rec in enumerate(iterator):
        clean, sr = load_audio(rec["audio_path"])
        seed = seed_base + index
        for condition in conditions:
            degraded, meta = pipeline.apply(clean, sr, condition, seed)
            hypothesis = model.transcribe(degraded, sr)
            row = {
                "utt_id": rec["utt_id"],
                "model": model.name,
                "condition": condition,
                "seed": seed,
                "reference": rec["text"],
                "hypothesis": hypothesis,
                **{f: rec.get(f) for f in _CARRY_FIELDS},
                **score_utterance(rec["text"], hypothesis),
                "degradation": meta,
            }
            results.append(row)
    return results


def write_results(results: list[dict], path: str | Path) -> Path:
    """Write result rows to a ``.jsonl`` file (parent directories created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in results:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def read_results(path: str | Path) -> list[dict]:
    """Read result rows back from a ``.jsonl`` file."""
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
