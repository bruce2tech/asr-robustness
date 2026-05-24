"""Aggregate decode results into per-(model, condition) summaries.

The decode runner emits one row per (utterance, condition). This module rolls
those up to the condition level -- corpus WER, error-type rates, and the
hallucination signals -- and parses the SNR out of condition names so that
WER-vs-SNR curves fall out directly.
"""

from __future__ import annotations

import re
from collections import defaultdict

import pandas as pd

from asr_robustness.eval.runner import read_results
from asr_robustness.eval.scoring import aggregate

# Matches condition names that carry an SNR, e.g. "noise_-5db", "babble_0db".
_SNR_RE = re.compile(r"^(.*?)_(-?\d+)db$")


def parse_condition(name: str) -> tuple[str, int | None]:
    """Split a condition name into ``(family, snr_db)``.

    ``"noise_-5db" -> ("noise", -5)``; ``"clean" -> ("clean", None)``.
    The family groups conditions for plotting (one WER-vs-SNR curve per family).
    """
    match = _SNR_RE.match(name)
    if match:
        return match.group(1), int(match.group(2))
    return name, None


def summarize(results: list[dict]) -> pd.DataFrame:
    """Aggregate raw result rows into a per-(model, condition) DataFrame.

    Columns: model, condition, family, snr_db, plus the corpus metrics from
    :func:`asr_robustness.eval.scoring.aggregate` (wer, insertion_rate, ...).
    """
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in results:
        groups[(row["model"], row["condition"])].append(row)

    records = []
    for (model, condition), group in groups.items():
        family, snr = parse_condition(condition)
        records.append(
            {"model": model, "condition": condition, "family": family, "snr_db": snr}
            | aggregate(group)
        )
    df = pd.DataFrame(records)
    return df.sort_values(
        ["model", "family", "snr_db"], na_position="first"
    ).reset_index(drop=True)


def load_summary(results_path: str) -> pd.DataFrame:
    """Read a results ``.jsonl`` file and summarize it."""
    return summarize(read_results(results_path))
