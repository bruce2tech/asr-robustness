"""Plots: WER curves, hallucination signals, condition comparisons.

All plots use the headless Agg backend so they render under `make report` or in
CI without a display.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def wer_vs_snr(df: pd.DataFrame, out_path: str | Path, family: str = "noise") -> Path | None:
    """One WER-vs-SNR curve per model for a given degradation ``family``."""
    sub = df[df["family"] == family]
    if sub.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, g in sub.groupby("model"):
        g = g.sort_values("snr_db")
        ax.plot(g["snr_db"], g["wer"] * 100, marker="o", label=model)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("WER (%)")
    ax.set_title(f"WER vs SNR — {family}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return _save(fig, out_path)


def noise_vs_babble(df: pd.DataFrame, out_path: str | Path) -> Path | None:
    """Overlay stationary-noise and babble curves -- shows noise *type* matters."""
    sub = df[df["family"].isin(["noise", "babble"])]
    if sub.empty or "babble" not in set(sub["family"]):
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (model, family), g in sub.groupby(["model", "family"]):
        g = g.sort_values("snr_db")
        style = "-" if family == "noise" else "--"
        ax.plot(g["snr_db"], g["wer"] * 100, style, marker="o",
                label=f"{model} ({family})")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("WER (%)")
    ax.set_title("Stationary noise vs. babble at matched SNR")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, out_path)


def condition_bars(df: pd.DataFrame, out_path: str | Path) -> Path | None:
    """Grouped bar chart for non-SNR conditions (reverb, telephone, codecs, packet loss)."""
    sub = df[df["snr_db"].isna()]
    if sub.empty:
        return None
    conditions = sorted(sub["condition"].unique())
    models = sorted(sub["model"].unique())
    fig, ax = plt.subplots(figsize=(max(7, len(conditions) * 0.9), 4.5))
    x = np.arange(len(conditions))
    bar_w = 0.8 / max(1, len(models))
    for i, model in enumerate(models):
        values = []
        for cond in conditions:
            row = sub[(sub["model"] == model) & (sub["condition"] == cond)]
            values.append(float(row["wer"].iloc[0] * 100) if len(row) else 0.0)
        offset = i * bar_w - 0.4 + bar_w / 2
        ax.bar(x + offset, values, width=bar_w, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=30, ha="right")
    ax.set_ylabel("WER (%)")
    ax.set_title("WER by condition (non-SNR)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, out_path)


def hallucination_vs_snr(
    df: pd.DataFrame, out_path: str | Path, family: str = "noise"
) -> Path | None:
    """Insertion rate vs SNR -- under heavy noise Whisper inserts/hallucinates words."""
    sub = df[df["family"] == family]
    if sub.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, g in sub.groupby("model"):
        g = g.sort_values("snr_db")
        ax.plot(g["snr_db"], g["insertion_rate"], marker="o", label=model)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("insertion rate (insertions per reference word)")
    ax.set_title(f"Hallucination signal — insertion rate vs SNR ({family})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return _save(fig, out_path)
