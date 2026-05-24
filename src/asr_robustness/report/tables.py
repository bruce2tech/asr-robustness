"""Tabular summaries (markdown + CSV)."""

from __future__ import annotations

import pandas as pd


def wer_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """A model × condition table of corpus WER (%), ready for rendering."""
    pivot = df.pivot_table(index="model", columns="condition", values="wer")
    return (pivot * 100).round(1).reset_index()


def df_to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavored markdown table.

    Floats are formatted to one decimal place; ``NaN`` becomes ``-``.
    """
    cols = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            value = row[col]
            if pd.isna(value):
                cells.append("-")
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                cells.append(f"{value:.1f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
