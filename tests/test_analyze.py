"""Unit tests for the analysis layer (parse_condition, summarize, plots, tables)."""

import pandas as pd
import pytest

from asr_robustness.report import plots
from asr_robustness.report.analyze import parse_condition, summarize
from asr_robustness.report.tables import df_to_markdown, wer_pivot


def _row(model, condition, **scores):
    """A synthetic result row -- only fields aggregate() needs are required."""
    defaults = {
        "wer": 0.0, "cer": 0.0, "hits": 10, "substitutions": 0, "insertions": 0,
        "deletions": 0, "ref_words": 10, "hyp_words": 10, "length_ratio": 1.0,
    }
    return {"model": model, "condition": condition, **defaults, **scores}


def test_parse_condition_handles_snr_swept_names():
    assert parse_condition("clean") == ("clean", None)
    assert parse_condition("noise_10db") == ("noise", 10)
    assert parse_condition("noise_-5db") == ("noise", -5)
    assert parse_condition("babble_0db") == ("babble", 0)
    assert parse_condition("reverb") == ("reverb", None)
    # Compound names: the SNR still parses; everything before is the family.
    assert parse_condition("reverb_noise_5db") == ("reverb_noise", 5)


def test_summarize_aggregates_per_model_and_condition():
    results = [
        _row("m1", "clean"),
        _row("m1", "clean", substitutions=1, ref_words=10),
        _row("m1", "noise_0db", substitutions=5, ref_words=10),
        _row("m2", "clean"),
    ]
    df = summarize(results)

    assert len(df) == 3  # 2 (model,condition) pairs for m1 + 1 for m2
    m1_clean = df[(df["model"] == "m1") & (df["condition"] == "clean")].iloc[0]
    # m1/clean: 2 utterances, total 20 ref words, 1 substitution -> WER = 1/20.
    assert m1_clean["wer"] == pytest.approx(1 / 20)
    assert m1_clean["family"] == "clean"
    assert pd.isna(m1_clean["snr_db"])

    m1_noise = df[df["condition"] == "noise_0db"].iloc[0]
    assert m1_noise["family"] == "noise"
    assert int(m1_noise["snr_db"]) == 0


def test_wer_pivot_is_percentages_with_model_column():
    results = [_row("m1", "clean", substitutions=1, ref_words=10),
               _row("m1", "noise_0db", substitutions=5, ref_words=10),
               _row("m2", "clean")]
    pivot = wer_pivot(summarize(results))
    assert "model" in pivot.columns
    m1 = pivot[pivot["model"] == "m1"].iloc[0]
    assert m1["clean"] == pytest.approx(10.0)
    assert m1["noise_0db"] == pytest.approx(50.0)


def test_df_to_markdown_handles_nans_and_floats():
    df = pd.DataFrame({"a": ["x", "y"], "wer": [12.34, None]})
    md = df_to_markdown(df)
    assert md.startswith("| a | wer |")
    assert "12.3" in md
    assert "| - |" in md  # NaN rendered as "-"


def test_wer_vs_snr_skips_empty_family(tmp_path):
    df = summarize([_row("m1", "clean")])  # no "noise" family rows
    assert plots.wer_vs_snr(df, tmp_path / "x.png") is None


def test_plots_emit_png_when_data_present(tmp_path):
    rows = [
        _row("m1", "clean"),
        _row("m1", "noise_10db", substitutions=1, ref_words=10),
        _row("m1", "noise_0db", substitutions=4, ref_words=10),
        _row("m1", "babble_0db", substitutions=6, ref_words=10),
        _row("m1", "telephone", substitutions=2, ref_words=10),
    ]
    df = summarize(rows)
    assert plots.wer_vs_snr(df, tmp_path / "snr.png") is not None
    assert plots.noise_vs_babble(df, tmp_path / "nb.png") is not None
    assert plots.condition_bars(df, tmp_path / "bars.png") is not None
    assert plots.hallucination_vs_snr(df, tmp_path / "hall.png") is not None
