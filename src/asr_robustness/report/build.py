"""CLI: build the report (plots + CSV + markdown summary) from a results file.

    python -m asr_robustness.report.build --results results/snr_sweep.jsonl

Writes ``summary.csv``, ``summary.md``, and one PNG per available plot into
the output directory (default ``reports/``). Plots whose data is missing (e.g.
no babble conditions in this run) are quietly skipped.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from asr_robustness.report import plots
from asr_robustness.report.analyze import load_summary
from asr_robustness.report.tables import df_to_markdown, wer_pivot


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the degradation analysis report.")
    ap.add_argument("--results", default="results/snr_sweep.jsonl")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args(argv)

    df = load_summary(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "summary.csv", index=False)

    figures = {
        "WER vs SNR": plots.wer_vs_snr(df, out / "wer_vs_snr.png"),
        "Stationary noise vs babble": plots.noise_vs_babble(df, out / "noise_vs_babble.png"),
        "Conditions (non-SNR)": plots.condition_bars(df, out / "conditions.png"),
        "Hallucination signal": plots.hallucination_vs_snr(df, out / "hallucination.png"),
    }

    lines = [
        "# ASR Robustness Under Acoustic Degradation — Results",
        "",
        f"Source: `{args.results}` — {len(df)} (model, condition) summaries",
        "",
        "## WER (%) by model × condition",
        "",
        df_to_markdown(wer_pivot(df)),
        "",
    ]
    for label, path in figures.items():
        if path is not None:
            lines += [f"### {label}", f"![{label}]({path.name})", ""]
    (out / "summary.md").write_text("\n".join(lines))

    n_plots = sum(1 for v in figures.values() if v is not None)
    print(f"wrote {out / 'summary.csv'}, {out / 'summary.md'}, and {n_plots} figure(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
