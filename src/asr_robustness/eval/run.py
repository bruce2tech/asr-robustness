"""CLI: run a decode experiment defined by a YAML config.

    python -m asr_robustness.eval.run --config configs/experiments/main.yaml

The config names a manifest, a degradation config + condition list, the noise /
RIR bank directories, and the list of models. The runner decodes every model
over every condition, prints a per-condition summary, and writes all result
rows to one ``.jsonl`` file for downstream analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from asr_robustness import models as model_pkg
from asr_robustness.data.manifest import read_manifest
from asr_robustness.degrade.banks import NoiseBank, RIRBank
from asr_robustness.degrade.pipeline import DegradationPipeline
from asr_robustness.eval.runner import run_decode, write_results
from asr_robustness.eval.scoring import aggregate


def _maybe_bank(cls, path: str | None):
    """Build a bank from ``path`` if it exists; otherwise None (some conditions need none)."""
    if path and Path(path).is_dir():
        return cls.from_dir(path)
    return None


def _print_summary(model_name: str, results: list[dict], conditions: list[str]) -> None:
    print(f"\n{model_name}")
    print(f"  {'condition':18s} {'WER':>7s} {'ins_rate':>9s} {'len_ratio':>10s} {'n':>6s}")
    for cond in conditions:
        agg = aggregate([r for r in results if r["condition"] == cond])
        print(
            f"  {cond:18s} {agg['wer']:7.3f} {agg.get('insertion_rate', 0):9.3f} "
            f"{agg.get('length_ratio', 0):10.2f} {agg['n_utterances']:6d}"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run a decode experiment.")
    ap.add_argument("--config", required=True, help="experiment YAML config")
    ap.add_argument("--limit", type=int, help="override: decode only the first N utterances")
    args = ap.parse_args(argv)

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    records = read_manifest(cfg["manifest"])
    pipeline = DegradationPipeline.from_config(
        cfg["degradation_config"],
        noise_bank=_maybe_bank(NoiseBank, cfg.get("noise_bank")),
        rir_bank=_maybe_bank(RIRBank, cfg.get("rir_bank")),
    )
    conditions = cfg["conditions"]
    limit = args.limit if args.limit is not None else cfg.get("limit")

    all_results: list[dict] = []
    for spec in cfg["models"]:
        model_pkg.ensure_loaded(spec["key"])
        kwargs = {"name": spec["name"]}
        if "model_id" in spec:
            kwargs["model_id"] = spec["model_id"]
        model = model_pkg.create(spec["key"], **kwargs)
        print(f"\n[{spec['name']}] decoding {len(records[:limit] if limit else records)} "
              f"utterances x {len(conditions)} conditions ...")
        results = run_decode(
            records, model, pipeline, conditions,
            seed_base=cfg.get("seed_base", 0), limit=limit,
        )
        all_results.extend(results)
        _print_summary(spec["name"], results, conditions)

    out = write_results(all_results, cfg["output"])
    print(f"\nwrote {len(all_results)} result rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
