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
    ap.add_argument(
        "--models",
        help=(
            "comma-separated list of model NAMES from the YAML to run (others "
            "are skipped). Lets you re-run a subset without editing the YAML, "
            "e.g. after a single arm failed mid-loop."
        ),
    )
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

    name_filter = set(s.strip() for s in args.models.split(",")) if args.models else None

    out_path = Path(cfg["output"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    for spec in cfg["models"]:
        if name_filter is not None and spec["name"] not in name_filter:
            print(f"\n[{spec['name']}] skipped (--models filter)")
            continue
        model_pkg.ensure_loaded(spec["key"])
        # Pass through every YAML field EXCEPT the two run-level meta keys
        # ('name' is the display label, 'key' is the registry key). Anything
        # adapter-specific (model_id, base_model_id, device, beam_size, etc.)
        # is forwarded verbatim. This means new adapter kwargs don't need a
        # corresponding change to run.py.
        kwargs = {k: v for k, v in spec.items() if k != "key"}
        model = model_pkg.create(spec["key"], **kwargs)
        print(f"\n[{spec['name']}] decoding {len(records[:limit] if limit else records)} "
              f"utterances x {len(conditions)} conditions ...")
        results = run_decode(
            records, model, pipeline, conditions,
            seed_base=cfg.get("seed_base", 0), limit=limit,
        )
        all_results.extend(results)
        # Per-arm checkpoint: write this arm's rows to a side file as soon as
        # it finishes, so if a later arm crashes (or the laptop is closed) we
        # don't lose hours of decoding. The merge into the main output file
        # still happens at the end -- the per-arm file is purely a safety net.
        per_arm_path = out_path.with_name(f"{out_path.stem}__{spec['name']}.jsonl")
        write_results(results, per_arm_path)
        print(f"  -> per-arm checkpoint: {per_arm_path}")
        _print_summary(spec["name"], results, conditions)

    out = write_results(all_results, out_path)
    print(f"\nwrote {len(all_results)} result rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
