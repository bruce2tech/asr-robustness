"""CLI: fine-tune a pretrained ESPnet ASR model on pre-rendered data.

    python -m asr_robustness.train.espnet_ft --config configs/train/espnet_clean_ft.yaml

Path II of the Phase 6 ESPnet plan: invoke ESPnet's *native* training pipeline
(``espnet2.bin.asr_train``) on data we've pre-rendered to disk with
:mod:`asr_robustness.train.espnet_render_data`. We use ESPnet's own training
code so we don't have to re-implement the model's encoder/decoder/CTC head
training loop -- and so the fine-tune is mechanically identical to how the
upstream ESPnet team trains these models.

What this driver actually does:

1. Read our YAML config (model_id, data_dirs, training hyperparameters).
2. Download the pretrained ESPnet bundle via :class:`ModelDownloader`. The
   bundle contains the model checkpoint plus the training-time config, BPE
   model, and token list -- everything we need to *continue* training.
3. Generate the ``utt2num_samples`` shape files ESPnet's data loader needs
   for bucketing (one per train/dev split).
4. Construct an ``asr_train`` command line that points at our data, uses
   the bundle's tokenizer/BPE, initializes from the pretrained checkpoint
   (``--init_param``), and overrides the optimizer/scheduler/steps for our
   FT recipe.
5. Run that command as a subprocess, streaming its stdout/stderr live.
6. On success, copy/symlink the final checkpoint to ``output_dir``.

Compute target is the same as the Whisper / wav2vec 2.0 drivers: single CUDA
GPU, off-box. The script will run on a CPU host (it just won't make much
training progress).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import soundfile as sf
import yaml
from tqdm import tqdm


def _load_pretrained_bundle(model_id: str) -> dict:
    """Download and unpack the pretrained ESPnet model. Returns asset paths.

    ``espnet_model_zoo.ModelDownloader`` returns a dict whose keys include
    ``asr_train_config`` (the YAML we'll extend), ``asr_model_file`` (the
    checkpoint we'll initialize from), and the tokenizer artifacts. The
    download is cached so re-runs are fast.
    """
    # Imported here so this module remains import-cheap when ESPnet isn't
    # installed (matches the pattern in models/espnet_pretrained.py).
    from espnet_model_zoo.downloader import ModelDownloader

    print(f"[espnet_ft] downloading pretrained model: {model_id}")
    bundle = ModelDownloader().download_and_unpack(model_id)
    print(f"[espnet_ft] bundle keys: {sorted(bundle)}")
    return bundle


def _valid_asr_train_keys() -> set[str]:
    """Return the set of top-level argument names ``espnet2.bin.asr_train`` accepts.

    Used to filter unknown top-level keys out of the pretrained model's
    training config before passing it to asr_train. We get this by asking the
    ASRTask's argparse builder for the list of action destination names --
    this is what asr_train internally cross-references each config key
    against, so anything not in this set will be rejected as "unrecognized
    arguments".

    Importing ASRTask is slow (pulls in torch + the model graph code), so we
    do it lazily inside this function rather than at module top level.
    """
    from espnet2.tasks.asr import ASRTask

    parser = ASRTask.get_parser()
    return {a.dest for a in parser._actions if a.dest and a.dest != "help"}


# Keys in the pretrained bundle that describe its original multi-GPU
# distributed training setup. They're valid asr_train args (so the unknown-key
# filter doesn't catch them), but they carry the bundle's 8-GPU world size
# into our single-GPU FT run -- asr_train then sits in init_process_group
# waiting for 7 phantom workers and times out after 10 minutes. This driver
# is single-GPU by design, so we always strip these and let asr_train fall
# back to non-distributed defaults.
_DISTRIBUTED_MODE_KEYS = (
    "multiprocessing_distributed",
    "dist_world_size",
    "dist_rank",
    "dist_init_method",
    "dist_backend",
    "dist_master_addr",
    "dist_master_port",
    "dist_launcher",
    "sharded_ddp",
)


def _patch_pretrained_config(
    bundle_config_path: str,
    out_path: Path,
    ft_overrides: dict,
) -> Path:
    """Write a patched copy of the bundle's training config:

    1. Drop top-level keys the locally-installed ESPnet no longer recognizes
       (e.g. legacy ``distributed`` block from older ESPnet versions).
    2. Drop distributed-mode keys (``dist_world_size``, etc.) so the bundle's
       8-GPU setup doesn't bleed into our single-GPU FT.
    3. Apply our FT hyperparameter overrides INTO THE YAML rather than via
       CLI. This is the load-bearing step: ESPnet's ``--optim_conf`` and
       ``--scheduler_conf`` CLI args use ``NestedDictAction``, which in this
       version leaves nested values as raw strings (so ``lr=1e-05`` becomes
       the string ``"1e-05"``, crashing Adam's ``0.0 <= lr`` check with
       ``TypeError: '<=' not supported between instances of 'float' and 'str'``).
       Routing the overrides through YAML sidesteps the broken CLI parser:
       ``yaml.safe_dump`` writes proper types, and ``yaml.safe_load`` reads
       them back as floats/ints.

    For nested ``optim_conf``: we MERGE (preserve bundle defaults like ``eps``,
    ``betas``, ``weight_decay``; user wins on key collision). For ``scheduler``
    + ``scheduler_conf``: we REPLACE both atomically -- if the user changes
    scheduler type from ``noamlr`` to ``warmuplr``, the bundle's
    ``scheduler_conf`` keys (e.g. ``model_size``) become invalid for the new
    scheduler.
    """
    with open(bundle_config_path) as fh:
        cfg = yaml.safe_load(fh)
    valid = _valid_asr_train_keys()
    dropped_unknown = [k for k in list(cfg) if k not in valid]
    for key in dropped_unknown:
        cfg.pop(key)
    dropped_distributed = [k for k in _DISTRIBUTED_MODE_KEYS if k in cfg]
    for key in dropped_distributed:
        cfg.pop(key)
    applied_overrides = {}
    for key, value in ft_overrides.items():
        if key == "optim_conf" and isinstance(value, dict):
            existing = dict(cfg.get("optim_conf") or {})
            existing.update(value)
            cfg["optim_conf"] = existing
            applied_overrides[key] = existing
        else:
            cfg[key] = value
            applied_overrides[key] = value
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    if dropped_unknown:
        print(f"[espnet_ft] patched train config: dropped {len(dropped_unknown)} unrecognized "
              f"keys {dropped_unknown}")
    else:
        print("[espnet_ft] patched train config: no unrecognized keys to drop")
    if dropped_distributed:
        print(f"[espnet_ft] patched train config: dropped {len(dropped_distributed)} "
              f"distributed-mode keys (single-GPU driver) {dropped_distributed}")
    if applied_overrides:
        print(f"[espnet_ft] patched train config: applied FT overrides {applied_overrides}")
    return out_path


def _write_shape_file(data_dir: Path, out_path: Path) -> None:
    """Write a Kaldi-style 'utt_id num_samples' shape file from wav.scp.

    ESPnet's bucketing data loader needs per-utt sample counts. Reading the
    audio header via ``soundfile.info`` is fast (no full decode), so this is
    O(N) in number of utts -- a few seconds for 28k utts.
    """
    wav_scp = data_dir / "wav.scp"
    if not wav_scp.exists():
        raise FileNotFoundError(f"missing wav.scp in {data_dir}")

    lines = []
    with open(wav_scp, encoding="utf-8") as fh:
        scp_entries = [line.rstrip("\n").split(" ", 1) for line in fh if line.strip()]

    for utt_id, wav_path in tqdm(scp_entries, desc=f"shape({data_dir.name})"):
        info = sf.info(wav_path)
        n_samples = int(info.frames)
        lines.append(f"{utt_id} {n_samples}\n")

    out_path.write_text("".join(lines))


def _build_ft_overrides(cfg: dict) -> dict:
    """Pull the user's FT hyperparameters out of our YAML config and shape
    them into a dict that ``_patch_pretrained_config`` can merge into the
    bundle's training YAML.

    We route hyperparameters through the YAML rather than CLI because
    ESPnet's ``--optim_conf`` / ``--scheduler_conf`` CLI args don't type-
    convert nested values in this version (``lr=1e-05`` would stay a string
    and crash Adam's float bound check).
    """
    t = cfg.get("training", {})
    return {
        "max_epoch": int(t.get("max_epoch", 5)),
        "batch_bins": int(t.get("batch_bins", 5_000_000)),
        "num_workers": int(t.get("num_workers", 4)),
        "log_interval": int(t.get("log_interval", 50)),
        "seed": int(cfg.get("seed", 0)),
        "optim_conf": dict(t.get("optim_conf", {"lr": 1e-5})),
        "scheduler": t.get("scheduler", "warmuplr"),
        "scheduler_conf": dict(t.get("scheduler_conf", {"warmup_steps": 500})),
    }


def _build_asr_train_command(
    bundle: dict,
    train_dir: Path,
    valid_dir: Path,
    train_shape: Path,
    valid_shape: Path,
    output_dir: Path,
) -> list[str]:
    """Construct the espnet2.bin.asr_train command line for continued FT.

    Hyperparameters (optim_conf, scheduler*, max_epoch, etc.) are pre-baked
    into the patched YAML config -- see ``_patch_pretrained_config``. The
    CLI only carries args that are environment-specific (data paths, output
    dir, GPU count, distributed flags) or that argparse handles correctly
    (these accept simple int/str values, no nested-dict broken-parser hazard).
    """
    # NOTE on tokenizer assets: `token_list`, `token_type`, `bpemodel` are NOT
    # top-level bundle keys -- the ModelDownloader bundle only exposes
    # `asr_train_config` and `asr_model_file` (plus optional LM assets).
    # The tokenizer paths are embedded INSIDE asr_train_config, so passing
    # `--config <that yaml>` resolves them automatically. No CLI override
    # needed for the BPE/tokenizer.
    cmd: list[str] = [
        sys.executable, "-m", "espnet2.bin.asr_train",
        "--config", bundle["asr_train_config"],
        "--init_param", bundle["asr_model_file"],
        "--output_dir", str(output_dir),
        # Training and validation data (Kaldi style, "path,name,type" triplets).
        "--train_data_path_and_name_and_type", f"{train_dir/'wav.scp'},speech,sound",
        "--train_data_path_and_name_and_type", f"{train_dir/'text'},text,text",
        "--valid_data_path_and_name_and_type", f"{valid_dir/'wav.scp'},speech,sound",
        "--valid_data_path_and_name_and_type", f"{valid_dir/'text'},text,text",
        "--train_shape_file", str(train_shape),
        "--valid_shape_file", str(valid_shape),
        # GPU handling: ESPnet's asr_train defaults ngpu=1 if CUDA is available.
        "--ngpu", "1" if _cuda_available() else "0",
        # Force single-process / non-distributed regardless of what the
        # pretrained bundle's config says. The bundle was trained with
        # world_size=8; without these overrides, asr_train sits in
        # init_process_group waiting 10 min for 7 phantom workers on our
        # single-GPU pod. CLI args override YAML config values in argparse,
        # so this is robust to whatever the bundle config carries.
        "--multiprocessing_distributed", "false",
        "--dist_world_size", "1",
        "--dist_rank", "0",
    ]
    return cmd


def _cuda_available() -> bool:
    """True if torch sees a CUDA device. Used to decide --ngpu in the CLI."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _validate_data_dir(data_dir: Path) -> None:
    """Confirm the data dir has the Kaldi-style files ESPnet expects."""
    required = ["wav.scp", "text", "utt2spk", "spk2utt"]
    missing = [f for f in required if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"data_dir {data_dir} is missing required files: {missing}. "
            f"Run `python -m asr_robustness.train.espnet_render_data` first."
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fine-tune a pretrained ESPnet ASR model.")
    ap.add_argument("--config", required=True, help="training YAML config")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the asr_train command without executing it",
    )
    args = ap.parse_args(argv)

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    train_dir = Path(cfg["train_data_dir"])
    valid_dir = Path(cfg["eval_data_dir"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    _validate_data_dir(train_dir)
    _validate_data_dir(valid_dir)

    bundle = _load_pretrained_bundle(cfg["model_id"])

    # Patch the bundle's training config: strip version-drift keys, strip
    # distributed-mode keys (single-GPU driver), and bake our FT hyperparams
    # in via YAML rather than relying on broken CLI nested-dict parsing.
    patched_train_config = output_dir / "patched_train_config.yaml"
    ft_overrides = _build_ft_overrides(cfg)
    _patch_pretrained_config(
        bundle["asr_train_config"], patched_train_config, ft_overrides
    )
    bundle = dict(bundle)  # don't mutate the cached bundle dict
    bundle["asr_train_config"] = str(patched_train_config)

    # Shape files needed by ESPnet's bucketing dataloader.
    train_shape = output_dir / "train_shape.scp"
    valid_shape = output_dir / "valid_shape.scp"
    if not train_shape.exists():
        _write_shape_file(train_dir, train_shape)
    if not valid_shape.exists():
        _write_shape_file(valid_dir, valid_shape)

    cmd = _build_asr_train_command(
        bundle, train_dir, valid_dir, train_shape, valid_shape, output_dir
    )

    print("\n[espnet_ft] command to invoke:")
    for token in cmd:
        print(f"    {token}")
    print()

    if args.dry_run:
        print("[espnet_ft] --dry-run: not executing.")
        return 0

    # Stream the subprocess output live -- ESPnet's training loop logs to
    # stderr; we forward both so tqdm bars and per-step metrics show up in
    # the user's terminal in real time.
    proc = subprocess.run(cmd, env=os.environ)
    if proc.returncode != 0:
        print(f"\n[espnet_ft] asr_train exited with code {proc.returncode}", file=sys.stderr)
        return proc.returncode

    print(f"\n[espnet_ft] training complete -- output in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
