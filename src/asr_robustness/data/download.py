"""Download and extract the archive-based datasets.

Usage::

    python -m asr_robustness.data.download --datasets librispeech musan rirs
    python -m asr_robustness.data.download --minimal          # smoke-test subset
    python -m asr_robustness.data.download --list             # show the registry

The operation is **idempotent**: an archive whose extracted directory already
exists is skipped, so ``make data`` can be re-run safely. ``hf`` and ``manual``
datasets cannot be auto-downloaded -- their access instructions are printed.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

from asr_robustness.data.sources import (
    DATASETS,
    MINIMAL_LIBRISPEECH_SUBSETS,
    Archive,
)


def _download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` with a progress bar (atomic via .part file)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp:  # noqa: S310 - trusted dataset hosts
        total = int(resp.headers.get("Content-Length", 0))
        with open(tmp, "wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)


def _verify_md5(path: Path, expected: str) -> None:
    h = hashlib.md5()  # noqa: S324 - integrity check, not security
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected:
        raise ValueError(f"checksum mismatch for {path.name}: {actual} != {expected}")


def _extract(archive_path: Path, root: Path) -> None:
    """Extract a .tar.gz or .zip archive into ``root``."""
    if archive_path.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive_path, "r:gz") as tf:
            try:
                # filter="data" guards against path-escape; needs Python >= 3.11.4
                tf.extractall(root, filter="data")
            except TypeError:
                tf.extractall(root)
    elif archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(root)
    else:
        raise ValueError(f"unsupported archive type: {archive_path.name}")


def fetch(archive: Archive, root: Path, keep_archive: bool = True, force: bool = False) -> Path:
    """Download + extract one :class:`Archive`; returns its extracted directory."""
    extracted = root / archive.member_root
    if extracted.exists() and not force:
        print(f"  [skip]     {archive.name}: already extracted at {extracted}")
        return extracted

    local = root / "_archives" / archive.url.rsplit("/", 1)[-1]
    if not local.exists() or force:
        print(f"  [download] {archive.name} <- {archive.url}")
        _download(archive.url, local)
    if archive.md5:
        _verify_md5(local, archive.md5)
    print(f"  [extract]  {archive.name} -> {extracted}")
    _extract(local, root)
    if not keep_archive:
        local.unlink(missing_ok=True)
    return extracted


def fetch_dataset(
    name: str, root: Path, subsets: list[str] | None, force: bool = False
) -> None:
    """Fetch every archive of dataset ``name`` (optionally restricted to ``subsets``)."""
    dataset = DATASETS[name]
    print(f"[{name}] kind={dataset.kind}")
    if dataset.kind != "archive":
        print(f"  manual step required:\n    {dataset.note}")
        return
    selected = dataset.archives
    if subsets:
        selected = {k: v for k, v in dataset.archives.items() if k in subsets}
        missing = set(subsets) - set(dataset.archives)
        if missing:
            print(f"  warning: unknown subsets ignored: {sorted(missing)}")
    for archive in selected.values():
        fetch(archive, root, force=force)


def _print_registry() -> None:
    for name, ds in DATASETS.items():
        print(f"{name:14s} kind={ds.kind}")
        for sub in ds.archives:
            print(f"               - {sub}")
        if ds.note:
            print(f"               note: {ds.note}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download project datasets.")
    ap.add_argument("--root", default="data", help="data directory (default: data/)")
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=["librispeech", "musan", "rirs"],
        help="datasets to fetch",
    )
    ap.add_argument("--subsets", nargs="+", help="restrict LibriSpeech subsets")
    ap.add_argument("--minimal", action="store_true", help="smallest subset, for smoke tests")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    ap.add_argument("--list", action="store_true", help="print the dataset registry and exit")
    args = ap.parse_args(argv)

    if args.list:
        _print_registry()
        return 0

    root = Path(args.root)
    subsets = MINIMAL_LIBRISPEECH_SUBSETS if args.minimal else args.subsets
    for name in args.datasets:
        if name not in DATASETS:
            print(f"unknown dataset: {name} (known: {sorted(DATASETS)})", file=sys.stderr)
            return 2
        ls_subsets = subsets if name == "librispeech" else None
        fetch_dataset(name, root, ls_subsets, force=args.force)
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
