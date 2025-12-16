#!/usr/bin/env python3
"""
Merge two HDF5 dataset files, preferring missions/groups from the first file when duplicates exist.

Paths can be absolute or bare dataset names; bare names are resolved against the dataset
directory from utils/paths.get_paths().
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, List

try:
    import h5py  # type: ignore
except ImportError as e:  # pragma: no cover - handled at runtime
    raise SystemExit("h5py is required to merge datasets (pip install h5py).") from e

from utils.paths import get_paths


def resolve_dataset_path(raw: str, datasets_root: Path, *, must_exist: bool = False) -> Path:
    """
    Interpret `raw` as either a full path or a dataset filename under datasets_root.
    If must_exist is True, also try appending .h5 for suffix-less names before failing.
    """
    p = Path(raw)
    if not p.is_absolute() and p.parent == Path("."):
        p = datasets_root / p
    if must_exist and not p.exists() and p.suffix == "":
        alt = p.with_suffix(".h5")
        if alt.exists():
            p = alt
    if must_exist and not p.exists():
        raise SystemExit(f"HDF5 file not found: {p}")
    return p


def copy_items(src: h5py.File, dst: h5py.File, *, skip_existing: bool) -> Tuple[List[str], List[str]]:
    added: List[str] = []
    skipped: List[str] = []
    for name in src.keys():
        if skip_existing and name in dst:
            skipped.append(name)
            continue
        src.copy(src[name], dst, name=name)
        added.append(name)
    return added, skipped


def merge_files(first: Path, second: Path, output: Path, *, overwrite: bool = False) -> None:
    if output.exists():
        if not overwrite:
            raise SystemExit(f"Output file already exists: {output} (use --overwrite to replace)")
        if output.is_dir():
            raise SystemExit(f"Output path is a directory: {output}")
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(first, "r") as f1, h5py.File(second, "r") as f2, h5py.File(output, "w") as fout:
        # Preserve file-level attrs from the first file and only fill missing from the second.
        for k, v in f1.attrs.items():
            fout.attrs[k] = v
        for k, v in f2.attrs.items():
            fout.attrs.setdefault(k, v)

        added_first, _ = copy_items(f1, fout, skip_existing=False)
        added_second, skipped = copy_items(f2, fout, skip_existing=True)

    print(f"[ok] merged -> {output}")
    print(f" - copied {len(added_first)} groups from first: {first}")
    print(f" - copied {len(added_second)} groups from second (skipped {len(skipped)} duplicates): {second}")
    if skipped:
        print(f"   duplicates skipped (kept first): {', '.join(skipped)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge two HDF5 dataset files (prefer missions from the first file).")
    ap.add_argument("first", help="First HDF5 file (preferred when missions overlap).")
    ap.add_argument("second", help="Second HDF5 file.")
    ap.add_argument("-o", "--output", default="merged.h5",
                    help="Output path or filename. Bare names are saved under the datasets directory.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite the output file if it already exists.")
    args = ap.parse_args()

    paths = get_paths()
    datasets_root = Path(paths["DATASETS"])

    first = resolve_dataset_path(args.first, datasets_root, must_exist=True)
    second = resolve_dataset_path(args.second, datasets_root, must_exist=True)
    output = resolve_dataset_path(args.output, datasets_root, must_exist=False)

    out_resolved = output.resolve()
    if out_resolved in {first.resolve(), second.resolve()}:
        raise SystemExit("Output path must differ from both input files.")

    merge_files(first, second, output, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
