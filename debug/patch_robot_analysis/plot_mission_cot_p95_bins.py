#!/usr/bin/env python3
"""
Visualize which missions have cot_patch_p95 values in specified bins.

Bins:
- > 5
- 4 to 5 (inclusive)
- 3 to 4 (inclusive of 3, exclusive of 4)
- all < 3
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make src/ importable when running from repo root
import sys
THIS_FILE = Path(__file__).resolve()


def _resolve_repo_root(file_path: Path) -> Path:
    for parent in file_path.parents:
        if (parent / "src").exists():
            return parent
    raise SystemExit("Could not find repository root (missing 'src' directory).")


REPO_ROOT = _resolve_repo_root(THIS_FILE)
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.paths import get_paths


DEFAULT_PATCH_SIZE_M: float = 5.0
DEFAULT_REPORT_DIR = Path(get_paths()["REPO_ROOT"]) / "reports" / "zz_patch_analysis_robot_data"

COT_P95_COL = "cot_patch_p95"
MISSION_COL = "mission_display"


def _patch_label(patch_size_m: float | None) -> str:
    size = DEFAULT_PATCH_SIZE_M if patch_size_m is None else patch_size_m
    label_num = f"{size:.3f}".rstrip("0").rstrip(".")
    return f"{label_num}m"


def _decode_attr_val(val):
    if isinstance(val, (bytes, bytearray)):
        try:
            return val.decode("utf-8")
        except Exception:
            return val
    return val


def _default_dataset_path(patch_size_m: float | None) -> Path:
    size = DEFAULT_PATCH_SIZE_M if patch_size_m is None else patch_size_m
    label = f"{size:.3f}".rstrip("0").rstrip(".")
    return Path(get_paths()["DATASETS"]) / f"patches_{label}m.h5"


def _resolve_dataset_path(dataset_arg: Path | None, patch_size_m: float | None) -> Path:
    if dataset_arg is None:
        return _default_dataset_path(patch_size_m)
    dataset_path = Path(dataset_arg)
    if dataset_path.is_absolute():
        return dataset_path
    if dataset_path.parent == Path("."):
        if dataset_path.suffix == "":
            dataset_path = dataset_path.with_suffix(".h5")
        return Path(get_paths()["DATASETS"]) / dataset_path.name
    return dataset_path


def _load_patch_groups(h5_path: Path, missions: Sequence[str] | None) -> list[pd.DataFrame]:
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise SystemExit("h5py is required to read the patch dataset (pip install h5py).") from exc

    if not h5_path.exists():
        raise SystemExit(f"Dataset not found: {h5_path}")

    requested = set(missions) if missions else None
    dfs: list[pd.DataFrame] = []
    with h5py.File(h5_path, "r") as f:
        for grp_name in sorted(f.keys()):
            grp = f[grp_name]
            attrs = {k: _decode_attr_val(v) for k, v in grp.attrs.items()}
            display = str(attrs.get("mission_display") or grp_name)
            if requested and grp_name not in requested and display not in requested:
                continue
            if "patches" not in grp:
                continue
            ds = grp["patches"]
            records = ds[:]
            df = pd.DataFrame.from_records(records)
            df.columns = [c.decode("utf-8") if isinstance(c, (bytes, bytearray)) else str(c) for c in df.columns]
            col_order_attr = ds.attrs.get("column_order")
            col_order: list[str] = []
            if isinstance(col_order_attr, (bytes, str)):
                try:
                    col_order = list(json.loads(col_order_attr))
                except Exception:
                    col_order = []
            if col_order:
                missing_cols = [c for c in col_order if c not in df.columns]
                if not missing_cols:
                    df = df[col_order]
            df["mission_display"] = display
            dfs.append(df)
    if requested:
        missing = requested - {d["mission_display"].iloc[0] for d in dfs} if dfs else requested
        if missing:
            print(f"[warn] Requested missions not found in dataset: {sorted(missing)}")
    return dfs


def _bin_flags(values: np.ndarray) -> tuple[bool, bool, bool, bool]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False, False, False, False
    gt_5 = np.any(finite > 5.0)
    in_4_5 = np.any((finite >= 4.0) & (finite <= 5.0))
    in_3_4 = np.any((finite >= 3.0) & (finite < 4.0))
    all_lt_3 = np.all(finite < 3.0)
    return gt_5, in_4_5, in_3_4, all_lt_3


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize missions by cot_patch_p95 bins.")
    ap.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=(
            "Path to patch HDF5 dataset. If a bare filename is provided, it is"
            " resolved under DATASETS (default: DATASETS/patches_<patch-size>m.h5)."
        ),
    )
    ap.add_argument(
        "--patch-size",
        type=float,
        default=None,
        help=f"Patch size to build default dataset path (meters, default: {DEFAULT_PATCH_SIZE_M}).",
    )
    ap.add_argument(
        "--missions",
        nargs="*",
        default=None,
        help="Optional mission ids/displays to include (default: all missions in the dataset).",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for figure (default: reports/zz_patch_analysis_robot_data/<patch>/mission_cot_p95_bins.png).",
    )
    args = ap.parse_args()

    dataset_path = _resolve_dataset_path(args.dataset, args.patch_size)
    patch_label = _patch_label(args.patch_size)
    default_out = DEFAULT_REPORT_DIR / patch_label / "mission_cot_p95_bins.png"

    dfs = _load_patch_groups(dataset_path, args.missions)
    if not dfs:
        raise SystemExit("No missions found in dataset (after filtering).")

    required = {COT_P95_COL, MISSION_COL}
    if any(not required.issubset(set(df.columns)) for df in dfs):
        raise SystemExit(f"Required columns missing: need '{COT_P95_COL}' and '{MISSION_COL}'.")

    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all[[MISSION_COL, COT_P95_COL]]
    df_all = df_all[df_all[MISSION_COL].notna()]
    if df_all.empty:
        raise SystemExit("No mission/cot_patch_p95 data to plot.")

    missions = sorted(df_all[MISSION_COL].astype(str).unique().tolist())
    bins = [">5", "4-5", "3-4", "all<3"]
    matrix = np.zeros((len(missions), len(bins)), dtype=int)
    no_data: list[str] = []

    for i, mission in enumerate(missions):
        vals = df_all.loc[df_all[MISSION_COL].astype(str) == mission, COT_P95_COL].to_numpy(dtype=np.float64)
        flags = _bin_flags(vals)
        if not any(flags):
            no_data.append(mission)
        matrix[i, :] = np.array(flags, dtype=int)

    def _print_list(label: str, mask: np.ndarray) -> None:
        items = [m for m, ok in zip(missions, mask) if ok]
        print(f"[info] {label} ({len(items)}): {items}")

    _print_list("missions with >5", matrix[:, 0].astype(bool))
    _print_list("missions with 4-5", matrix[:, 1].astype(bool))
    _print_list("missions with 3-4", matrix[:, 2].astype(bool))
    _print_list("missions all <3", matrix[:, 3].astype(bool))
    if no_data:
        print(f"[info] missions with no finite {COT_P95_COL} data ({len(no_data)}): {no_data}")

    height = max(4.0, 0.35 * len(missions))
    fig, ax = plt.subplots(figsize=(6.5, height))
    cmap = plt.cm.get_cmap("Blues")
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(bins)))
    ax.set_xticklabels(bins)
    ax.set_yticks(np.arange(len(missions)))
    ax.set_yticklabels(missions)
    ax.set_xlabel("cot_patch_p95 bins")
    ax.set_ylabel("mission")
    ax.set_title("Mission membership by cot_patch_p95 bins")
    ax.set_xticks(np.arange(-0.5, len(bins), 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, len(missions), 1.0), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6)
    ax.axvline(2.5, color="red", linewidth=2.0)
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()

    out_path = args.output if args.output is not None else default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
