#!/usr/bin/env python3
"""
Plot mean cot_patch_p95 for missions that have at least one cot_patch_p95 > threshold.

By default, threshold is 3.0 and the mean is computed over all finite cot_patch_p95
values in each qualifying mission.
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

COT_COL = "cot_patch_p95"
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot mean cot_patch_p95 for missions with any cot_patch_p95 > threshold.")
    ap.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to patch HDF5 dataset (default: DATASETS/patches_<patch-size>m.h5).",
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
        "--threshold",
        type=float,
        default=3.0,
        help="Threshold for selecting missions (default: 3.0).",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for figure (default: reports/zz_patch_analysis_robot_data/<patch>/mission_mean_cot_p95_gt3.png).",
    )
    args = ap.parse_args()

    dataset_path = args.dataset if args.dataset is not None else _default_dataset_path(args.patch_size)
    patch_label = _patch_label(args.patch_size)
    default_out = DEFAULT_REPORT_DIR / patch_label / "mission_mean_cot_p95_gt3.png"

    dfs = _load_patch_groups(dataset_path, args.missions)
    if not dfs:
        raise SystemExit("No missions found in dataset (after filtering).")

    required = {COT_COL, MISSION_COL}
    if any(not required.issubset(set(df.columns)) for df in dfs):
        raise SystemExit(f"Required columns missing: need '{COT_COL}' and '{MISSION_COL}'.")

    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all[[MISSION_COL, COT_COL]]
    df_all = df_all[df_all[MISSION_COL].notna()]
    df_all[COT_COL] = pd.to_numeric(df_all[COT_COL], errors="coerce")
    df_all = df_all[np.isfinite(df_all[COT_COL])]
    if df_all.empty:
        raise SystemExit("No finite mission/cot_patch data to plot.")

    mission_stats: list[tuple[str, float, int]] = []
    for mission, grp in df_all.groupby(df_all[MISSION_COL].astype(str)):
        cot_vals = grp[COT_COL].to_numpy(dtype=np.float64)
        if np.any(cot_vals > args.threshold):
            mean_val = float(np.mean(cot_vals))
            mission_stats.append((mission, mean_val, len(cot_vals)))

    if not mission_stats:
        raise SystemExit(f"No missions found with any {COT_COL} > {args.threshold}.")

    mission_stats.sort(key=lambda x: x[1], reverse=True)
    missions = [m for m, _, _ in mission_stats]
    means = np.array([m for _, m, _ in mission_stats], dtype=np.float64)
    counts = [n for _, _, n in mission_stats]

    print(f"[info] Dataset: {dataset_path}")
    print(f"[info] Threshold: {args.threshold}")
    for mission, mean_val, n in mission_stats:
        print(f"[info] {mission}: mean {COT_COL}={mean_val:.4g} (n={n})")

    height = max(4.0, 0.35 * len(missions))
    fig, ax = plt.subplots(figsize=(7.0, height))
    y = np.arange(len(missions))
    ax.barh(y, means, color="tab:blue", alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(missions)
    ax.invert_yaxis()
    ax.set_xlabel(f"mean {COT_COL}")
    ax.set_ylabel("mission")
    ax.set_title(f"Mean {COT_COL} for missions with any {COT_COL} > {args.threshold:g}")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    out_path = args.output if args.output is not None else default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
