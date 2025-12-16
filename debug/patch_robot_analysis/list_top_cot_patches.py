#!/usr/bin/env python3
"""
List the patches with the highest cot_patch values in a patch HDF5 dataset.

For each of the top-N patches (default: 5), prints mission, cot_patch, time span,
distance traveled, and mean speeds (actual + commanded when available).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import yaml

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
COT_COL = "cot_patch"
DATASET_CFG = _resolve_repo_root(THIS_FILE) / "config" / "dataset.yaml"


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


def _finite(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _load_selection_offsets(cfg_path: Path) -> dict[str, float]:
    """Return per-mission time offsets from selection.time_ranges_s (min start per mission)."""
    if not cfg_path.exists():
        return {}
    try:
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
    except Exception:
        return {}
    sel = (cfg.get("selection") or {}).get("time_ranges_s") or {}
    offsets: dict[str, float] = {}
    for mission, ranges in sel.items():
        try:
            starts = [float(r[0]) for r in ranges if isinstance(r, (list, tuple)) and len(r) == 2]
        except Exception:
            starts = []
        if starts:
            offsets[str(mission)] = float(np.nanmin(starts))
    return offsets


def _format_table(df: pd.DataFrame, cols: list[str]) -> str:
    df_fmt = df.copy()
    for c in cols:
        if c not in df_fmt.columns:
            df_fmt[c] = math.nan
    df_fmt = df_fmt[cols]
    # compact numeric formatting
    with pd.option_context("display.max_rows", None, "display.width", 160):
        return df_fmt.to_string(index=False, float_format=lambda x: f"{x:.4g}")


def main() -> None:
    ap = argparse.ArgumentParser(description="List top-N patches by cot_patch with timing/speed context.")
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
        "--top",
        type=int,
        default=5,
        help="Number of top patches to list (default: 5).",
    )
    ap.add_argument(
        "--min",
        action="store_true",
        help="List the lowest cot_patch values instead of the highest.",
    )
    args = ap.parse_args()

    dataset_path = args.dataset if args.dataset is not None else _default_dataset_path(args.patch_size)
    dfs = _load_patch_groups(dataset_path, args.missions)
    if not dfs:
        raise SystemExit("No missions found in dataset (after filtering).")

    selection_offsets = _load_selection_offsets(DATASET_CFG)

    df_all = pd.concat(dfs, ignore_index=True)
    if COT_COL not in df_all.columns:
        raise SystemExit(f"Column '{COT_COL}' not found in dataset.")

    df_all = df_all[np.isfinite(df_all[COT_COL])]
    if df_all.empty:
        raise SystemExit(f"No finite values in column '{COT_COL}'.")

    # Compute mission-relative times so we show patch timing offset from mission start.
    if "t_start" in df_all.columns and "mission_display" in df_all.columns:
        baselines: dict[str, float] = {}
        for mission, grp in df_all.groupby("mission_display"):
            ts = _finite(grp["t_start"].to_numpy(dtype=np.float64))
            baselines[mission] = float(np.nanmin(ts)) if ts.size else math.nan
    else:
        baselines = {}

    df_sorted = df_all.sort_values(COT_COL, ascending=args.min).head(max(args.top, 1))

    if baselines:
        def _rel(val: float, mission: str) -> float:
            base = baselines.get(mission, math.nan)
            if not (np.isfinite(val) and np.isfinite(base)):
                return math.nan
            offset = selection_offsets.get(mission, 0.0)
            return float(val - base + offset)

        df_sorted["t_start_rel_s"] = [
            _rel(row.get("t_start", math.nan), row.get("mission_display", "")) for _, row in df_sorted.iterrows()
        ]
        df_sorted["t_end_rel_s"] = [
            _rel(row.get("t_end", math.nan), row.get("mission_display", "")) for _, row in df_sorted.iterrows()
        ]

    cols_preferred = [
        "mission_display",
        COT_COL,
        "cot_patch_p95",
        "t_start_rel_s",
        "t_end_rel_s",
        "time_span_s",
        "distance_traveled_m",
        "speed_mean",
        "v_cmd_mean",
        "pitch_deg_mean",
    ]
    cols_present = [c for c in cols_preferred if c in df_sorted.columns]
    table = _format_table(df_sorted, cols_present)
    print(f"[info] Dataset: {dataset_path}")
    print(f"[info] Total patches considered: {len(df_all)}")
    print(f"[info] Showing top {len(df_sorted)} by '{COT_COL}':")
    print(table)


if __name__ == "__main__":
    main()
