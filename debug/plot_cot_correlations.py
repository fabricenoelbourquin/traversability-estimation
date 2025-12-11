#!/usr/bin/env python3
"""
Quick correlation checks between patch-level COT and other features.

Plots COT vs distance_traveled_m and COT vs speed_mean, and prints Pearson r.
Intended to verify that these variables are not strongly correlated with COT.
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
DEFAULT_REPORT_DIR = Path(get_paths()["REPO_ROOT"]) / "reports" / "zz_cot_correlation_checks"

COT_COL = "cot_patch"
DIST_COL = "distance_traveled_m"
SPEED_COL = "speed_mean"


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


def _finite_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description="Check correlation of COT with distance and speed.")
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
        "--output",
        type=Path,
        default=None,
        help="Output path for figure (default: reports/zz_cot_correlation_checks/cot_correlations.png).",
    )
    ap.add_argument(
        "--gridsize",
        type=int,
        default=50,
        help="Hexbin grid size (default: 50).",
    )
    args = ap.parse_args()

    dataset_path = args.dataset if args.dataset is not None else _default_dataset_path(args.patch_size)
    base_out = args.output if args.output is not None else DEFAULT_REPORT_DIR / "cot_correlations.png"

    dfs = _load_patch_groups(dataset_path, args.missions)
    if not dfs:
        raise SystemExit("No missions found in dataset (after filtering).")

    required = [COT_COL, DIST_COL, SPEED_COL]
    for r in required:
        if any(r not in df.columns for df in dfs):
            raise SystemExit(f"Required column '{r}' missing in dataset.")

    cot = np.concatenate([df[COT_COL].to_numpy(dtype=np.float64) for df in dfs])
    dist = np.concatenate([df[DIST_COL].to_numpy(dtype=np.float64) for df in dfs])
    speed = np.concatenate([df[SPEED_COL].to_numpy(dtype=np.float64) for df in dfs])

    cot_dist, dist_f = _finite_pair(cot, dist)
    cot_speed, speed_f = _finite_pair(cot, speed)

    if cot_dist.size == 0 or cot_speed.size == 0:
        raise SystemExit("No finite COT/distance/speed data to plot.")

    r_dist = _pearson(cot_dist, dist_f)
    r_speed = _pearson(cot_speed, speed_f)
    print(f"[info] Pearson r (cot vs distance_traveled_m): {r_dist:.4f} (n={cot_dist.size})")
    print(f"[info] Pearson r (cot vs speed_mean): {r_speed:.4f} (n={cot_speed.size})")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    hb0 = axes[0].hexbin(dist_f, cot_dist, gridsize=args.gridsize, cmap="viridis", mincnt=1, linewidths=0.0)
    axes[0].set_xlabel("distance_traveled_m")
    axes[0].set_ylabel("cot_patch")
    axes[0].set_title(f"COT vs distance (r={r_dist:.2f})")
    cb0 = fig.colorbar(hb0, ax=axes[0])
    cb0.set_label("patch count")
    axes[0].grid(alpha=0.25)

    hb1 = axes[1].hexbin(speed_f, cot_speed, gridsize=args.gridsize, cmap="viridis", mincnt=1, linewidths=0.0)
    axes[1].set_xlabel("speed_mean")
    axes[1].set_ylabel("cot_patch")
    axes[1].set_title(f"COT vs speed_mean (r={r_speed:.2f})")
    cb1 = fig.colorbar(hb1, ax=axes[1])
    cb1.set_label("patch count")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    base_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_out, dpi=200)
    plt.close(fig)
    print(f"[ok] wrote {base_out}")


if __name__ == "__main__":
    main()
