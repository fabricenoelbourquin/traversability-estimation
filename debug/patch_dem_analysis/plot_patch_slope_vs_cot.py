#!/usr/bin/env python3
"""
Plot patch mean slope magnitude (deg) vs patch mean cost_of_transport from the
HDF5 dataset produced by build_patch_dataset.py.
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
DEFAULT_REPORT_DIR = Path(get_paths()["REPO_ROOT"]) / "reports" / "zz_incline_patch_analysis"

SLOPE_COL = "slope_mag_mean"
COT_COL = "metric_cost_of_transport_mean"


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


def _slope_to_deg(values: np.ndarray) -> np.ndarray:
    return np.rad2deg(np.arctan(values.astype(np.float64)))


def _prepare(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if SLOPE_COL not in df.columns or COT_COL not in df.columns:
        raise SystemExit(f"Required columns missing: need '{SLOPE_COL}' and '{COT_COL}'.")
    slope = _slope_to_deg(df[SLOPE_COL].to_numpy(dtype=np.float64))
    cot = df[COT_COL].to_numpy(dtype=np.float64)
    mask = np.isfinite(slope) & np.isfinite(cot)
    return slope[mask], cot[mask]


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    if x.size < 2 or y.size < 2:
        return None
    coeffs = np.polyfit(x, y, 1)
    return float(coeffs[0]), float(coeffs[1])


def _remove_outliers(x: np.ndarray, y: np.ndarray, pct_low: float = 2.0, pct_high: float = 98.0) -> tuple[np.ndarray, np.ndarray]:
    if x.size == 0 or y.size == 0:
        return x, y
    x_low, x_high = np.percentile(x, [pct_low, pct_high])
    y_low, y_high = np.percentile(y, [pct_low, pct_high])
    mask = (x >= x_low) & (x <= x_high) & (y >= y_low) & (y <= y_high)
    return x[mask], y[mask]


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot mean slope magnitude (deg) vs mean cost_of_transport for patches.")
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
        help="Output path for figure (default: reports/zz_incline_patch_analysis/patch_slope_vs_cot.png).",
    )
    ap.add_argument(
        "--gridsize",
        type=int,
        default=50,
        help="Hexbin grid size (default: 50).",
    )
    ap.add_argument(
        "--y-range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=None,
        help="Optional y-axis range (cost_of_transport). When set, saves both unrestricted and restricted plots.",
    )
    args = ap.parse_args()

    dataset_path = args.dataset if args.dataset is not None else _default_dataset_path(args.patch_size)
    dfs = _load_patch_groups(dataset_path, args.missions)
    if not dfs:
        raise SystemExit("No missions found in dataset (after filtering).")

    slope_all: list[np.ndarray] = []
    cot_all: list[np.ndarray] = []
    for df in dfs:
        s, c = _prepare(df)
        slope_all.append(s)
        cot_all.append(c)

    slope_deg = _finite(np.concatenate(slope_all)) if slope_all else np.array([])
    cot_vals = _finite(np.concatenate(cot_all)) if cot_all else np.array([])
    if slope_deg.size == 0 or cot_vals.size == 0:
        raise SystemExit("No finite slope/cost_of_transport data to plot.")

    fit_all = _fit_line(slope_deg, cot_vals)
    slope_nr, cot_nr = _remove_outliers(slope_deg, cot_vals)
    fit_no_outliers = _fit_line(slope_nr, cot_nr)
    x_plot = np.linspace(np.min(slope_deg), np.max(slope_deg), 100) if slope_deg.size else np.array([])

    def _plot(y_range: tuple[float, float] | None, suffix: str) -> Path:
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        hb = ax.hexbin(
            slope_deg,
            cot_vals,
            gridsize=args.gridsize,
            cmap="viridis",
            mincnt=1,
            linewidths=0.0,
        )
        if y_range is not None:
            ax.set_ylim(y_range)
        ax.set_xlabel("slope_mag_mean [deg]")
        ax.set_ylabel("metric_cost_of_transport_mean")
        title_suffix = "" if y_range is None else f" (y∈[{y_range[0]}, {y_range[1]}])"
        ax.set_title(f"Patch mean slope vs mean cost_of_transport{title_suffix}")
        if x_plot.size and fit_all is not None:
            ax.plot(x_plot, fit_all[0] * x_plot + fit_all[1], color="tab:red", lw=1.4, label="fit (all)")
        if x_plot.size and fit_no_outliers is not None:
            ax.plot(x_plot, fit_no_outliers[0] * x_plot + fit_no_outliers[1], color="tab:orange", lw=1.4, linestyle="--", label="fit (no outliers)")
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("patch count")
        ax.grid(alpha=0.25)
        fig.tight_layout()

        base_out = args.output if args.output is not None else DEFAULT_REPORT_DIR / "patch_slope_vs_cot.png"
        out_path = base_out if suffix == "" else base_out.with_name(f"{base_out.stem}{suffix}{base_out.suffix}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"[ok] wrote {out_path}")
        return out_path

    _plot(None, "")
    if args.y_range is not None:
        _plot((args.y_range[0], args.y_range[1]), "_restricted")


if __name__ == "__main__":
    main()
