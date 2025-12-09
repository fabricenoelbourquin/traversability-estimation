#!/usr/bin/env python3
"""
Plot patch forward-oriented slope (deg) vs patch mean cost_of_transport.

Forward slope is computed from slope_e_mean / slope_n_mean projected onto the
robot's mean heading (bearing quaternion in the patch dataset).
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

SLOPE_E_COL = "slope_e_mean"
SLOPE_N_COL = "slope_n_mean"
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


def _patch_label(patch_size_m: float | None) -> str:
    size = DEFAULT_PATCH_SIZE_M if patch_size_m is None else patch_size_m
    label_num = f"{size:.3f}".rstrip("0").rstrip(".")
    return f"{label_num}m"


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


def _normalize_quat_arrays(qw: np.ndarray, qx: np.ndarray, qy: np.ndarray, qz: np.ndarray):
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    n[n == 0.0] = 1.0
    return qw / n, qx / n, qy / n, qz / n


def _get_quaternion_block(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    candidates = [
        ("bearing_qw", "bearing_qx", "bearing_qy", "bearing_qz"),
        ("qw_WB", "qx_WB", "qy_WB", "qz_WB"),
        ("qw", "qx", "qy", "qz"),
    ]
    for cols in candidates:
        if all(c in df.columns for c in cols):
            return tuple(df[c].to_numpy(dtype=np.float64) for c in cols)  # type: ignore
    raise SystemExit("Quaternion columns not found (need bearing_qw..qz or qw_WB..qz_WB or qw..qz).")


def _yaw_deg_from_quat(df: pd.DataFrame) -> np.ndarray:
    qw, qx, qy, qz = _get_quaternion_block(df)
    qw, qx, qy, qz = _normalize_quat_arrays(qw, qx, qy, qz)

    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    r00 = 1.0 - 2.0 * (yy + zz)
    r10 = 2.0 * (xy + wz)

    yaw = np.arctan2(r10, r00)
    yaw_deg = np.rad2deg(yaw) * -1.0  # navigation-friendly (north=0, clockwise positive)
    return yaw_deg


def _finite(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _prepare(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    missing = [c for c in (SLOPE_E_COL, SLOPE_N_COL, COT_COL) if c not in df.columns]
    if missing:
        raise SystemExit(f"Required columns missing: {missing}")
    yaw_deg = _yaw_deg_from_quat(df)
    heading_rad = np.deg2rad(yaw_deg)

    slope_e = df[SLOPE_E_COL].to_numpy(dtype=np.float64)
    slope_n = df[SLOPE_N_COL].to_numpy(dtype=np.float64)

    # Forward component: project (slope_e, slope_n) onto heading (east=north basis)
    dir_e = np.sin(heading_rad)
    dir_n = np.cos(heading_rad)
    slope_forward = slope_e * dir_e + slope_n * dir_n
    slope_forward_deg = np.rad2deg(np.arctan(slope_forward))

    cot = df[COT_COL].to_numpy(dtype=np.float64)
    mask = np.isfinite(slope_forward_deg) & np.isfinite(cot)
    return slope_forward_deg[mask], cot[mask]


def _fit_quad(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float] | None:
    if x.size < 3 or y.size < 3:
        return None
    coeffs = np.polyfit(x, y, 2)
    return float(coeffs[0]), float(coeffs[1]), float(coeffs[2])


def _remove_outliers(x: np.ndarray, y: np.ndarray, pct_low: float = 2.0, pct_high: float = 98.0) -> tuple[np.ndarray, np.ndarray]:
    if x.size == 0 or y.size == 0:
        return x, y
    x_low, x_high = np.percentile(x, [pct_low, pct_high])
    y_low, y_high = np.percentile(y, [pct_low, pct_high])
    mask = (x >= x_low) & (x <= x_high) & (y >= y_low) & (y <= y_high)
    return x[mask], y[mask]


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot forward-oriented slope (deg) vs mean cost_of_transport for patches.")
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
        help="Output path for figure (default: reports/zz_incline_patch_analysis/patch_oriented_slope_vs_cot.png).",
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
    patch_label = _patch_label(args.patch_size)
    default_out = DEFAULT_REPORT_DIR / patch_label / "patch_oriented_slope_vs_cot.png"
    dfs = _load_patch_groups(dataset_path, args.missions)
    if not dfs:
        raise SystemExit("No missions found in dataset (after filtering).")

    slope_all: list[np.ndarray] = []
    cot_all: list[np.ndarray] = []
    for df in dfs:
        s, c = _prepare(df)
        slope_all.append(s)
        cot_all.append(c)

    slope_forward_deg = _finite(np.concatenate(slope_all)) if slope_all else np.array([])
    cot_vals = _finite(np.concatenate(cot_all)) if cot_all else np.array([])
    if slope_forward_deg.size == 0 or cot_vals.size == 0:
        raise SystemExit("No finite oriented slope/cost_of_transport data to plot.")

    fit_all = _fit_quad(slope_forward_deg, cot_vals)
    slope_nr, cot_nr = _remove_outliers(slope_forward_deg, cot_vals)
    fit_no_outliers = _fit_quad(slope_nr, cot_nr)
    x_plot = np.linspace(np.min(slope_forward_deg), np.max(slope_forward_deg), 100) if slope_forward_deg.size else np.array([])

    def _plot(y_range: tuple[float, float] | None, suffix: str) -> Path:
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        hb = ax.hexbin(
            slope_forward_deg,
            cot_vals,
            gridsize=args.gridsize,
            cmap="viridis",
            mincnt=1,
            linewidths=0.0,
        )
        if y_range is not None:
            ax.set_ylim(y_range)
        ax.set_xlabel("forward slope [deg]")
        ax.set_ylabel("metric_cost_of_transport_mean")
        title_suffix = "" if y_range is None else f" (y∈[{y_range[0]}, {y_range[1]}])"
        ax.set_title(f"Patch forward-oriented slope vs mean cost_of_transport{title_suffix}")
        if x_plot.size and fit_all is not None:
            a, b, c = fit_all
            ax.plot(x_plot, a * x_plot * x_plot + b * x_plot + c, color="tab:red", lw=1.4, label="quad fit (all)")
        if x_plot.size and fit_no_outliers is not None:
            a, b, c = fit_no_outliers
            ax.plot(x_plot, a * x_plot * x_plot + b * x_plot + c, color="tab:orange", lw=1.4, linestyle="--", label="quad fit (no outliers)")
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("patch count")
        ax.grid(alpha=0.25)
        fig.tight_layout()

        base_out = args.output if args.output is not None else default_out
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
