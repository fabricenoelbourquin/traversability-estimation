#!/usr/bin/env python3
"""
Plot histograms comparing robot pitch to oriented DEM slope for patches.

Generates two figures:
  1) pitch vs oriented slope vs difference (pitch - slope) in degrees.
  2) |pitch| vs |oriented slope| vs difference |pitch| - |slope|.
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


DEFAULT_PATCH_SIZE_M: float = 3.0
DEFAULT_REPORT_DIR = Path(get_paths()["REPO_ROOT"]) / "reports" / "zz_compare_dem_robot"

SLOPE_E_COL = "slope_e"
SLOPE_N_COL = "slope_n"
PITCH_COL = "pitch_deg_mean"


def _decode_attr_val(val):
    if isinstance(val, (bytes, bytearray)):
        try:
            return val.decode("utf-8")
        except Exception:
            return val
    return val


def _patch_label(patch_size_m: float | None) -> str:
    size = DEFAULT_PATCH_SIZE_M if patch_size_m is None else patch_size_m
    label_num = f"{size:.3f}".rstrip("0").rstrip(".")
    return f"{label_num}m"


def _slugify(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)


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


def _normalize_quat_arrays(qw: np.ndarray, qx: np.ndarray, qy: np.ndarray, qz: np.ndarray):
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    n[n == 0.0] = 1.0
    return qw / n, qx / n, qy / n, qz / n


def _yaw_deg_from_quat(df: pd.DataFrame) -> np.ndarray:
    candidates = [
        ("bearing_qw", "bearing_qx", "bearing_qy", "bearing_qz"),
        ("qw_WB", "qx_WB", "qy_WB", "qz_WB"),
        ("qw", "qx", "qy", "qz"),
    ]
    for cols in candidates:
        if all(c in df.columns for c in cols):
            qw, qx, qy, qz = (df[c].to_numpy(dtype=np.float64) for c in cols)
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
    raise SystemExit("Quaternion columns not found (need bearing_qw..qz or qw_WB..qz_WB or qw..qz).")


def _prepare(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    missing = [c for c in (SLOPE_E_COL, SLOPE_N_COL, PITCH_COL) if c not in df.columns]
    if missing:
        raise SystemExit(f"Required columns missing: {missing}")
    yaw_deg = _yaw_deg_from_quat(df)
    heading_rad = np.deg2rad(-yaw_deg)

    slope_e = df[SLOPE_E_COL].to_numpy(dtype=np.float64)
    slope_n = df[SLOPE_N_COL].to_numpy(dtype=np.float64)

    dir_e = np.sin(heading_rad)
    dir_n = np.cos(heading_rad)
    slope_forward = slope_e * dir_e + slope_n * dir_n
    slope_forward_deg = np.rad2deg(np.arctan(slope_forward))

    pitch = df[PITCH_COL].to_numpy(dtype=np.float64)
    mask = np.isfinite(slope_forward_deg) & np.isfinite(pitch)
    return pitch[mask], slope_forward_deg[mask]


def _plot_histograms(pitch: np.ndarray, slope: np.ndarray, title: str, out_path: Path) -> None:
    diff = pitch - slope
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    axes[0].hist(pitch, bins=60, alpha=0.8, color="tab:blue")
    axes[0].set_title("pitch [deg]")
    axes[0].set_ylabel("count")
    axes[0].grid(alpha=0.25)

    axes[1].hist(slope, bins=60, alpha=0.8, color="tab:orange")
    axes[1].set_title("oriented slope [deg]")
    axes[1].set_ylabel("count")
    axes[1].grid(alpha=0.25)

    axes[2].hist(diff, bins=60, alpha=0.8, color="tab:green")
    axes[2].set_title("pitch - slope [deg]")
    axes[2].set_xlabel("degrees")
    axes[2].set_ylabel("count")
    axes[2].grid(alpha=0.25)

    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[ok] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot pitch vs oriented slope histograms for patches.")
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
        "--exclude",
        nargs="*",
        default=None,
        help="Mission names/displays to exclude from the plots.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: reports/zz_compare_dem_robot/<patch-size>/).",
    )
    args = ap.parse_args()

    dataset_path = args.dataset if args.dataset is not None else _default_dataset_path(args.patch_size)
    patch_label = _patch_label(args.patch_size)
    base_out_dir = args.output_dir if args.output_dir is not None else DEFAULT_REPORT_DIR / patch_label

    dfs = _load_patch_groups(dataset_path, args.missions)
    if not dfs:
        raise SystemExit("No missions found in dataset (after filtering).")

    exclude_set = set(args.exclude) if args.exclude else set()
    if exclude_set:
        dfs = [d for d in dfs if str(d["mission_display"].iloc[0]) not in exclude_set]
        if not dfs:
            raise SystemExit("All missions excluded; nothing to plot.")

    pitch_all: list[np.ndarray] = []
    slope_all: list[np.ndarray] = []
    for df in dfs:
        p, s = _prepare(df)
        if p.size and s.size:
            pitch_all.append(p)
            slope_all.append(s)

    if not pitch_all or not slope_all:
        raise SystemExit("No finite pitch/slope data to plot.")

    pitch_vals = _finite(np.concatenate(pitch_all))
    slope_vals = _finite(np.concatenate(slope_all))
    if pitch_vals.size == 0 or slope_vals.size == 0:
        raise SystemExit("No finite pitch/slope data to plot.")

    suffix = ""
    if exclude_set:
        suffix = "_excl-" + "_".join(_slugify(m) for m in sorted(exclude_set))

    # Raw signed
    _plot_histograms(
        pitch_vals,
        slope_vals,
        "Pitch vs oriented slope (deg)",
        base_out_dir / f"pitch_vs_oriented_slope_hist{suffix}.png",
    )

    # Magnitude
    _plot_histograms(
        np.abs(pitch_vals),
        np.abs(slope_vals),
        "Pitch vs oriented slope magnitudes (deg)",
        base_out_dir / f"pitch_vs_oriented_slope_mag_hist{suffix}.png",
    )


if __name__ == "__main__":
    main()
