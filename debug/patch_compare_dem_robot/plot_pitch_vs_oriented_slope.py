#!/usr/bin/env python3
"""
Compare patch mean robot pitch (deg) with DEM forward-oriented slope (deg) for patches.
Forward slope is computed from slope_e/slope_n projected onto the robot's bearing quaternion.
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
    # Invert heading sign to align DEM gradient direction with robot forward (nav-friendly yaw flips).
    heading_rad = np.deg2rad(-yaw_deg)

    slope_e = df[SLOPE_E_COL].to_numpy(dtype=np.float64)
    slope_n = df[SLOPE_N_COL].to_numpy(dtype=np.float64)

    # Forward component: project (slope_e, slope_n) onto heading (east=north basis)
    dir_e = np.sin(heading_rad)
    dir_n = np.cos(heading_rad)
    slope_forward = slope_e * dir_e + slope_n * dir_n
    slope_forward_deg = np.rad2deg(np.arctan(slope_forward))

    pitch = df[PITCH_COL].to_numpy(dtype=np.float64)
    mask = np.isfinite(slope_forward_deg) & np.isfinite(pitch)
    return pitch[mask], slope_forward_deg[mask]


def _remove_outliers(x: np.ndarray, y: np.ndarray, pct_low: float = 2.0, pct_high: float = 98.0) -> tuple[np.ndarray, np.ndarray]:
    if x.size == 0 or y.size == 0:
        return x, y
    x_low, x_high = np.percentile(x, [pct_low, pct_high])
    y_low, y_high = np.percentile(y, [pct_low, pct_high])
    mask = (x >= x_low) & (x <= x_high) & (y >= y_low) & (y <= y_high)
    return x[mask], y[mask]


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot pitch_deg_mean vs oriented DEM slope (deg) for patches.")
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
        help="Output path for figure (default: reports/zz_compare_dem_robot/<patch-size>/pitch_vs_oriented_slope.png).",
    )
    ap.add_argument(
        "--gridsize",
        type=int,
        default=50,
        help="Hexbin grid size (default: 50).",
    )
    ap.add_argument(
        "--show-missions",
        action="store_true",
        help="Color by mission instead of patch count; outputs *_missions.png.",
    )
    args = ap.parse_args()

    dataset_path = args.dataset if args.dataset is not None else _default_dataset_path(args.patch_size)
    patch_label = _patch_label(args.patch_size)
    default_name = "pitch_vs_oriented_slope_missions.png" if args.show_missions else "pitch_vs_oriented_slope.png"
    default_out = DEFAULT_REPORT_DIR / patch_label / default_name

    dfs = _load_patch_groups(dataset_path, args.missions)
    if not dfs:
        raise SystemExit("No missions found in dataset (after filtering).")

    pitch_all: list[np.ndarray] = []
    slope_all: list[np.ndarray] = []
    mission_points: list[tuple[str, np.ndarray, np.ndarray]] = []
    for df in dfs:
        p, s = _prepare(df)
        if p.size == 0 or s.size == 0:
            continue
        pitch_all.append(p)
        slope_all.append(s)
        mission_points.append((df["mission_display"].iloc[0], p, s))

    pitch_vals = _finite(np.concatenate(pitch_all)) if pitch_all else np.array([])
    slope_deg = _finite(np.concatenate(slope_all)) if slope_all else np.array([])
    if pitch_vals.size == 0 or slope_deg.size == 0:
        raise SystemExit("No finite pitch/oriented slope data to plot.")

    base_out = args.output if args.output is not None else default_out

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    if args.show_missions:
        cmap = plt.get_cmap("tab20")
        unique_missions = [m for m, _, _ in mission_points]
        color_map = {m: cmap(i % cmap.N) for i, m in enumerate(sorted(set(unique_missions)))}
        for m, p, s in mission_points:
            ax.scatter(p, s, s=14, color=color_map[m], alpha=0.8, label=m)
        # deduplicate legend
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        uniq_h, uniq_l = [], []
        for h, l in zip(handles, labels):
            if l in seen:
                continue
            seen.add(l)
            uniq_h.append(h)
            uniq_l.append(l)
        if uniq_h:
            ax.legend(uniq_h, uniq_l, title="mission", loc="upper left")
    else:
        hb = ax.hexbin(
            pitch_vals,
            slope_deg,
            gridsize=args.gridsize,
            cmap="viridis",
            mincnt=1,
            linewidths=0.0,
        )
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("patch count")

    ax.set_xlabel("pitch_deg_mean [deg]")
    ax.set_ylabel("oriented slope [deg]")
    ax.set_title("Patch pitch vs oriented DEM slope")
    # 45-degree reference line (pitch == slope)
    lim_min = min(np.min(pitch_vals), np.min(slope_deg))
    lim_max = max(np.max(pitch_vals), np.max(slope_deg))
    diag = np.linspace(lim_min, lim_max, 100)
    ax.plot(diag, diag, color="tab:gray", linestyle="--", linewidth=1.0, label="pitch = slope")
    if not args.show_missions:
        ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    base_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_out, dpi=200)
    plt.close(fig)
    print(f"[ok] wrote {base_out}")


if __name__ == "__main__":
    main()
