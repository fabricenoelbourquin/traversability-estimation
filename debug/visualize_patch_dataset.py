#!/usr/bin/env python3
"""
Visualize slope coverage in a patch-level HDF5 dataset (output of build_patch_dataset.py).

Creates:
  - Histogram panel for slope magnitude min/mean/max (converted to degrees).
  - Optional orientation comparison (tilt from bearing quaternion vs slope).

Usage:
  # default: use dataset path from config/dataset.yaml (patches_{patch_size_m}m.h5) and all missions
  python debug/visualize_patch_dataset.py
  # custom dataset and optional mission filtering (repeatable)
  python debug/visualize_patch_dataset.py /path/to/patches_5m.h5 --mission ETH-1 --mission GRI-1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd
import yaml


from utils.paths import get_paths  # noqa: E402

# ---------------- Quaternion utilities (aligned with video_metric_viewer.py) ----------------

def _normalize_quat_arrays(qw: np.ndarray, qx: np.ndarray, qy: np.ndarray, qz: np.ndarray):
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    n[n == 0.0] = 1.0
    return qw / n, qx / n, qy / n, qz / n


def euler_zyx_from_qWB(qw: np.ndarray, qx: np.ndarray, qy: np.ndarray, qz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert quaternion q_WB (body→world, active) to yaw/pitch/roll (ZYX) in degrees.
    World: ENU. Body: x-forward, y-left, z-up.
    Matches video_metric_viewer.py (yaw/pitch sign flipped for navigation convention).
    """
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
    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (xx + yy)

    yaw = np.arctan2(r10, r00)
    pitch = np.arctan2(-r20, np.clip(np.sqrt(r00 * r00 + r10 * r10), 1e-12, None))
    roll = np.arctan2(r21, r22)

    yaw_deg = np.rad2deg(yaw) * -1.0
    pitch_deg = np.rad2deg(pitch) * -1.0
    roll_deg = np.rad2deg(roll)
    return yaw_deg, pitch_deg, roll_deg


# ---------------- Data loading and helpers ----------------

def load_yaml(p: Path) -> dict:
    try:
        return yaml.safe_load(p.read_text())
    except Exception:
        return {}


def format_patch_size_label(size_m: float) -> str:
    return f"{size_m:.3f}".rstrip("0").rstrip(".")


def resolve_default_dataset_path(paths: dict) -> Path:
    cfg_path = Path(paths["REPO_ROOT"]) / "config" / "dataset.yaml"
    cfg = load_yaml(cfg_path)
    patch_cfg = cfg.get("patch", {}) if isinstance(cfg, dict) else {}
    output_cfg = cfg.get("output", {}) if isinstance(cfg, dict) else {}

    size_m = float(patch_cfg.get("size_m", 5.0))
    label = format_patch_size_label(size_m)
    fname_tpl = output_cfg.get("filename", "patches_{patch_size_m}m.h5")
    fname = fname_tpl.format(patch_size_m=label)
    out_dir = output_cfg.get("path") or paths.get("DATASETS")
    return Path(out_dir) / fname


def load_patch_dataframe(h5_path: Path, missions: list[str] | None) -> tuple[pd.DataFrame, list[str]]:
    with h5py.File(h5_path, "r") as f:
        group_names = [k for k in f.keys() if isinstance(f[k], h5py.Group)]
        if not group_names:
            raise SystemExit(f"No groups found in {h5_path}.")

        target_groups = group_names if missions is None else [m for m in missions if m in group_names]
        missing = [] if missions is None else [m for m in missions if m not in group_names]
        if missing:
            raise SystemExit(f"Requested mission(s) {missing} not in {h5_path}. Available: {group_names}")

        records = []
        used_groups: list[str] = []
        for g in target_groups:
            grp = f[g]
            if "patches" not in grp:
                continue
            ds = grp["patches"]
            data = ds[()]
            col_attr = ds.attrs.get("column_order", grp.attrs.get("column_order", None))
            if isinstance(col_attr, bytes):
                col_attr = col_attr.decode("utf-8", errors="ignore")
            columns = None
            if isinstance(col_attr, str):
                try:
                    columns = json.loads(col_attr)
                except Exception:
                    columns = None
            df = pd.DataFrame.from_records(data, columns=columns)
            df["mission"] = g
            records.append(df)
            used_groups.append(g)

        if not records:
            raise SystemExit(f"No 'patches' datasets found in groups {target_groups}.")

        df_all = pd.concat(records, axis=0, ignore_index=True)
        return df_all, used_groups


def slope_to_deg(values: np.ndarray) -> np.ndarray:
    """Convert slope (rise/run) to degrees for readability."""
    return np.rad2deg(np.arctan(values.astype(np.float64)))


def finite(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


# ---------------- Plotting ----------------

def plot_slope_hist(slopes_deg: dict[str, np.ndarray], out_path: Path, bins: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    keys = ("slope_mag_min", "slope_mag_mean", "slope_mag_max")
    titles = ("Slope mag MIN [deg]", "Slope mag MEAN [deg]", "Slope mag MAX [deg]")
    for ax, key, title in zip(axes, keys, titles):
        data = finite(slopes_deg.get(key, np.array([])))
        ax.hist(data, bins=bins, color="#4f81bd", alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel("degrees")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("patch count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_orientation_comparison(
    slopes_deg: dict[str, np.ndarray],
    pitch_deg: np.ndarray,
    roll_deg: np.ndarray,
    tilt_deg: np.ndarray,
    out_path: Path,
    bins: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Tilt histogram
    ax0 = axes[0]
    ax0.hist(finite(np.abs(pitch_deg)), bins=bins, color="#c0504d", alpha=0.4, label="|pitch|")
    ax0.hist(finite(np.abs(roll_deg)), bins=bins, color="#8064a2", alpha=0.4, label="|roll|")
    ax0.hist(finite(tilt_deg), bins=bins, color="#9bbb59", alpha=0.9, label="tilt=√(pitch²+roll²)", zorder=3)
    ax0.set_title("Robot inclination [deg]")
    ax0.set_xlabel("degrees")
    ax0.set_ylabel("patch count")
    ax0.legend()
    ax0.grid(True, alpha=0.25)

    # Mean slope vs tilt
    ax1 = axes[1]
    mean_deg = slopes_deg.get("slope_mag_mean", np.array([]))
    mask_mean = np.isfinite(mean_deg) & np.isfinite(tilt_deg)
    if np.any(mask_mean):
        ax1.hexbin(
            mean_deg[mask_mean],
            tilt_deg[mask_mean],
            gridsize=40,
            cmap="Blues",
            mincnt=1,
            norm=colors.LogNorm(vmin=1.0),
            alpha=0.9,
        )
    ax1.set_title("Slope mean vs tilt")
    ax1.set_xlabel("slope_mag_mean [deg]")
    ax1.set_ylabel("tilt [deg]")
    ax1.grid(True, alpha=0.25)

    # Mean slope vs tilt (uniform scatter for spread visibility)
    ax2 = axes[2]
    if np.any(mask_mean):
        ax2.scatter(
            mean_deg[mask_mean],
            tilt_deg[mask_mean],
            s=8,
            color="tab:blue",
            alpha=0.35,
            edgecolors="none",
        )
    ax2.set_title("Slope mean vs tilt (uniform)")
    ax2.set_xlabel("slope_mag_mean [deg]")
    ax2.set_ylabel("tilt [deg]")
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(description="Visualize patch dataset slopes and compare to robot bearing.")
    ap.add_argument("dataset", nargs="?", type=Path, help="Path to the HDF5 produced by build_patch_dataset.py (default: config/dataset.yaml output)")
    ap.add_argument("--mission", action="append", help="Mission/group name to analyze (repeatable). Default: all groups in the file.")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent, help="Directory to write figures (default: debug/)")
    ap.add_argument("--bins", type=int, default=60, help="Histogram bins")
    args = ap.parse_args()

    paths = get_paths()
    dataset_path = args.dataset if args.dataset is not None else resolve_default_dataset_path(paths)
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    missions = args.mission if args.mission else None
    df, used_groups = load_patch_dataframe(dataset_path, missions)
    mission_label = "all" if len(used_groups) > 1 else used_groups[0]

    slopes_deg = {}
    for key in ("slope_mag_min", "slope_mag_mean", "slope_mag_max"):
        if key not in df.columns:
            raise SystemExit(f"Column '{key}' missing in dataset.")
        slopes_deg[key] = slope_to_deg(df[key].to_numpy(dtype=np.float64))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    slope_fig = args.out_dir / f"patch_slopes_{mission_label}.png"
    plot_slope_hist(slopes_deg, slope_fig, bins=args.bins)

    # Orientation comparison (if bearing quaternion is present)
    has_bearing = all(c in df.columns for c in ("bearing_qw", "bearing_qx", "bearing_qy", "bearing_qz"))
    if has_bearing:
        qw = df["bearing_qw"].to_numpy(dtype=np.float64)
        qx = df["bearing_qx"].to_numpy(dtype=np.float64)
        qy = df["bearing_qy"].to_numpy(dtype=np.float64)
        qz = df["bearing_qz"].to_numpy(dtype=np.float64)
        _yaw_deg, pitch_deg, roll_deg = euler_zyx_from_qWB(qw, qx, qy, qz)
        tilt_deg = np.hypot(pitch_deg, roll_deg)

        orient_fig = args.out_dir / f"patch_slope_tilt_{mission_label}.png"
        plot_orientation_comparison(slopes_deg, pitch_deg, roll_deg, tilt_deg, orient_fig, bins=args.bins)

        # Simple numeric correlation for quick inspection
        mask_corr = np.isfinite(slopes_deg["slope_mag_mean"]) & np.isfinite(tilt_deg)
        if np.any(mask_corr):
            corr = np.corrcoef(slopes_deg["slope_mag_mean"][mask_corr], tilt_deg[mask_corr])[0, 1]
            print(f"[info] Pearson r (slope_mag_mean_deg vs tilt_deg): {corr:.3f} (n={mask_corr.sum()})")
        print(f"[ok] wrote {orient_fig}")
    else:
        print("[warn] bearing_q* columns missing; skipping orientation comparison.")

    print(f"[ok] wrote {slope_fig}")
    print(f"[info] patches: {len(df)}, groups: {used_groups}")
    print(f"[info] dataset: {dataset_path}")


if __name__ == "__main__":
    main()
