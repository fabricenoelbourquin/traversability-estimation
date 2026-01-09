#!/usr/bin/env python3
"""
Plot pitch/roll of default q_WB vs map-based q_WB (if present), mission-aware.
Optionally include yaw with --plot-yaw.

Usage:
  python src/visualization/plot_quat_compare.py --mission ETH-1
  python src/visualization/plot_quat_compare.py --mission ETH-1 --output myplot.png
  python src/visualization/plot_quat_compare.py --mission ETH-1 --plot-yaw
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.paths import get_paths
from utils.cli import add_mission_arguments, resolve_mission_from_args

P = get_paths()


def quat_to_rpy(qw: np.ndarray, qx: np.ndarray, qy: np.ndarray, qz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quaternion (body->world) to roll, pitch, yaw (rad), Tait-Bryan XYZ."""
    # roll
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # pitch
    sinp = 2 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    # yaw
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def main() -> None:
    ap = argparse.ArgumentParser()
    add_mission_arguments(ap)
    ap.add_argument("--tables", type=Path, help="Override tables directory (defaults to mission tables)")
    ap.add_argument("--output", type=Path, help="Output PNG path (default: reports/<mission>/quat_pitch_roll_compare.png)")
    ap.add_argument("--plot-yaw", action="store_true", help="Include yaw subplot (default: off)")
    args = ap.parse_args()

    mp = resolve_mission_from_args(args, P)
    tables_dir = args.tables or mp.tables
    base_path = tables_dir / "base_orientation.parquet"
    if not base_path.exists():
        raise SystemExit(f"base_orientation.parquet not found at {base_path}")

    df = pd.read_parquet(base_path).sort_values("t")
    have_map = all(c in df.columns for c in ("qw_map", "qx_map", "qy_map", "qz_map"))
    if not have_map:
        print("[warn] map quaternion columns not found; plotting default only.")

    # Default orientation
    qw = df["qw"].to_numpy()
    qx = df["qx"].to_numpy()
    qy = df["qy"].to_numpy()
    qz = df["qz"].to_numpy()
    # Canonicalize sign
    neg = qw < 0
    qw[neg], qx[neg], qy[neg], qz[neg] = -qw[neg], -qx[neg], -qy[neg], -qz[neg]
    roll, pitch, yaw = quat_to_rpy(qw, qx, qy, qz)
    # Match visualization convention used in video_metric_viewer (flip pitch/yaw).
    pitch *= -1.0
    yaw *= -1.0

    # Map orientation (optional)
    roll_map = pitch_map = yaw_map = None
    if have_map:
        qwm = df["qw_map"].to_numpy()
        qxm = df["qx_map"].to_numpy()
        qym = df["qy_map"].to_numpy()
        qzm = df["qz_map"].to_numpy()
        negm = qwm < 0
        qwm[negm], qxm[negm], qym[negm], qzm[negm] = -qwm[negm], -qxm[negm], -qym[negm], -qzm[negm]
        roll_map, pitch_map, yaw_map = quat_to_rpy(qwm, qxm, qym, qzm)
        pitch_map *= -1.0
        yaw_map *= -1.0

    t = df["t"].to_numpy()
    deg = 180.0 / np.pi

    nrows = 3 if args.plot_yaw else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 8 if args.plot_yaw else 6), sharex=True)
    axes = np.atleast_1d(axes)
    axes[0].plot(t, pitch * deg, label="default pitch")
    if pitch_map is not None:
        axes[0].plot(t, pitch_map * deg, label="map pitch", alpha=0.8)
    axes[0].set_ylabel("Pitch [deg]")
    axes[0].legend()

    axes[1].plot(t, roll * deg, label="default roll")
    if roll_map is not None:
        axes[1].plot(t, roll_map * deg, label="map roll", alpha=0.8)
    axes[1].set_ylabel("Roll [deg]")
    axes[1].legend()

    if args.plot_yaw:
        axes[2].plot(t, yaw * deg, label="default yaw")
        if yaw_map is not None:
            axes[2].plot(t, yaw_map * deg, label="map yaw", alpha=0.8)
        axes[2].set_ylabel("Yaw [deg]")
        axes[2].legend()

    axes[-1].set_xlabel("time [s]")

    fig.tight_layout()

    out_path = args.output
    if out_path is None:
        out_path = Path(P["REPO_ROOT"]) / "reports" / mp.display / "quat_pitch_roll_compare.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
