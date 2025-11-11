#!/usr/bin/env python3
"""
Plot time-series metrics for a mission from the synced parquet.

Default behavior:
  - Creates one subplot per metric listed under `metrics.names` in config/metrics.yaml (if present in the parquet).

You can also pass a subset via --metrics.

Examples:
  python src/visualization/plot_timeseries.py --mission ETH-1
  python src/visualization/plot_timeseries.py --mission ETH-1 --hz 10 --metrics speed_error_abs speed_tracking_score
  python src/visualization/plot_timeseries.py --mission ETH-1 --with-speeds
  python src/visualization/plot_timeseries.py --mission TRIM-1 --overlay-pitch
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- path helper ---
import sys
from pathlib import Path
THIS_FILE = Path(__file__).resolve()
SRC_ROOT = THIS_FILE.parents[1]    # .../src
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from utils.paths import get_paths
from utils.missions import resolve_mission
from utils.filtering import filter_signal, load_metrics_config

PITCH_LABEL = "pitch [deg]"
PITCH_COLOR = "tab:orange"


def _get_quaternion_block(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return quaternion columns if present, preferring *_WB."""
    if all(c in df.columns for c in ("qw_WB", "qx_WB", "qy_WB", "qz_WB")):
        qw = df["qw_WB"].to_numpy(dtype=np.float64)
        qx = df["qx_WB"].to_numpy(dtype=np.float64)
        qy = df["qy_WB"].to_numpy(dtype=np.float64)
        qz = df["qz_WB"].to_numpy(dtype=np.float64)
        return qw, qx, qy, qz
    if all(c in df.columns for c in ("qw", "qx", "qy", "qz")):
        print("[warn] Using (qw,qx,qy,qz) instead of qw_WB..qz_WB — confirm frame is world-aligned.")
        qw = df["qw"].to_numpy(dtype=np.float64)
        qx = df["qx"].to_numpy(dtype=np.float64)
        qy = df["qy"].to_numpy(dtype=np.float64)
        qz = df["qz"].to_numpy(dtype=np.float64)
        return qw, qx, qy, qz
    raise KeyError("Quaternion columns not found (need qw_WB..qz_WB or qw..qz).")


def _normalize_quat_arrays(qw: np.ndarray, qx: np.ndarray, qy: np.ndarray, qz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    n[n == 0.0] = 1.0
    return qw / n, qx / n, qy / n, qz / n


def _euler_zyx_from_qWB(qw: np.ndarray, qx: np.ndarray, qy: np.ndarray, qz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert unit quaternion q_WB to yaw/pitch/roll (ZYX ordering) in degrees."""
    qw, qx, qy, qz = _normalize_quat_arrays(qw, qx, qy, qz)

    xx = qx * qx; yy = qy * qy; zz = qz * qz
    xy = qx * qy; xz = qx * qz; yz = qy * qz
    wx = qw * qx; wy = qw * qy; wz = qw * qz

    r00 = 1.0 - 2.0 * (yy + zz)
    r10 = 2.0 * (xy + wz)
    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (xx + yy)

    yaw = np.arctan2(r10, r00)
    pitch = np.arctan2(-r20, np.clip(np.sqrt(r00 * r00 + r10 * r10), 1e-12, None))
    roll = np.arctan2(r21, r22)
    return np.rad2deg(yaw), np.rad2deg(pitch), np.rad2deg(roll)


def _compute_pitch_deg(df: pd.DataFrame) -> np.ndarray:
    qw, qx, qy, qz = _get_quaternion_block(df)
    _, pitch_deg, _ = _euler_zyx_from_qWB(qw, qx, qy, qz)
    return -pitch_deg  # align with navigation convention (nose-up positive)


def _overlay_pitch(ax, tt, pitch_deg: np.ndarray):
    """Overlay pitch on a twin axis so metrics can be compared to slope."""
    ax2 = ax.twinx()
    ax2.plot(tt, pitch_deg, color=PITCH_COLOR, linewidth=1.2, label=PITCH_LABEL)
    ax2.set_ylabel(PITCH_LABEL, color=PITCH_COLOR)
    ax2.tick_params(axis="y", colors=PITCH_COLOR)
    ax2.spines["right"].set_color(PITCH_COLOR)
    ax2.grid(False)
    return ax2

def _pick_synced(sync_dir: Path, hz: int | None) -> Path:
    # If Hz specified, try metrics file first
    if hz is not None:
        p_metrics = sync_dir / f"synced_{hz}Hz_metrics.parquet"
        if p_metrics.exists():
            return p_metrics
        p_plain = sync_dir / f"synced_{hz}Hz.parquet"
        if p_plain.exists():
            return p_plain
        raise FileNotFoundError(f"Neither {p_metrics} nor {p_plain} found")

    metrics = sorted(sync_dir.glob("synced_*Hz_metrics.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    if metrics:
        return metrics[0]

    plains = sorted(sync_dir.glob("synced_*Hz.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not plains:
        raise FileNotFoundError(f"No synced parquets in {sync_dir}")
    return plains[0]

def _detect_metric_cols(df: pd.DataFrame, include: Sequence[str] | None, whitelist: Iterable[str]) -> List[str]:
    if include:
        missing = [c for c in include if c not in df.columns]
        if missing:
            raise KeyError(f"Requested metrics not present: {missing}")
        return list(include)

    wl = [c for c in (whitelist or []) if isinstance(c, str)]
    if wl:
        missing = [c for c in wl if c not in df.columns]
        if missing:
            print(f"[warn] Metrics listed in config but missing from parquet: {missing}")
        cols = [c for c in wl if c in df.columns]
        if not cols:
            raise RuntimeError("None of the metrics listed in config/metrics.yaml are present in the dataframe.")
        return cols

    # fallback: keep previous heuristic so script still works if config lacks names
    num = df.select_dtypes(include="number").columns
    cols = [c for c in num if c not in {"t"}]
    return cols

def main():
    ap = argparse.ArgumentParser(description="Plot metrics time-series for a mission.")
    ap.add_argument("--mission", required=True, help="Mission alias or UUID")
    ap.add_argument("--hz", type=int, default=None, help="Pick synced_<Hz>Hz.parquet (default: latest)")
    ap.add_argument("--metrics", nargs="*", default=None, help="Subset of metric columns to plot")
    ap.add_argument("--with-speeds", action="store_true", help="Add a top panel with v_cmd vs v_actual")
    ap.add_argument(
        "--overlay-pitch",
        action="store_true",
        help="Overlay pitch [deg] on each subplot using a secondary Y-axis.",
    )
    ap.add_argument("--out", default=None, help="Output image path (default: reports/<short>_timeseries.png)")
    args = ap.parse_args()

    P = get_paths()
    metrics_cfg = load_metrics_config(Path(P["REPO_ROOT"]) / "config" / "metrics.yaml")
    filters_cfg = metrics_cfg.get("filters", {})
    mp = resolve_mission(args.mission, P)
    sync_dir, mission_id, display_name = mp.synced, mp.mission_id, mp.display

    synced_path = _pick_synced(sync_dir, args.hz)

    df = pd.read_parquet(synced_path)
    if "t" not in df.columns:
        raise KeyError(f"'t' column missing in {synced_path}")
    t0 = float(df["t"].iloc[0])
    tt = df["t"] - t0

    pitch_deg = None
    if args.overlay_pitch:
        try:
            pitch_deg = _compute_pitch_deg(df)
        except Exception as exc:
            print(f"[warn] Unable to overlay pitch: {exc}")
            pitch_deg = None

    whitelist = metrics_cfg.get("metrics", {}).get("names", []) if metrics_cfg else []
    metric_cols = _detect_metric_cols(df, args.metrics, whitelist)

    # Decide whether to add the speed panel (explicit flag + available columns)
    has_cmd_speed = ("v_cmd" in df.columns) or ({"v_cmd_x", "v_cmd_y"}.issubset(df.columns))
    has_act_speed = ("v_actual" in df.columns) or ({"vx", "vy"}.issubset(df.columns))
    add_speed_panel = bool(args.with_speeds and (has_cmd_speed or has_act_speed))

    n_extra = 1 if add_speed_panel else 0
    n_panels = len(metric_cols) + n_extra
    if n_panels == 0:
        raise RuntimeError("No metric columns detected. Run compute_metrics or specify --metrics.")

    fig_h = max(3.0, 2.3 * n_panels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, fig_h), sharex=True)
    if n_panels == 1:
        axes = [axes]  # normalize to list

    ax_i = 0
    if add_speed_panel:
        # Build commanded speed
        if "v_cmd" in df.columns:
            v_cmd = df["v_cmd"]
        elif {"v_cmd_x", "v_cmd_y"}.issubset(df.columns):
            v_cmd = np.hypot(df["v_cmd_x"], df["v_cmd_y"])
        else:
            v_cmd = None

        # Build actual speed
        if "v_actual" in df.columns:
            v_act = df["v_actual"]
        elif {"vx", "vy"}.issubset(df.columns):
            v_act = np.hypot(df["vx"], df["vy"])
        else:
            v_act = None

        if v_cmd is not None:
            axes[ax_i].plot(tt, v_cmd, label="v_cmd")
        if v_act is not None:
            axes[ax_i].plot(tt, v_act, label="v_actual")
        axes[ax_i].set_ylabel("speed [m/s]")
        axes[ax_i].legend(loc="upper right")
        axes[ax_i].set_title("Commanded vs actual speed")
        if pitch_deg is not None:
            _overlay_pitch(axes[ax_i], tt, pitch_deg)
        ax_i += 1

    # Plot each requested/auto-detected metric
    for c in metric_cols:
        ax = axes[ax_i]
        raw_vals = df[c].to_numpy(dtype=np.float64)
        filt_vals = filter_signal(raw_vals, c, filters_cfg=filters_cfg, log_fn=print)
        series = filt_vals if filt_vals is not None else raw_vals
        ax.plot(tt, series, label=c)
        ax.set_ylabel(c)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", frameon=False)
        if pitch_deg is not None:
            _overlay_pitch(ax, tt, pitch_deg)
        ax_i += 1

    axes[-1].set_xlabel("time [s]")
    fig.suptitle(f"{display_name} — {mission_id}", y=0.995, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    # Save under reports/<mission_folder>/
    out_dir = P["REPO_ROOT"] / "reports" / display_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else (out_dir / f"{display_name}_timeseries.png")
    fig.savefig(out_path, dpi=160)
    print(f"[ok] saved {out_path}")

if __name__ == "__main__":
    main()
