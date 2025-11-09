#!/usr/bin/env python3
"""
Plot time-series metrics for a mission from the synced parquet.

Default behavior:
  - Creates one subplot per metric column found in the synced parquet.
  - Metrics are detected as:
      (a) columns that match names from metrics.REGISTRY (if importable),
      or (b) numeric columns excluding common raw/state fields.

You can also pass a subset via --metrics.

Examples:
  python src/visualization/plot_timeseries.py --mission ETH-1
  python src/visualization/plot_timeseries.py --mission ETH-1 --hz 10 --metrics speed_error_abs speed_tracking_score
  python src/visualization/plot_timeseries.py --mission ETH-1 --with-speeds
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Sequence

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

# optional: use registry to detect metric names if present
try:
    from metrics import REGISTRY as METRIC_REGISTRY  # if you kept metrics in src/metrics.py
except Exception:
    try:
        from metrics.metrics import REGISTRY as METRIC_REGISTRY  # if later you move to a package
    except Exception:
        METRIC_REGISTRY = {}

# columns we don't treat as "metrics" when auto-detecting
EXCLUDE_DEFAULT = {
    "t", "x", "y", "z",
    "lat", "lon", "alt",
    "vx", "vy", "vz",
    "v_cmd_x", "v_cmd_y", "w_cmd_z",
    "v_cmd", "v_actual", "speed",
    "qw_x", "qx_x", "qy_x", "qz_x", 
    "ax", "ay", "az",
    "wx", "wy", "wz",
    "qw_WB", "qx_WB", "qy_WB", "qz_WB",
    "dem_grad_e", "dem_grad_n",
    "dem_slope_lat", "dem_slope_long",
    "dem_slope_lat_deg", "dem_slope_long_deg",
      # convenience/raw
}

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

    # Otherwise pick the newest, preferring *_metrics.parquet
    cands = sorted(
        list(sync_dir.glob("synced_*Hz_metrics.parquet")) +
        list(sync_dir.glob("synced_*Hz.parquet")),
        key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not cands:
        raise FileNotFoundError(f"No synced parquets in {sync_dir}")
    return cands[0]

def _detect_metric_cols(df: pd.DataFrame, include: Sequence[str] | None) -> List[str]:
    if include:
        missing = [c for c in include if c not in df.columns]
        if missing:
            raise KeyError(f"Requested metrics not present: {missing}")
        return list(include)

    # try registry first
    reg_names = [k for k in METRIC_REGISTRY.keys() if k in df.columns]
    cols: List[str]
    if reg_names:
        cols = reg_names
    else:
        # fall back: all numeric, excluding "raw-ish" columns
        num = df.select_dtypes(include="number").columns
        cols = [c for c in num if c not in EXCLUDE_DEFAULT]

    # drop all-NaN or near-constant metrics
    good = []
    for c in cols:
        s = df[c]
        if s.notna().sum() < 5:
            continue
        if float(s.max() - s.min()) == 0.0:
            continue
        good.append(c)
    if not good and cols:
        # if filtering removed all, keep original cols to show something
        good = cols
    return good

def main():
    ap = argparse.ArgumentParser(description="Plot metrics time-series for a mission.")
    ap.add_argument("--mission", required=True, help="Mission alias or UUID")
    ap.add_argument("--hz", type=int, default=None, help="Pick synced_<Hz>Hz.parquet (default: latest)")
    ap.add_argument("--metrics", nargs="*", default=None, help="Subset of metric columns to plot")
    ap.add_argument("--with-speeds", action="store_true", help="Add a top panel with v_cmd vs v_actual")
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

    metric_cols = _detect_metric_cols(df, args.metrics)

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
