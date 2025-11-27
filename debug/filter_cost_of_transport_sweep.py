#!/usr/bin/env python3
"""
Quick visual sweep of filter windows for cost_of_transport.

Adjust METRIC / MEDIAN_WINDOWS / MEAN_WINDOWS below to try other window sizes.
Each combination is plotted in a grid so you can visually compare smoothing strength. Uses the
metric implementation from src/metrics.py so you see the same logic as compute_metrics, with
custom filter chains injected for cost_of_transport_speed and cost_of_transport.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Make src/ importable when running from repo root
import sys
THIS_FILE = Path(__file__).resolve()
SRC_ROOT = THIS_FILE.parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.paths import get_paths
from utils.missions import resolve_mission
from utils.filtering import format_chain, load_metrics_config
import metrics as metrics_mod

# ---- knobs to tweak ----
METRIC: str = "cost_of_transport"
MEDIAN_WINDOWS: list[int] = [1, 3, 5, 7, 9]          # applied to cost_of_transport_speed and cost_of_transport
MEAN_WINDOWS: list[int] = [5, 15, 25, 35, 45]        # applied to cost_of_transport_speed and cost_of_transport


def _pick_synced(sync_dir: Path, hz: int | None) -> Path:
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


def _cot_filters(base_filters: dict, med_w: int | None, mean_w: int | None, default_power: Iterable[dict]) -> dict:
    """Return a filters dict overriding cost_of_transport* chains."""
    overrides = {}
    if med_w is None or mean_w is None:
        overrides["cost_of_transport_speed"] = []
        overrides["cost_of_transport"] = []
    else:
        chain = [
            {"type": "moving_median", "window": med_w},
            {"type": "moving_average", "window": mean_w},
        ]
        overrides["cost_of_transport_speed"] = chain
        overrides["cost_of_transport"] = chain
    overrides["cost_of_transport_power"] = list(default_power)
    merged = dict(base_filters)
    merged.update(overrides)
    return merged


def _compute_cot(df: pd.DataFrame, cfg_base: dict, filters_override: dict) -> np.ndarray:
    """Run metrics.cost_of_transport with supplied filter overrides."""
    cfg = dict(cfg_base)
    base_filters = dict(cfg.get("filters") or {})
    base_filters.update(filters_override)
    cfg["filters"] = base_filters
    return metrics_mod.REGISTRY[METRIC](df, cfg).to_numpy(dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize filter window effects for cost-of-transport metrics.")
    ap.add_argument("--mission", required=True, help="Mission alias or UUID")
    ap.add_argument("--hz", type=int, default=None, help="Pick synced_<Hz>Hz.parquet (default: latest)")
    ap.add_argument("--distance", action="store_true", help="Use cumulative distance for X-axis instead of time.")
    ap.add_argument("--tmin", type=float, default=None, help="Minimum X (time or distance) to plot.")
    ap.add_argument("--tmax", type=float, default=None, help="Maximum X (time or distance) to plot.")
    args = ap.parse_args()

    P = get_paths()
    metrics_cfg = load_metrics_config(Path(P["REPO_ROOT"]) / "config" / "metrics.yaml")
    cfg_base = {
        "filters": metrics_cfg.get("filters") or {},
        "params": metrics_cfg.get("params") or {},
        "robot": metrics_cfg.get("robot") or {},
    }
    mp = resolve_mission(args.mission, P)
    synced_path = _pick_synced(mp.synced, args.hz)
    df = pd.read_parquet(synced_path)

    if args.distance:
        if "dist_m" not in df.columns:
            raise KeyError("dist_m column missing in synced parquet. Re-run sync_streams.py to add it.")
        x = df["dist_m"].to_numpy(dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0)
        x = x - x[0]
        x_label = "distance traveled [m]"
    else:
        if "t" not in df.columns:
            raise KeyError("'t' column missing in synced parquet.")
        t0 = float(df["t"].iloc[0])
        x = (df["t"] - t0).to_numpy(dtype=np.float64)
        x_label = "time [s]"

    if args.tmin is not None or args.tmax is not None:
        mask = np.ones_like(x, dtype=bool)
        if args.tmin is not None:
            mask &= x >= float(args.tmin)
        if args.tmax is not None:
            mask &= x <= float(args.tmax)
        if not mask.any():
            raise ValueError("No samples remain after applying tmin/tmax window.")
        df = df.loc[mask].reset_index(drop=True)
        x = x[mask]

    base_filters = dict(cfg_base.get("filters") or {})
    default_power = base_filters.get("cost_of_transport_power") or [
        {"type": "moving_median", "window": 1},
        {"type": "moving_average", "window": 1},
    ]

    raw_filters = _cot_filters(base_filters, None, None, default_power)
    y_raw = _compute_cot(df, cfg_base, raw_filters)

    n_rows = len(MEDIAN_WINDOWS)
    n_cols = len(MEAN_WINDOWS)
    fig_w = 3.4 * n_cols
    fig_h = 2.1 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), sharex=True, sharey=False)

    for i, med_w in enumerate(MEDIAN_WINDOWS):
        for j, mean_w in enumerate(MEAN_WINDOWS):
            ax = axes[i][j] if n_rows > 1 else axes[j]  # type: ignore[index]
            filters_override = _cot_filters(base_filters, med_w, mean_w, default_power)
            y_filt = _compute_cot(df, cfg_base, filters_override)
            desc = format_chain(filters_override["cost_of_transport"])
            ax.plot(x, y_raw, color="0.75", linewidth=0.9, label="raw", zorder=1)
            ax.plot(x, y_filt, linewidth=1.1, label=desc, zorder=2)
            ax.grid(True, alpha=0.2)
            if i == n_rows - 1:
                ax.set_xlabel(x_label)
            if j == 0:
                ax.set_ylabel(METRIC)
            ax.set_title(f"median={med_w}, mean={mean_w}", fontsize=9)
            if i == 0 and j == 0:
                ax.legend(loc="upper right", frameon=False, fontsize=8)

    fig.suptitle(f"{mp.display} — {METRIC}", y=0.995, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    plt.show()


if __name__ == "__main__":
    main()
