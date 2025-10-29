#!/usr/bin/env python3
"""
Debug plot from *unsynced* tables.

Usage (repeat --plot for multiple tables):
python src/visualization/plot_unsynced_debug.py \
--mission ETH-1 \
--plot "cmd_vel:v_cmd_x,v_cmd_y,w_cmd_z" \
--plot "odom:vx,vy"

This reads from: data/tables/<mission_folder>/<table>.parquet
and saves to:   reports/<mission_folder>/unsynced_<table>.png
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Allow running directly
import sys
THIS = Path(__file__).resolve()
SRC_ROOT = THIS.parents[1]  # .../src
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.paths import get_paths

def parse_specs(specs: List[str]) -> Dict[str, List[str]]:
    """
    Parse ["table:col1,col2", ...] -> {"table": ["col1","col2"], ...}
    """
    out: Dict[str, List[str]] = {}
    for s in specs:
        if ":" not in s:
            raise SystemExit(f"Bad --plot spec '{s}'. Expected 'table:col1,col2'.")
        table, cols = s.split(":", 1)
        table = table.strip()
        cols_list = [c.strip() for c in cols.split(",") if c.strip()]
        if not table or not cols_list:
            raise SystemExit(f"Bad --plot spec '{s}'.")
        out[table] = cols_list
    return out

def resolve_mission(mission: str, P) -> Tuple[Path, Path, str, str]:
    mj = P["REPO_ROOT"] / "config" / "missions.json"
    meta = json.loads(mj.read_text())
    mission_id = meta.get("_aliases", {}).get(mission, mission)
    entry = meta.get(mission_id)
    if not entry:
        raise KeyError(f"Mission '{mission}' not found in missions.json")
    folder = entry["folder"]
    return (P["TABLES"] / folder), (P["SYNCED"] / folder), mission_id, folder

def main():
    ap = argparse.ArgumentParser(description="Plot raw (unsynced) tables for debugging.")
    ap.add_argument("--mission", required=True, help="Mission alias or UUID")
    ap.add_argument("--plot", required=True, action="append",
                    help="Spec 'table:col1,col2'. Repeat for multiple tables.")
    args = ap.parse_args()

    P = get_paths()
    tables_dir, _, mission_id, short = resolve_mission(args.mission, P)
    specs = parse_specs(args.plot)

    out_base = P["REPO_ROOT"] / "reports" / short
    out_base.mkdir(parents=True, exist_ok=True)

    for table, cols in specs.items():
        parquet_path = tables_dir / f"{table}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"{parquet_path} not found")

        df = pd.read_parquet(parquet_path)
        if "t" not in df.columns:
            raise KeyError(f"'t' column missing in {parquet_path.name}")

        # Check columns exist
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"Columns {missing} not in {parquet_path.name}. "
                           f"Available: {sorted(df.columns.tolist())}")

        t0 = float(df["t"].iloc[0])
        tt = df["t"] - t0

        n = len(cols)
        fig_h = max(3.0, 2.3 * n)
        fig, axes = plt.subplots(n, 1, figsize=(10, fig_h), sharex=True)
        if n == 1:
            axes = [axes]

        # choose a gap breaker, e.g. 0.25s (tune per topic)
        GAP_S = 0.25

        for ax, c in zip(axes, cols):
            s = df[c].copy()

            # break the line when the time gap is big: set the first sample after a big gap to NaN
            dt = tt.diff()
            big_gap = dt > GAP_S
            s_plot = s.copy()
            s_plot[big_gap] = np.nan  # this inserts a NaN so matplotlib breaks the line

            # draw: thin line + markers to see actual sample positions
            ax.plot(tt, s_plot, lw=1.0, label=c)
            ax.plot(tt, s, linestyle="None", marker=".", markersize=3)  # actual points

            # optional: mark NaNs explicitly
            if s.isna().any():
                nan_mask = s.isna()
                ax.plot(tt[nan_mask], np.zeros(nan_mask.sum()), ".", alpha=0.5, label=f"{c} NaN")

            # small stats printout to console
            if len(dt):
                median_dt = float(np.nanmedian(dt))
                n_gaps = int(np.nansum(big_gap))
                max_gap = float(dt.max()) if np.isfinite(dt.max()) else float("nan")
                print(f"[{c}] median dt={median_dt:.3f}s | gaps>{GAP_S:.2f}s: {n_gaps} | max gap={max_gap:.2f}s")

            ax.set_ylabel(c)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper right", frameon=False)


        axes[-1].set_xlabel("time [s]")
        fig.suptitle(f"{mission_id} — {short} — {table}.parquet", y=0.995, fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.98])

        out_path = out_base / f"unsynced_{table}.png"
        fig.savefig(out_path, dpi=160)
        print(f"[ok] saved {out_path}")

if __name__ == "__main__":
    main()
