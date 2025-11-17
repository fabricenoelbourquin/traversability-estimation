#!/usr/bin/env python3
"""
Build a single, time-aligned parquet for a mission.

Inputs (if present) from data/tables/<mission>/ :
  - cmd_vel.parquet  (cols: t, v_cmd_x, v_cmd_y, w_cmd_z)
  - odom.parquet     (cols: t, vx, vy, x, y, speed [optional])
  - gps.parquet      (cols: t, lat, lon, alt)
  - imu.parquet      (optional; cols: t, roll, pitch, yaw, ...)

Outputs to data/synced/<mission>/ :
  - synced_<Hz>Hz.parquet
  - meta.json (params + coverage)
  - (optional) reports/<mission>_sync_qc.png with --plot

- Build a master time grid at a user/configured rate (Hz), then
  merge-asof each table onto that grid with a configurable matching tolerance,
  followed by bounded interpolation to close small gaps.

Example usage:
python src/sync_streams.py \
  --mission ETH-1 \
  --plot
"""

from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

# --- path helpers ---

from utils.paths import get_paths
from utils.missions import resolve_mission

# Resolve common repo paths once at import time. Expects keys:
# RAW / TABLES / SYNCED / REPO_ROOT / MISSIONS_JSON
P = get_paths()

# --- helpers ---
def load_yaml(p: Path) -> dict:
    """
    Load a YAML file if it exists, else return an empty dict.
    """
    return yaml.safe_load(p.read_text()) if p.exists() else {}

def load_df(path: Path) -> Optional[pd.DataFrame]:
    """
    Load a parquet file (if present) and normalize the time column:
      - keep only rows with finite 't'
      - cast 't' to float seconds
      - sort by 't'
      - drop duplicate timestamps (keep last)
      - drop metadata columns that break merges (e.g., 'stamp_ns')
    Returns None if the file does not exist.
    """
    if path.exists():
        df = pd.read_parquet(path)
        # Ensure float seconds & monotonic t
        df = df[df["t"].notna()].copy()
        df["t"] = df["t"].astype(float)
        df = df.sort_values("t").drop_duplicates("t", keep="last")
        # Drop metadata timestamps; 't' is our canonical time
        df = df.drop(columns=[c for c in df.columns if c.startswith("stamp_ns")], errors="ignore")
        return df
    return None

def make_master_time(dfs, hz: float) -> pd.DataFrame:
    """
    Create a master time grid covering the union of time spans across all input tables.
    The grid is uniform at 1/hz seconds.

    Include the last step if the range boundary is within half a step to avoid
    dropping the final sample due to floating-point rounding.

    Returns a DataFrame with a single column 't'.
    """
    tmins = [df["t"].min() for df in dfs if df is not None]
    tmaxs = [df["t"].max() for df in dfs if df is not None]
    if not tmins:
        raise RuntimeError("No input tables present.")
    t0, t1 = min(tmins), max(tmaxs)
    dt = 1.0 / hz
    # Inclusive last step if close to boundary
    # Number of steps including start; add 1 so that t0 is included.
    n = int(math.floor((t1 - t0) / dt)) + 1
    # Use rounding to a few extra decimals to avoid cumulative fp drift.
    t = np.round(t0 + np.arange(n) * dt, 9)
    return pd.DataFrame({"t": t})

def asof_join(base: pd.DataFrame, df: pd.DataFrame, direction: str, tol_s: float) -> pd.DataFrame:
    """
    As-of join df onto base by float-seconds 't'.
    Drops metadata columns like 'stamp_ns*' to avoid suffix collisions across chained merges.
    """
    left = base.sort_values("t").copy()
    right = df.sort_values("t").copy()

    # Remove problematic metadata columns (present in many sources) that cause repeated suffixing
    left = left.drop(columns=[c for c in left.columns if c.startswith("stamp_ns")], errors="ignore")
    right = right.drop(columns=[c for c in right.columns if c.startswith("stamp_ns")], errors="ignore")

    # Decide tolerance type based on dtype of 't'
    t_dtype = left["t"].dtype
    if np.issubdtype(t_dtype, np.number):
        tol = tol_s
    else:
        tol = pd.Timedelta(seconds=tol_s)

    return pd.merge_asof(
        left,
        right,
        on="t",
        direction=direction,
        tolerance=tol,
    )


def interpolate_numeric(df: pd.DataFrame, limit_s: float, method: str) -> pd.DataFrame:
    """
    Interpolate numeric columns over small gaps, bounded by `limit_s`.

    Implementation details:
    - Set a TimedeltaIndex based on 't' seconds.
    - Use pandas' 'time' or 'linear' interpolation.
    - Compute 'limit' (max consecutive NaNs to fill) from the median dt.

    Returns a new DataFrame with interpolated numeric columns.
    """
    out = df.copy()
    # Build a time-based index; this enables 'time' interpolation
    out = out.set_index(pd.to_timedelta(out["t"], unit="s"))
    # Identify numeric columns to interpolate (avoid touching non-numeric).
    num = out.select_dtypes(include="number").columns
    if len(num):
        # Translate seconds limit to a number of consecutive steps (samples)
        median_dt = out.index.to_series().diff().median()
        # Guard: if median_dt is NaT (single row) or zero, avoid div-by-zero
        step = (median_dt.total_seconds() if pd.notna(median_dt) else None) or 1e-9
        limit = None if limit_s is None else int(round(limit_s / step))
        out[num] = out[num].interpolate(method=("time" if method=="time" else "linear"), limit=limit, limit_direction="both")
    # Restore a simple RangeIndex
    out = out.reset_index(drop=True)
    return out

def rolling_mean(df: pd.DataFrame, cols: list[str], win_s: float) -> None:
    """
    In-place rolling mean over specified columns using a window of ~win_s seconds.
    Window length is computed from the median dt in 't'.

    No-op if cols empty, or win_s <= 0, or dt is invalid.
    """
    if not cols or win_s is None or win_s <= 0:
        return
    dt = float(df["t"].diff().median())
    if not np.isfinite(dt) or dt <= 0:
        return
    w = max(1, int(round(win_s / dt)))
    for c in cols:
        if c in df:
            df[c] = df[c].rolling(window=w, min_periods=1, center=True).mean()

def compute_convenience_cols(df: pd.DataFrame) -> None:
    """
    Add common convenience columns in-place:
      - v_cmd    := hypot(v_cmd_x, v_cmd_y)  (commanded planar speed)
      - v_actual := hypot(vx, vy)            (measured planar speed)
    Only added if the required inputs are present.
    """
    if {"v_cmd_x","v_cmd_y"}.issubset(df.columns):
        df["v_cmd"] = np.hypot(df["v_cmd_x"], df["v_cmd_y"])
    if {"vx","vy"}.issubset(df.columns):
        df["v_actual"] = np.hypot(df["vx"], df["vy"])

def add_distance_traveled(df: pd.DataFrame) -> None:
    """
    Estimate planar distance from odometry (x, y).
    Adds columns:
      - dist_m_per_step: incremental progress between samples.
      - dist_m: cumulative distance with NaNs treated as 0.
    """
    n = len(df)
    if n == 0:
        df["dist_m_per_step"] = pd.Series(dtype=float)
        df["dist_m"] = pd.Series(dtype=float)
        return
    if not {"x", "y"}.issubset(df.columns):
        raise KeyError("Distance computation requires x and y columns.")
    coords = np.stack([df["x"].to_numpy(dtype=np.float64), df["y"].to_numpy(dtype=np.float64)], axis=1)
    deltas = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    step = pd.Series(np.concatenate([[0.0], deltas]), index=df.index)
    df["dist_m_per_step"] = step
    df["dist_m"] = step.fillna(0.0).cumsum()

def main():
    """
    CLI entry point:
      1) Load config (sync.yaml) and mission metadata.
      2) Read available input tables (cmd/odom/gps/imu).
      3) Build master time grid.
      4) Optionally shift cmd timestamps (lag compensation).
      5) Merge-asof all tables to the grid with tolerance.
      6) Interpolate small numeric gaps.
      7) Add convenience columns + optional smoothing.
      8) Compute simple coverage stats.
      9) Write synced parquet + meta.json (+ optional QC plot).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", required=True, help="Mission alias or UUID (from missions.json)")
    ap.add_argument("--hz", type=float, help="Resample Hz (overrides config)")
    ap.add_argument("--tolerance-ms", type=float, help="As-of match tolerance in ms")
    ap.add_argument("--direction", choices=["nearest","backward"], help="As-of direction")
    ap.add_argument("--lag-cmd-ms", type=float, help="Shift cmd_vel timestamps forward by ms")
    ap.add_argument("--plot", action="store_true", help="Write a small QC plot into reports/")
    args = ap.parse_args()

    # Load synchronization parameters from config/sync.yaml with CLI overrides.
    sync_cfg = load_yaml(P["REPO_ROOT"] / "config" / "sync.yaml")
    hz = args.hz or float(sync_cfg.get("resample_hz", 10))
    direction = args.direction or sync_cfg.get("align", {}).get("direction", "nearest")
    tol_ms = args.tolerance_ms or float(sync_cfg.get("align", {}).get("tolerance_ms", 120))
    interp_method = sync_cfg.get("interpolate", {}).get("method", "linear")
    interp_limit_s = float(sync_cfg.get("interpolate", {}).get("limit_seconds", 2.0))
    lag_cmd_ms = args.lag_cmd_ms if args.lag_cmd_ms is not None else float(sync_cfg.get("lag", {}).get("cmd_ms", 0))
    smooth_win_s = float(sync_cfg.get("smooth", {}).get("speed_window_s", 0.0))

    # Resolve mission -> tables dir, output dir, canonical mission_id
    mp = resolve_mission(args.mission, P)
    tables_dir, out_dir, mission_id, display_name = mp.tables, mp.synced, mp.mission_id, mp.display
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load available tables
    df_cmd  = load_df(tables_dir / "cmd_vel.parquet")
    df_odom = load_df(tables_dir / "odom.parquet")
    df_gps  = load_df(tables_dir / "gps.parquet")
    df_imu  = load_df(tables_dir / "imu.parquet")
    df_qwb  = load_df(tables_dir / "base_orientation.parquet")
    df_js   = load_df(tables_dir / "joint_states.parquet")         # q_*, qd_*
    df_sea  = load_df(tables_dir / "actuator_readings.parquet")    # tau_*

    dfs = [d for d in [df_cmd, df_odom, df_gps, df_imu, df_js, df_sea] if d is not None and len(d)]

    # Build a uniform master time base spanning all inputs
    master = make_master_time(dfs, hz)

    # Optional lag compensation: if commands lead reality, shift cmd timestamps
    if df_cmd is not None and lag_cmd_ms and lag_cmd_ms != 0:
        df_cmd = df_cmd.copy()
        df_cmd["t"] = df_cmd["t"] + lag_cmd_ms * 1e-3

    tol_s = tol_ms * 1e-3
    # Merge-asof each table onto the master time base
    synced = master
    if df_cmd is not None:
        synced = asof_join(synced, df_cmd, direction, tol_s)
    if df_odom is not None:
        synced = asof_join(synced, df_odom, direction, tol_s)
    if df_gps is not None:
        synced = asof_join(synced, df_gps, direction, tol_s)
    if df_imu is not None:
        synced = asof_join(synced, df_imu, direction, tol_s)
    if df_qwb is not None and len(df_qwb):
        # rename to a canonical block so we don’t clash with odom/imu quats
        q = df_qwb.rename(columns={"qw":"qw_WB","qx":"qx_WB","qy":"qy_WB","qz":"qz_WB"})
        synced = asof_join(synced, q[["t","qw_WB","qx_WB","qy_WB","qz_WB"]], direction, tol_s)
        # canonicalize sign
        if "qw_WB" in synced:
            neg = synced["qw_WB"] < 0
            for c in ("qw_WB","qx_WB","qy_WB","qz_WB"):
                synced.loc[neg, c] = -synced.loc[neg, c]
    # Joint states & torques
    if df_js is not None and len(df_js):
        synced = asof_join(synced, df_js, direction, tol_s)
    if df_sea is not None and len(df_sea):
        synced = asof_join(synced, df_sea, direction, tol_s)
    # Interpolate numeric gaps (bounded by limit_seconds)
    synced = interpolate_numeric(synced, interp_limit_s, interp_method)

    # Convenience fields + optional smoothing
    compute_convenience_cols(synced)
    add_distance_traveled(synced)
    rolling_mean(synced, ["v_cmd","v_actual"], smooth_win_s)

    # Coverage stats
    coverage = {}
    for key, cols in {
        "cmd": ["v_cmd_x","v_cmd_y"],
        "odom": ["vx","vy"],
        "gps": ["lat","lon"],
        "imu": ["roll","pitch","yaw"],
        "qwb": ["qw_WB","qx_WB","qy_WB","qz_WB"],
    }.items():
        present = [c for c in cols if c in synced]
        if present:
            coverage[key] = float(np.mean(np.all(synced[present].notna().values, axis=1)))

    # write outputs
    out_parquet = out_dir / f"synced_{int(hz)}Hz.parquet"
    synced.to_parquet(out_parquet)

    meta = {
        "mission_id": mission_id,
        "params": {
            "hz": hz, "direction": direction, "tolerance_ms": tol_ms,
            "interp_method": interp_method, "interp_limit_seconds": interp_limit_s,
            "lag_cmd_ms": lag_cmd_ms, "smooth_speed_window_s": smooth_win_s
        },
        "coverage": coverage,
        "inputs": { "tables_dir": str(tables_dir) },
        "rows": int(len(synced)),
        "t_start": float(synced["t"].iloc[0]),
        "t_end": float(synced["t"].iloc[-1]),
        "columns": sorted(list(synced.columns)),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[ok] wrote {out_parquet}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt

            t = synced["t"] - synced["t"].iloc[0]
            plt.figure(figsize=(9, 3.5))
            if "v_cmd" in synced:
                plt.plot(t, synced["v_cmd"], label="v_cmd")
            if "v_actual" in synced:
                plt.plot(t, synced["v_actual"], label="v_actual")
            plt.legend(); plt.xlabel("time [s]"); plt.tight_layout()

            # Put plots under reports/<mission_folder>/
            mission_folder = out_dir.name  # e.g., the same folder under SYNCED/
            rep = P["REPO_ROOT"] / "reports" / display_name
            rep.mkdir(parents=True, exist_ok=True)
            short_id = mission_id.split("-")[0]
            f = rep / f"{display_name}_{short_id}_sync_qc.png"
            plt.savefig(f, dpi=160)
            print(f"[ok] plot: {f}")
        except Exception as e:
            print(f"[warn] plot failed: {e}")

if __name__ == "__main__":
    main()
