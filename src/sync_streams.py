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
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yaml

# --- path helpers ---

from utils.paths import get_paths

# Resolve common repo paths once at import time. Expects keys:
# RAW / TABLES / SYNCED / REPO_ROOT / MISSIONS_JSON
P = get_paths()

# --- helpers ---
def load_yaml(p: Path) -> dict:
    """
    Load a YAML file if it exists, else return an empty dict.
    """
    return yaml.safe_load(p.read_text()) if p.exists() else {}

def resolve_mission_folder(mission: str) -> Path:
    """
    Resolve a mission alias or UUID into:
      (tables_dir, synced_out_dir, mission_id)

    Looks up config/missions.json:
      - If `mission` is an alias, map it to its mission_id via _aliases.
      - Otherwise treat `mission` as the mission_id.
    Then uses the mission entry's 'folder' to construct data paths.

    Raises:
      FileNotFoundError if missions.json is missing or tables dir not found.
      KeyError if mission/alias is not in missions.json.
    """
    mj = P["REPO_ROOT"] / "config" / "missions.json"
    if not mj.exists():
        raise FileNotFoundError(f"{mj} not found (run download first or create manually).")
    meta = json.loads(mj.read_text())
    mission_id = mission
    # Map alias -> mission_id if needed
    aliases = meta.get("_aliases", {})
    if mission in aliases:
        mission_id = aliases[mission]
    # Resolve folder
    entry = meta.get(mission_id)
    if not entry:
        raise KeyError(f"Mission '{mission}' not in missions.json")
    folder = entry["folder"]
    tables_dir = P["TABLES"] / folder
    if not tables_dir.exists():
        raise FileNotFoundError(f"{tables_dir} not found. Did you run extract?")
    return tables_dir, (P["SYNCED"] / folder), mission_id

def load_df(path: Path) -> Optional[pd.DataFrame]:
    """
    Load a parquet file (if present) and normalize the time column:
      - keep only rows with finite 't'
      - cast 't' to float seconds
      - sort by 't'
      - drop duplicate timestamps (keep last)
    Returns None if the file does not exist.
    """
    if path.exists():
        df = pd.read_parquet(path)
        # Ensure float seconds & monotonic t
        df = df[df["t"].notna()].copy()
        df["t"] = df["t"].astype(float)
        df = df.sort_values("t").drop_duplicates("t", keep="last")
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
    Uses numeric tolerance for numeric keys; Timedelta for datetimelike keys.
    """
    left = base.sort_values("t")
    right = df.sort_values("t")

    # Decide tolerance type based on dtype of 't'
    t_dtype = left["t"].dtype
    if np.issubdtype(t_dtype, np.number):
        tol = tol_s  # numeric seconds for numeric 't'
    else:
        tol = pd.Timedelta(seconds=tol_s)  # for datetime64[ns] keys

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
    tables_dir, out_dir, mission_id = resolve_mission_folder(args.mission)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load available tables
    df_cmd  = load_df(tables_dir / "cmd_vel.parquet")
    df_odom = load_df(tables_dir / "odom.parquet")
    df_gps  = load_df(tables_dir / "gps.parquet")
    df_imu  = load_df(tables_dir / "imu.parquet")  # optional

    dfs = [d for d in [df_cmd, df_odom, df_gps, df_imu] if d is not None]

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

    # Interpolate numeric gaps (bounded by limit_seconds)
    synced = interpolate_numeric(synced, interp_limit_s, interp_method)

    # Convenience fields + optional smoothing
    compute_convenience_cols(synced)
    rolling_mean(synced, ["v_cmd","v_actual"], smooth_win_s)

    # Coverage stats
    coverage = {}
    for key, cols in {
        "cmd": ["v_cmd_x","v_cmd_y"],
        "odom": ["vx","vy"],
        "gps": ["lat","lon"],
        "imu": ["roll","pitch","yaw"],
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
            rep = P["REPO_ROOT"] / "reports" / mission_folder
            rep.mkdir(parents=True, exist_ok=True)

            f = rep / f"{mission_id.split('-')[0]}_sync_qc.png"
            plt.savefig(f, dpi=160)
            print(f"[ok] plot: {f}")
        except Exception as e:
            print(f"[warn] plot failed: {e}")

if __name__ == "__main__":
    main()
