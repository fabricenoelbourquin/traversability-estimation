#!/usr/bin/env python3
"""
Compare DEM-derived longitudinal/lateral slopes (as angles) with pitch/roll from the quaternion.

- DEM angles:
    dem_slope_long_deg ~ pitch (nose up +)
    dem_slope_lat_deg  ~ roll  (left up +)

- Quaternion angles (from q_WB in ENU, with body FLU: x=forward, y=left, z=up):
    pitch_quat_deg = asin(f_w_z) * 180/pi, where f_w = R(q_WB)*[1,0,0]
    roll_quat_deg  = asin(l_w_z) * 180/pi, where l_w = R(q_WB)*[0,1,0]

If DEM slope columns are missing in metrics, compute them on the fly from the DEM.

Usage:
  python src/visualization/compare_dem_vs_quat_pitch_roll.py --mission TRIM-1 --hz 10
  # optional DEM override:
  python src/visualization/compare_dem_vs_quat_pitch_roll.py --mission TRIM-1 --hz 10 --dem /path/to/dem.tif
"""

from __future__ import annotations
import argparse, math, sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import Affine
from pyproj import Transformer

# add repo src/ to path
THIS_FILE = Path(__file__).resolve()
SRC_ROOT = THIS_FILE.parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.paths import get_paths
from utils.missions import resolve_mission


# ---------- helpers (shared with your other scripts) ----------

def find_lat_lon_cols(df: pd.DataFrame) -> Tuple[str, str]:
    cand_lat = [c for c in df.columns if "lat" in c.lower()]
    cand_lon = [c for c in df.columns if "lon" in c.lower()]
    if not cand_lat or not cand_lon:
        raise SystemExit(f"Couldn’t find lat/lon columns. Found: lat={cand_lat}, lon={cand_lon}")
    return cand_lat[0], cand_lon[0]

def pick_synced_path(sync_dir: Path, hz: int | None) -> Path:
    if hz is not None:
        p = sync_dir / f"synced_{int(hz)}Hz.parquet"
        if not p.exists():
            raise SystemExit(f"Synced parquet not found: {p}")
        return p
    cands = sorted(sync_dir.glob("synced_*Hz.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise SystemExit(f"No synced_*Hz.parquet in {sync_dir}")
    return cands[0]

def normalize_quat_arrays(qw, qx, qy, qz):
    n = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    n[n == 0.0] = 1.0
    return qw/n, qx/n, qy/n, qz/n

def rotate_vec_with_quat(qw, qx, qy, qz, vx, vy, vz):
    """Vectorized rotation of v by unit quaternion q (active). Base->World."""
    qv = np.stack([qx, qy, qz], axis=1)           # (N,3)
    v  = np.tile(np.array([vx, vy, vz]), (len(qw), 1))
    t  = 2.0 * np.cross(qv, v)
    v2 = v + (qw[:, None] * t) + np.cross(qv, t)
    return v2

def world_to_rowcol(transform: Affine, x: np.ndarray, y: np.ndarray):
    a, _, c, _, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
    col_f = (x - c) / a
    row_f = (y - f) / e
    return row_f, col_f

def bilinear_sample(grid: np.ndarray, row_f: np.ndarray, col_f: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    i = np.floor(row_f).astype(np.int64)
    j = np.floor(col_f).astype(np.int64)

    valid = (i >= 0) & (i < H - 1) & (j >= 0) & (j < W - 1)
    out = np.full(row_f.shape, np.nan, dtype=np.float64)
    if not np.any(valid):
        return out

    iv, jv = i[valid], j[valid]
    di = (row_f[valid] - iv)
    dj = (col_f[valid] - jv)

    v00 = grid[iv,     jv    ]
    v10 = grid[iv + 1, jv    ]
    v01 = grid[iv,     jv + 1]
    v11 = grid[iv + 1, jv + 1]

    nanmask = np.isnan(v00) | np.isnan(v10) | np.isnan(v01) | np.isnan(v11)
    w00 = (1 - di) * (1 - dj)
    w10 = di       * (1 - dj)
    w01 = (1 - di) * dj
    w11 = di       * dj

    vals = v00*w00 + v10*w10 + v01*w01 + v11*w11
    tmp = np.full_like(vals, np.nan)
    tmp[~nanmask] = vals[~nanmask]
    out[valid] = tmp
    return out

def compute_dem_gradients(dem_path: Path):
    with rasterio.open(dem_path) as ds:
        z = ds.read(1).astype(np.float64)
        nodata = ds.nodata
        transform = ds.transform
        crs = ds.crs

    if nodata is not None:
        z = np.where(z == nodata, np.nan, z)

    dz_drow, dz_dcol = np.gradient(z, edge_order=2)
    a = transform.a   # pixel size in x (E)
    e = transform.e   # pixel size in y (N)  (negative for north-up)
    p = dz_dcol / a   # ∂z/∂E
    q = dz_drow / e   # ∂z/∂N
    return p, q, transform, crs


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Compare DEM-based pitch/roll vs quaternion pitch/roll over time.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--mission")
    g.add_argument("--mission-id")
    ap.add_argument("--hz", type=int, default=None, help="Which synced_<Hz>Hz.parquet to use (default: latest).")
    ap.add_argument("--dem", type=str, default=None, help="Path to DEM GeoTIFF (default: <mission>/maps/swisstopo/alti3d_chip512_gsd050.tif)")
    ap.add_argument("--use-metrics", action="store_true", default=True,
                    help="Read DEM slope angles from synced_<Hz>Hz_metrics.parquet if available (default: on).")
    ap.add_argument("--smooth-win", type=int, default=1, help="Optional rolling mean window for display (samples).")
    args = ap.parse_args()

    P = get_paths()
    mp = resolve_mission(args.mission or args.mission_id, P)
    sync_dir, display_name, map_dir = mp.synced, mp.display, mp.maps

    synced_path = pick_synced_path(sync_dir, args.hz)
    if args.hz is None:
        try:
            hz = int(synced_path.stem.split("_")[1].replace("Hz", ""))
        except Exception:
            hz = 10
    else:
        hz = int(args.hz)

    # Load synced
    df = pd.read_parquet(synced_path).sort_values("t").reset_index(drop=True)
    lat_col, lon_col = find_lat_lon_cols(df)

    # Quaternions
    qcols = ("qw_WB", "qx_WB", "qy_WB", "qz_WB")
    if not all(c in df.columns for c in qcols):
        alt = ("qw", "qx", "qy", "qz")
        if all(c in df.columns for c in alt):
            print("[warn] qw_WB..qz_WB not found — falling back to (qw,qx,qy,qz).")
            qcols = alt
        else:
            raise SystemExit("No quaternion columns found (need qw_WB..qz_WB or qw..qz).")

    qw, qx, qy, qz = (df[qcols[0]].to_numpy(),
                      df[qcols[1]].to_numpy(),
                      df[qcols[2]].to_numpy(),
                      df[qcols[3]].to_numpy())
    qw, qx, qy, qz = normalize_quat_arrays(qw, qx, qy, qz)

    # Quaternion-derived pitch/roll
    f_w = rotate_vec_with_quat(qw, qx, qy, qz, 1.0, 0.0, 0.0)  # forward (x_B) in world
    l_w = rotate_vec_with_quat(qw, qx, qy, qz, 0.0, 1.0, 0.0)  # left    (y_B) in world
    pitch_quat_deg = np.degrees(np.arcsin(np.clip(f_w[:, 2], -1.0, 1.0)))  # nose up positive
    roll_quat_deg  = np.degrees(np.arcsin(np.clip(l_w[:, 2], -1.0, 1.0)))  # left  up positive

    # DEM-derived angles: try to read from metrics parquet
    dem_long_deg = dem_lat_deg = None
    if args.use_metrics:
        metrics_path = sync_dir / f"synced_{hz}Hz_metrics.parquet"
        if metrics_path.exists():
            m = pd.read_parquet(metrics_path).sort_values("t")
            if {"dem_slope_long_deg", "dem_slope_lat_deg"}.issubset(m.columns):
                mm = m[["t", "dem_slope_long_deg", "dem_slope_lat_deg"]]
                df = df.merge(mm, on="t", how="left")
                dem_long_deg = df["dem_slope_long_deg"].to_numpy()
                dem_lat_deg  = df["dem_slope_lat_deg"].to_numpy()
            else:
                print(f"[info] {metrics_path} found but DEM columns missing; will compute from DEM.")
        else:
            print(f"[info] metrics parquet not found at {metrics_path}; will compute from DEM.")

    # If still missing, compute DEM slopes here
    if dem_long_deg is None or dem_lat_deg is None:
        dem_path = Path(args.dem) if args.dem else (map_dir / "swisstopo" / "alti3d_chip512_gsd050.tif")
        if not dem_path.exists():
            raise SystemExit(f"DEM not found: {dem_path}")
        p_grid, q_grid, transform, dem_crs = compute_dem_gradients(dem_path)

        # Heading ψ from quaternion (ENU): ψ = atan2(N, E) using forward vector
        psi = np.arctan2(f_w[:, 1], f_w[:, 0])
        cosψ, sinψ = np.cos(psi), np.sin(psi)

        # Transform GPS to DEM CRS and sample p,q
        lat = df[lat_col].to_numpy()
        lon = df[lon_col].to_numpy()
        transformer = Transformer.from_crs("EPSG:4326", dem_crs, always_xy=True)
        x_e, y_n = transformer.transform(lon, lat)
        row_f, col_f = world_to_rowcol(transform, x_e, y_n)
        p_s = bilinear_sample(p_grid, row_f, col_f)
        q_s = bilinear_sample(q_grid, row_f, col_f)

        g_long = p_s * cosψ + q_s * sinψ
        g_lat  = -p_s * sinψ + q_s * cosψ
        dem_long_deg = np.degrees(np.arctan(g_long))
        dem_lat_deg  = np.degrees(np.arctan(g_lat))

        df["dem_slope_long_deg"] = dem_long_deg
        df["dem_slope_lat_deg"]  = dem_lat_deg

    # Optional smoothing for nicer plots
    if args.smooth_win and args.smooth_win > 1:
        def smooth(a): return pd.Series(a).rolling(args.smooth_win, center=True, min_periods=1).mean().to_numpy()
        pitch_quat_deg = smooth(pitch_quat_deg)
        roll_quat_deg  = smooth(roll_quat_deg)
        dem_long_deg   = smooth(dem_long_deg)
        dem_lat_deg    = smooth(dem_lat_deg)

    # Valid samples (finite on both)
    mask_pitch = np.isfinite(dem_long_deg) & np.isfinite(pitch_quat_deg)
    mask_roll  = np.isfinite(dem_lat_deg)  & np.isfinite(roll_quat_deg)

    t_rel = df["t"].to_numpy() - float(df["t"].iloc[0])

    # Stats
    def stats(a, b, mask):
        d = (a - b)[mask]
        rmse = float(np.sqrt(np.nanmean(d*d))) if np.any(mask) else np.nan
        bias = float(np.nanmean(d)) if np.any(mask) else np.nan
        return rmse, bias

    rmse_pitch, bias_pitch = stats(dem_long_deg, pitch_quat_deg, mask_pitch)
    rmse_roll,  bias_roll  = stats(dem_lat_deg,  roll_quat_deg,  mask_roll)

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax = axes[0]
    ax.plot(t_rel[mask_pitch], dem_long_deg[mask_pitch],  label="DEM pitch (longitudinal)", linewidth=1.5)
    ax.plot(t_rel[mask_pitch], pitch_quat_deg[mask_pitch], label="Quat pitch", linewidth=1.0)
    ax.set_ylabel("Pitch [deg]")
    ax.set_title(f"Pitch — RMSE={rmse_pitch:.2f}°, bias={bias_pitch:+.2f}°")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(t_rel[mask_roll], dem_lat_deg[mask_roll],   label="DEM roll (lateral)", linewidth=1.5)
    ax.plot(t_rel[mask_roll], roll_quat_deg[mask_roll], label="Quat roll", linewidth=1.0)
    ax.set_ylabel("Roll [deg]")
    ax.set_xlabel("Time since start [s]")
    ax.set_title(f"Roll — RMSE={rmse_roll:.2f}°, bias={bias_roll:+.2f}°")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    out_dir = (Path(get_paths()["REPO_ROOT"]) / "reports" / mp.display)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{mp.display}_dem_vs_quat_pitch_roll.png"
    fig.savefig(out_png, dpi=150)
    print("[done]")
    print(f"  synced:  {synced_path}")
    print(f"  figure:  {out_png}")
    print(f"  pitch    RMSE={rmse_pitch:.3f}°, bias={bias_pitch:+.3f}°")
    print(f"  roll     RMSE={rmse_roll:.3f}°,  bias={bias_roll:+.3f}°")

if __name__ == "__main__":
    main()
