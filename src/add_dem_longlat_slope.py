#!/usr/bin/env python3
"""
Compute longitudinal & lateral slope from a SwissALTI3D DEM along a mission path,
using robot heading from q_WB (+x_B) and append results to synced_<Hz>Hz_metrics.parquet.

Definitions (ENU):
  DEM gradients: p = ∂z/∂E, q = ∂z/∂N   [rise/run, unitless]
  Heading (yaw ψ) from quaternion rotating +x_B into ENU:
    t = [cos ψ, sin ψ] (tangent/forward), s = [-sin ψ, cos ψ] (leftward)
  Longitudinal slope  g_parallel = p*cos ψ + q*sin ψ
  Lateral (cross)     g_lateral  = -p*sin ψ + q*cos ψ
  Angle versions: atan(g_parallel), atan(g_lateral) in degrees.

Saves columns to metrics parquet:
  dem_grad_e, dem_grad_n, dem_slope_long, dem_slope_lat,
  dem_slope_long_deg, dem_slope_lat_deg

Usage:
  python src/add_dem_longlat_slope.py --mission TRIM-1 --hz 10
  # or by mission-id:
  python src/add_dem_longlat_slope.py --mission-id e97e35ad-... --hz 10
  # optional DEM override:
  python src/add_dem_longlat_slope.py --mission TRIM-1 --dem /path/to/dem.tif
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
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


# ---------------- helpers ----------------

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
    cands = sorted(sync_dir.glob("synced_*Hz.parquet"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
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

def atan_deg_safe(x: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan(x))

def world_to_rowcol(transform: Affine, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert world (map) coords to fractional (row, col) indices for a north-up geotransform.
    For standard Rasterio Affine with:
        x = a*col + c
        y = e*row + f    (e < 0 for north-up)
    """
    a, _, c, _, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
    col_f = (x - c) / a
    row_f = (y - f) / e
    return row_f, col_f

def bilinear_sample(grid: np.ndarray, row_f: np.ndarray, col_f: np.ndarray) -> np.ndarray:
    """
    Bilinear sample grid at fractional (row_f, col_f). Returns NaN where out-of-bounds or neighbors NaN.
    """
    H, W = grid.shape
    i = np.floor(row_f).astype(np.int64)
    j = np.floor(col_f).astype(np.int64)

    valid = (i >= 0) & (i < H - 1) & (j >= 0) & (j < W - 1)

    out = np.full(row_f.shape, np.nan, dtype=np.float64)
    if not np.any(valid):
        return out

    iv = i[valid]; jv = j[valid]
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

# ---------------- core ----------------

def compute_dem_gradients(dem_path: Path):
    """
    Load DEM and compute p=∂z/∂E, q=∂z/∂N (unitless rise/run) on the DEM grid.
    Returns (p_grid, q_grid, transform, crs)
    """
    with rasterio.open(dem_path) as ds:
        z = ds.read(1).astype(np.float64)
        nodata = ds.nodata
        transform = ds.transform
        crs = ds.crs

    # mask nodata as NaN
    if nodata is not None:
        z = np.where(z == nodata, np.nan, z)

    # array gradients wrt row/col (pixels)
    dz_drow, dz_dcol = np.gradient(z, edge_order=2)  # NaNs will propagate

    # Convert to ∂/∂E and ∂/∂N using affine.
    # For north-up: x = a*col + c, y = e*row + f (e < 0)
    a = transform.a
    e = transform.e
    # ∂z/∂x = dz/dcol * ∂col/∂x = dz/dcol * (1/a)
    # ∂z/∂y = dz/drow * ∂row/∂y = dz/drow * (1/e)
    p = dz_dcol / a
    q = dz_drow / e  # note e<0 => flips sign so q is ∂/∂North

    return p, q, transform, crs

def main():
    ap = argparse.ArgumentParser(description="Add DEM-based longitudinal & lateral slopes to metrics parquet.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--mission")
    g.add_argument("--mission-id")
    ap.add_argument("--hz", type=int, default=None, help="Which synced_<Hz>Hz.parquet to use (default: latest).")
    ap.add_argument("--dem", type=str, default=None, help="Path to DEM GeoTIFF (default: <mission>/maps/swisstopo/alti3d_chip512_gsd050.tif)")
    ap.add_argument("--min-speed", type=float, default=0.0, help="Optional future filter; unused here.")
    ap.add_argument("--write-to", choices=["metrics", "synced"], default="metrics",
                    help="Append to synced_<Hz>Hz_metrics.parquet (default) or overwrite synced_<Hz>Hz.parquet.")
    args = ap.parse_args()

    P = get_paths()
    mp = resolve_mission(args.mission or args.mission_id, P)
    sync_dir, short, display_name, map_dir = mp.synced, mp.folder, mp.display, mp.maps

    synced_path = pick_synced_path(sync_dir, args.hz)
    if args.hz is None:
        # Try to infer Hz for naming metrics file
        try:
            hz = int(synced_path.stem.split("_")[1].replace("Hz", ""))
        except Exception:
            hz = 10
    else:
        hz = int(args.hz)

    dem_path = Path(args.dem) if args.dem else (map_dir / "swisstopo" / "alti3d_chip512_gsd050.tif")
    if not dem_path.exists():
        raise SystemExit(f"DEM not found: {dem_path}")

    print(f"[load] synced: {synced_path}")
    df = pd.read_parquet(synced_path).sort_values("t").reset_index(drop=True)

    # Acquire lat/lon and q_WB
    lat_col, lon_col = find_lat_lon_cols(df)
    cols_needed = ["t", lat_col, lon_col]
    # Try standard q_WB naming; fall back to generic qw..qz
    qcols = ("qw_WB", "qx_WB", "qy_WB", "qz_WB")
    if not all(c in df.columns for c in qcols):
        alt = ("qw", "qx", "qy", "qz")
        if all(c in df.columns for c in alt):
            print("[warn] qw_WB..qz_WB not found — falling back to (qw,qx,qy,qz).")
            qcols = alt
        else:
            raise SystemExit("No quaternion columns found (need qw_WB..qz_WB or qw..qz).")
    cols_needed += list(qcols)

    df = df.loc[df["t"].notna() & df[lat_col].notna() & df[lon_col].notna(), cols_needed].copy()
    t = df["t"].to_numpy()
    lat = df[lat_col].to_numpy()
    lon = df[lon_col].to_numpy()
    qw, qx, qy, qz = (df[qcols[0]].to_numpy(),
                      df[qcols[1]].to_numpy(),
                      df[qcols[2]].to_numpy(),
                      df[qcols[3]].to_numpy())
    qw, qx, qy, qz = normalize_quat_arrays(qw, qx, qy, qz)

    # Heading from quaternion: +x_B rotated into ENU, yaw = atan2(N, E)
    v_world = rotate_vec_with_quat(qw, qx, qy, qz, 1.0, 0.0, 0.0)  # +x_B in world
    E = v_world[:, 0]; N = v_world[:, 1]
    psi = np.arctan2(N, E)  # ENU yaw

    # Load DEM & gradients
    print(f"[load] DEM: {dem_path}")
    p_grid, q_grid, transform, dem_crs = compute_dem_gradients(dem_path)

    # Transform GPS (WGS84) to DEM CRS
    transformer = Transformer.from_crs("EPSG:4326", dem_crs, always_xy=True)
    x_e, y_n = transformer.transform(lon, lat)  # note always_xy => (lon, lat)

    # Map world (x,y) to fractional (row,col) and bilinear sample p,q
    row_f, col_f = world_to_rowcol(transform, np.asarray(x_e), np.asarray(y_n))
    p_s = bilinear_sample(p_grid, row_f, col_f)  # ∂z/∂E
    q_s = bilinear_sample(q_grid, row_f, col_f)  # ∂z/∂N

    # Longitudinal & lateral slopes
    cosψ = np.cos(psi); sinψ = np.sin(psi)
    g_long = p_s * cosψ + q_s * sinψ
    g_lat  = -p_s * sinψ + q_s * cosψ

    out = pd.DataFrame({
        "t": t,
        "dem_grad_e": p_s,           # ∂z/∂E
        "dem_grad_n": q_s,           # ∂z/∂N
        "dem_slope_long": g_long,    # unitless grade
        "dem_slope_lat":  g_lat,
        "dem_slope_long_deg": atan_deg_safe(g_long),
        "dem_slope_lat_deg":  atan_deg_safe(g_lat),
    })

    # Merge to metrics or synced
    if args.write_to == "metrics":
        metrics_path = sync_dir / f"synced_{hz}Hz_metrics.parquet"
        if metrics_path.exists():
            m = pd.read_parquet(metrics_path)
            m = m.merge(out, on="t", how="outer", suffixes=("", "_dup"))
            # drop any accidental *_dup columns (keep new ones)
            drop_dups = [c for c in m.columns if c.endswith("_dup")]
            m = m.drop(columns=drop_dups)
        else:
            m = out
        m = m.sort_values("t").reset_index(drop=True)
        m.to_parquet(metrics_path, index=False)
        print(f"[save] {metrics_path}  (rows: {len(m)})")
    else:
        # Overwrite synced with new columns
        df_full = pd.read_parquet(synced_path)
        merged = df_full.merge(out, on="t", how="left")
        merged.to_parquet(synced_path, index=False)
        print(f"[save] {synced_path}  (rows: {len(merged)})")

    # Quick stats
    ok = np.isfinite(out["dem_slope_long"]).sum()
    total = len(out)
    print(f"[stats] valid samples: {ok}/{total} ({100.0*ok/total:.1f}%)")
    print("[done]")

if __name__ == "__main__":
    main()
