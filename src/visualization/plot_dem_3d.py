#!/usr/bin/env python3
"""
3D surface plot of the SwissALTI3D DEM patch, with optional trajectory overlay.

Usage:
  python src/visualization/plot_dem_3d.py --mission TRIM-1
  python src/visualization/plot_dem_3d.py --mission TRIM-1 --stride 1 --plot-trajectory --hz 10 --show
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.transform import Affine
from pyproj import Transformer
import pandas as pd
import matplotlib  # backend decided after parsing

# make src/ importable
THIS = Path(__file__).resolve()
SRC_ROOT = THIS.parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.paths import get_paths
from utils.missions import resolve_mission


# ---------- helpers ----------
def pick_synced_path(sync_dir: Path, hz: Optional[int]) -> Tuple[Path, int]:
    if hz is not None:
        p = sync_dir / f"synced_{int(hz)}Hz.parquet"
        if not p.exists():
            raise FileNotFoundError(p)
        return p, int(hz)
    cands = sorted(sync_dir.glob("synced_*Hz.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"No synced_*Hz.parquet in {sync_dir}")
    p = cands[0]
    try:
        inferred_hz = int(p.stem.split("_")[1].replace("Hz", ""))
    except Exception:
        inferred_hz = 10
    return p, inferred_hz

def find_lat_lon_cols(df: pd.DataFrame) -> Tuple[str, str]:
    lat_cols = [c for c in df.columns if "lat" in c.lower()]
    lon_cols = [c for c in df.columns if "lon" in c.lower()]
    if not lat_cols or not lon_cols:
        raise KeyError(f"Couldn't find lat/lon columns. Found lat={lat_cols}, lon={lon_cols}")
    return lat_cols[0], lon_cols[0]

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
    if not np.any(valid):  # quick exit
        return out
    iv, jv = i[valid], j[valid]
    di = (row_f[valid] - iv)
    dj = (col_f[valid] - jv)
    v00 = grid[iv,     jv    ]
    v10 = grid[iv + 1, jv    ]
    v01 = grid[iv,     jv + 1]
    v11 = grid[iv + 1, jv + 1]
    nanmask = np.isnan(v00) | np.isnan(v10) | np.isnan(v01) | np.isnan(v11)
    w00 = (1 - di) * (1 - dj); w10 = di * (1 - dj)
    w01 = (1 - di) * dj;       w11 = di * dj
    vals = v00*w00 + v10*w10 + v01*w01 + v11*w11
    tmp = np.full_like(vals, np.nan)
    tmp[~nanmask] = vals[~nanmask]
    out[valid] = tmp
    return out


def main():
    ap = argparse.ArgumentParser(description="3D DEM with optional trajectory overlay.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--mission"); g.add_argument("--mission-id")
    ap.add_argument("--dem", type=str, default=None,
                    help="Override DEM; default: <mission>/maps/swisstopo/alti3d_chip512_gsd050.tif")
    ap.add_argument("--stride", type=int, default=1, help="Downsample factor for plotting (>=1).")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--azim", type=float, default=None,
                    help="Override azimuth; default automatically picks the lowest DEM corner to face the camera.")
    ap.add_argument("--elev", type=float, default=45)
    ap.add_argument("--alpha", type=float, default=0.92, help="Surface alpha (helps see the path).")
    ap.add_argument("--show", action="store_true", default=False, help="Open interactive window.")
    # Trajectory
    ap.add_argument("--plot-trajectory", action="store_true", default=False)
    ap.add_argument("--hz", type=int, default=None)
    ap.add_argument("--traj-skip", type=int, default=1, help="Plot every Nth trajectory sample.")
    ap.add_argument("--traj-offset", type=float, default=0.02, help="Meters to lift path above surface.")
    ap.add_argument("--traj-width", type=float, default=1.2)
    ap.add_argument("--traj-color", type=str, default="crimson")
    args = ap.parse_args()

    # Backend: interactive only if --show
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource

    P = get_paths()
    mp = resolve_mission(args.mission or args.mission_id, P)
    dem_path = Path(args.dem) if args.dem else (mp.maps / "swisstopo" / "alti3d_chip512_gsd050.tif")
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM not found: {dem_path}")

    with rasterio.open(dem_path) as ds:
        z = ds.read(1).astype(np.float64)
        nodata = ds.nodata
        transform = ds.transform
        dem_crs = ds.crs
        pix_east, pix_north = ds.transform.a, abs(ds.transform.e)

    if nodata is not None:
        z = np.where(z == nodata, np.nan, z)

    # grid
    H, W = z.shape
    j = np.arange(W); i = np.arange(H)
    x = transform.c + j * transform.a
    y = transform.f + i * transform.e
    X, Y = np.meshgrid(x, y)

    s = max(1, int(args.stride))
    Xs, Ys, Zs = X[::s, ::s], Y[::s, ::s], z[::s, ::s]

    fig = plt.figure(figsize=(10, 8), dpi=args.dpi)
    ax = fig.add_subplot(111, projection="3d")

    # hillshade
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(Zs, cmap=plt.get_cmap("terrain"), vert_exag=1, blend_mode="soft")

    ax.plot_surface(Xs, Ys, Zs,
                    rstride=1, cstride=1,
                    facecolors=rgb, shade=False,
                    linewidth=0, antialiased=False,
                    alpha=float(args.alpha))

    # preserve metric scaling (makes XY look square if pixels are square)
    try:
        ax.set_box_aspect((Xs.ptp(), Ys.ptp(), np.nanptp(Zs) if np.isfinite(Zs).any() else 1.0))
    except Exception:
        pass

    # enforce a minimum z-span so axes stay readable on flat terrain
    if np.isfinite(Zs).any():
        z_min = np.nanmin(Zs)
        z_max = np.nanmax(Zs)
        current_span = z_max - z_min
        target_span = max(current_span, 100.0)
        pad = max(0.0, (target_span - current_span) / 2.0)
        ax.set_zlim(z_min - pad, z_max + pad)

    # trajectory overlay
    if args.plot_trajectory:
        synced_path, hz = pick_synced_path(mp.synced, args.hz)
        df = pd.read_parquet(synced_path).sort_values("t")
        lat_col, lon_col = find_lat_lon_cols(df)
        lat = df[lat_col].to_numpy(); lon = df[lon_col].to_numpy()

        transformer = Transformer.from_crs("EPSG:4326", dem_crs, always_xy=True)
        x_e, y_n = transformer.transform(lon, lat)

        row_f, col_f = world_to_rowcol(transform, x_e, y_n)
        z_traj = bilinear_sample(z, row_f, col_f)

        valid = np.isfinite(x_e) & np.isfinite(y_n) & np.isfinite(z_traj)
        valid_idx = np.nonzero(valid)[0]
        print(f"[traj] total={len(lat)} inside_dem={valid_idx.size}")

        if valid_idx.size > 1:
            if args.traj_skip > 1:
                valid_idx = valid_idx[::int(max(1, args.traj_skip))]
            z_plot = z_traj[valid_idx] + float(args.traj_offset)
            # thick, bright line so it stands out; plot last so it draws on top
            ax.plot3D(x_e[valid_idx], y_n[valid_idx], z_plot,
                      color=args.traj_color, linewidth=float(args.traj_width), alpha=1.0, zorder=10)
            # endpoints
            ax.scatter3D(x_e[valid_idx[0]], y_n[valid_idx[0]], z_plot[0],  c="lime", s=30, depthshade=False)
            ax.scatter3D(x_e[valid_idx[-1]], y_n[valid_idx[-1]], z_plot[-1], c="red",  s=30, depthshade=False)
        else:
            print("[traj] no points fall within the DEM extent — nothing to draw.")

    def pick_auto_azim(Xg: np.ndarray, Yg: np.ndarray, Zg: np.ndarray) -> float:
        """Aim camera toward the lowest corner so high terrain doesn't occlude the rest."""
        corners = [
            (0, 0),  # top-left
            (0, -1),  # top-right
            (-1, 0),  # bottom-left
            (-1, -1),  # bottom-right
        ]
        vals = []
        for ci, cj in corners:
            zc = Zg[ci, cj]
            if np.isnan(zc):
                continue
            xc = Xg[ci, cj]
            yc = Yg[ci, cj]
            vals.append((zc, xc, yc))
        if not vals:
            return -60.0
        z_min, xc, yc = min(vals, key=lambda t: t[0])
        x0 = np.nanmean(Xg)
        y0 = np.nanmean(Yg)
        dx = xc - x0
        dy = yc - y0
        if dx == 0 and dy == 0:
            return -60.0
        angle = np.degrees(np.arctan2(dy, dx))
        return float(angle)

    azim = args.azim if args.azim is not None else pick_auto_azim(Xs, Ys, Zs)
    ax.view_init(elev=float(args.elev), azim=float(azim))
    ax.set_xlabel("Easting [m]"); ax.set_ylabel("Northing [m]"); ax.set_zlabel("Elevation [m]")
    ax.set_title(f"{mp.display} — DEM 3D")
    fig.tight_layout()

    print("grid full:", z.shape, "after stride:", Zs.shape, "pixel size (m):", pix_east, pix_north)

    out_dir = Path(P["REPO_ROOT"]) / "reports" / mp.display
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{mp.display}_dem3d.png"
    fig.savefig(out_png)
    print(f"[ok] saved 3D DEM figure to {out_png}")

    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
