#!/usr/bin/env python3
"""
Compare heading from q_WB vs path yaw from GPS using synced parquet.
Now: heading is fixed as +x_B rotated by q_WB (no axis search).
We instead allow rotating the GPS path tangent by {0, 90, 180, 270} deg
to correct axis mistakes (with --gps-rot auto to pick the best).

Usage:
  python src/visualization/compare_yaw_measurements.py --mission TRIM-1
  # choose specific synced parquet
  python src/visualization/compare_yaw_measurements.py --mission TRIM-1 --hz 10
  # force a GPS rotation (default: auto)
  python src/visualization/compare_yaw_measurements.py --mission TRIM-1 --gps-rot 90
"""

from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS_FILE = Path(__file__).resolve()
SRC_ROOT = THIS_FILE.parents[1]    # .../src
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from utils.paths import get_paths
from utils.missions import resolve_mission

# ---------------- helpers ----------------

def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def unwrap(a: np.ndarray, origin: float | None = None) -> np.ndarray:
    au = np.unwrap(a)
    if origin is not None and len(au) > 0:
        au = au + (origin - au[0])
    return au

def latlon_to_local_enu(lats: np.ndarray, lons: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    # local tangent plane approx around first sample (sufficient for short missions)
    R = 6378137.0
    lat0 = np.deg2rad(lats[0]); lon0 = np.deg2rad(lons[0])
    lat = np.deg2rad(lats);     lon = np.deg2rad(lons)
    x = (lon - lon0) * math.cos(lat0) * R   # East
    y = (lat - lat0) * R                    # North
    return x, y

def path_dxdy_from_latlon(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    x, y = latlon_to_local_enu(lat, lon)
    dx = np.gradient(x); dy = np.gradient(y)
    return dx, dy

def yaw_from_dxdy(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    # ENU yaw: atan2(N, E) = atan2(dy, dx)
    return np.arctan2(dy, dx)

def rotate_dxdy(dx: np.ndarray, dy: np.ndarray, rot_deg: int) -> tuple[np.ndarray, np.ndarray]:
    """Rotate (dx,dy) in the ENU plane by multiples of 90° (right-hand, CCW)."""
    r = ((rot_deg % 360) + 360) % 360
    if r == 0:
        return dx, dy
    elif r == 90:
        # [dx'; dy'] = R90 * [dx; dy] with R90 = [[0,-1],[1,0]]
        return -dy, dx
    elif r == 180:
        return -dx, -dy
    elif r == 270:
        return dy, -dx
    else:
        raise ValueError("rot_deg must be one of {0,90,180,270}")

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

def rolling_smooth_angle(a: np.ndarray, win_median: int, win_mean: int) -> np.ndarray:
    """Unwrap -> rolling median -> rolling mean -> rewrap."""
    au = np.unwrap(a)
    s = pd.Series(au)
    if win_median > 1:
        s = s.rolling(win_median, center=True, min_periods=1).median()
    if win_mean > 1:
        s = s.rolling(win_mean, center=True, min_periods=1).mean()
    return np.vectorize(wrap_pi)(s.to_numpy())

def circ_rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = np.vectorize(wrap_pi)(a - b)
    return float(np.sqrt(np.nanmean(d*d)))

def find_lat_lon_cols(df: pd.DataFrame) -> tuple[str,str]:
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
    # choose latest by mtime
    cands = sorted(sync_dir.glob("synced_*Hz.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise SystemExit(f"No synced_*Hz.parquet in {sync_dir}")
    return cands[0]

# -------------- main --------------

def main():
    ap = argparse.ArgumentParser(description="Compare heading (from q_WB) vs path yaw (from GPS) using synced parquet.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--mission"); g.add_argument("--mission-id")
    ap.add_argument("--hz", type=int, default=None, help="Which synced_<Hz>Hz.parquet to use (default: latest).")
    ap.add_argument("--min-speed", type=float, default=0.2, help="Ignore samples slower than this [m/s].")
    ap.add_argument("--yaw-med-win", type=int, default=11, help="Rolling median window for GPS yaw.")
    ap.add_argument("--yaw-mean-win", type=int, default=11, help="Rolling mean window for GPS yaw.")
    ap.add_argument("--smooth-tf-win", type=int, default=5, help="Rolling mean window for TF heading (0=off).")
    ap.add_argument("--out-prefix", default=None, help="Output PNG prefix (path without extension).")
    # Deprecated: we no longer search body axes; heading is fixed to +x_B.
    ap.add_argument("--axis", choices=["x"], default="x",
                    help="Deprecated. Heading is fixed to +x_B rotated by q_WB.")
    ap.add_argument("--gps-rot", choices=["auto","0","90","180","270"], default="auto",
                    help="Rotate GPS path tangent by this many degrees CCW before computing yaw.")
    ap.add_argument("--unwrap-plot", action="store_true", default=False,
                    help="Plot continuous unwrapped yaw for visual comparison.")
    ap.add_argument("--unwrap-origin", type=float, default=0.0,
                    help="Radians; anchor unwrapping so first sample ~= this value.")
    args = ap.parse_args()

    P = get_paths()
    mp = resolve_mission(args.mission or args.mission_id, P)
    sync_dir, short, display_name, map_dir = mp.synced, mp.folder, mp.display, mp.maps
    out_dir_default = P["REPO_ROOT"] / "reports" / display_name
    out_dir_default.mkdir(parents=True, exist_ok=True)

    synced_path = pick_synced_path(sync_dir, args.hz)
    out_prefix = args.out_prefix or str(out_dir_default / f"{display_name}_compare_yaw")

    df = pd.read_parquet(synced_path).sort_values("t").reset_index(drop=True)

    # --- Acquire lat/lon + quaternions
    lat_col, lon_col = find_lat_lon_cols(df)
    df = df.loc[df["t"].notna() & df[lat_col].notna() & df[lon_col].notna(), ["t", lat_col, lon_col] +
                [c for c in df.columns if c.startswith("qw") or c.startswith("qx") or c.startswith("qy") or c.startswith("qz")]]

    use_block = ("qw_WB","qx_WB","qy_WB","qz_WB")
    if not all(c in df.columns for c in use_block):
        alt = ("qw","qx","qy","qz")
        if all(c in df.columns for c in alt):
            print("[warn] qw_WB..qz_WB not found — falling back to (qw,qx,qy,qz). "
                  "This is likely base-in-odom, not ENU world.")
            use_block = alt
        else:
            raise SystemExit("No quaternion columns found (need qw_WB..qz_WB or qw..qz).")

    # --- Pull arrays
    t   = df["t"].to_numpy()
    lat = df[lat_col].to_numpy(); lon = df[lon_col].to_numpy()
    qw, qx, qy, qz = (df[use_block[0]].to_numpy(),
                      df[use_block[1]].to_numpy(),
                      df[use_block[2]].to_numpy(),
                      df[use_block[3]].to_numpy())
    qw, qx, qy, qz = normalize_quat_arrays(qw, qx, qy, qz)

    # --- Heading from q_WB: FIXED +x_B axis
    v_world = rotate_vec_with_quat(qw, qx, qy, qz, 1.0, 0.0, 0.0)   # +x_B -> ENU
    yaw_h = np.arctan2(v_world[:,1], v_world[:,0])                  # ENU: atan2(N,E)
    if args.smooth_tf_win and args.smooth_tf_win > 1:
        s = pd.Series(np.unwrap(yaw_h))
        s = s.rolling(args.smooth_tf_win, center=True, min_periods=1).mean()
        yaw_h = np.vectorize(wrap_pi)(s.to_numpy())

    # --- GPS path yaw + speed gate (before rotation)
    dx, dy = path_dxdy_from_latlon(lat, lon)
    dt    = np.gradient(t)
    speed = np.hypot(dx, dy) / np.maximum(dt, 1e-6)

    moving = speed >= args.min_speed
    if moving.sum() < 10:
        raise SystemExit("Too few moving samples; lower --min-speed.")

    # --- Choose GPS rotation (applied to dx,dy) to minimize circ RMSE vs heading
    rotations = [0, 90, 180, 270] if args.gps_rot == "auto" else [int(args.gps_rot)]
    best = None
    for rot in rotations:
        dx_r, dy_r = rotate_dxdy(dx, dy, rot)
        yaw_path = yaw_from_dxdy(dx_r, dy_r)
        yaw_path_s = rolling_smooth_angle(yaw_path, args.yaw_med_win, args.yaw_mean_win)
        err = float(circ_rmse(yaw_h[moving], yaw_path_s[moving]))
        if best is None or err < best["err"]:
            best = {"rot": rot, "yaw_path_s": yaw_path_s, "err": err}

    gps_rot_deg = best["rot"]
    yaw_path_s = best["yaw_path_s"]

    # --- Plot (moving-only)
    t_rel = (t - t[0])[moving]
    yaw_h_plot    = yaw_h[moving]
    yaw_path_plot = yaw_path_s[moving]

    if args.unwrap_plot:
        yaw_h_plot    = unwrap(yaw_h_plot, origin=args.unwrap_origin)
        yaw_path_plot = unwrap(yaw_path_plot, origin=args.unwrap_origin)

    yaw_diff_deg = np.rad2deg(np.vectorize(wrap_pi)(yaw_h[moving] - yaw_path_s[moving]))

    fig1 = plt.figure(figsize=(10,4))
    plt.plot(t_rel, np.rad2deg(yaw_h_plot),    label="heading from q_WB (+x_B)")
    plt.plot(t_rel, np.rad2deg(yaw_path_plot), label=f"path yaw from GPS (rot={gps_rot_deg}°)")
    plt.xlabel("Time since start [s]"); plt.ylabel("Yaw [deg] (ENU: 0=East, +90=North)")
    plt.legend(); plt.title("Heading vs Path yaw (moving only)")
    fig1.tight_layout(); plt.savefig(f"{out_prefix}_timeseries.png", dpi=150)

    mu = float(np.nanmean(yaw_diff_deg)); sig = float(np.nanstd(yaw_diff_deg))
    L = float(np.nanmax(np.abs(yaw_diff_deg)))
    L = 5.0 * math.ceil(L / 5.0)

    fig2 = plt.figure(figsize=(5,4))
    nbins = max(20, min(120, int(2 * L)))
    edges = np.linspace(-L, L, nbins + 1)
    plt.hist(yaw_diff_deg, bins=edges)
    plt.xlim(-L, L)
    plt.axvline(0, lw=1, alpha=0.6)  # 0° reference
    plt.xlabel("Yaw difference [deg] (TF - GPS)"); plt.ylabel("Count")
    title = f"Yaw diff ~ N({mu:.1f}, {sig:.1f}) — GPS rot={gps_rot_deg}°, RMSE={math.degrees(best['err']):.1f}°"
    plt.title(title)
    fig2.tight_layout(); plt.savefig(f"{out_prefix}_hist.png", dpi=150)

    print("[done]")
    print(f"  synced: {synced_path}")
    print(f"  reports: {out_prefix}_timeseries.png")
    print(f"           {out_prefix}_hist.png")
    print(f"  moving samples: {moving.sum()}")
    print(f"  GPS rotation chosen: {gps_rot_deg} deg")
    print(f"  circular RMSE (moving): {math.degrees(best['err']):.2f} deg")

if __name__ == "__main__":
    main()
