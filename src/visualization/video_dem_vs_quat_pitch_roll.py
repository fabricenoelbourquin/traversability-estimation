#!/usr/bin/env python3
"""
Render a side-by-side video:

LEFT:  forward camera (from ROS1 bag)
RIGHT: two plots comparing DEM-derived vs quaternion-derived angles
       - Top:   Pitch  (DEM longitudinal vs quat pitch)
       - Bottom:Roll   (DEM lateral      vs quat roll)
       Each plot includes a red vertical time cursor synced to the video.

Usage:
  python src/visualization/video_dem_vs_quat_pitch_roll.py \
      --mission TRIM-1 \
      --camera-pattern "*_hdr_front.bag" \
      --camera-topic /boxi/hdr/front/image_raw/compressed \
      --hz 10 \
      --fps 15

Notes:
- If dem_slope_* columns aren't in synced_<Hz>Hz_metrics.parquet,
  slopes are computed from the DEM on-the-fly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import Affine
from pyproj import Transformer

from rosbags.highlevel import AnyReader
from rosbags.image import message_to_cvimage

# make src/ importable
THIS = Path(__file__).resolve()
SRC_ROOT = THIS.parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.paths import get_paths
from utils.missions import resolve_mission


# ------------------ helpers ------------------

def filter_valid_rosbags(paths: list[Path]) -> list[Path]:
    keep: list[Path] = []
    for p in paths:
        if not p.is_file() or p.name.startswith("._"):
            continue
        try:
            with p.open("rb") as f:
                head = f.read(13)
            if head.startswith(b"#ROSBAG V2.0"):
                keep.append(p)
        except OSError:
            continue
    return keep

def find_camera_bag(raw_dir: Path, pattern: str, topic: str) -> Path:
    matches = sorted(raw_dir.glob(pattern))
    matches = filter_valid_rosbags(matches)
    tried: list[str] = []
    for p in matches:
        try:
            with AnyReader([p]) as r:
                topics = {c.topic for c in r.connections}
            if topic in topics:
                return p
            if topic.endswith("/image_raw") and topic + "/compressed" in topics:
                return p
            tried.append(f"{p.name}: {sorted(topics)}")
        except UnicodeDecodeError:
            tried.append(f"{p.name}: <cannot open>")
    raise FileNotFoundError(
        f"Could not find bag in {raw_dir} matching {pattern!r} that has topic {topic!r}.\n"
        "Tried:\n" + "\n".join(tried)
    )

def guess_time_col(df: pd.DataFrame) -> str:
    for c in ("t", "time", "stamp", "timestamp", "time_s", "ros_time"):
        if c in df.columns:
            return c
    raise KeyError("No time column found in parquet.")

def pick_synced(sync_dir: Path, hz: Optional[int]) -> Tuple[Path, int]:
    if hz is not None:
        p = sync_dir / f"synced_{hz}Hz.parquet"
        if not p.exists():
            raise FileNotFoundError(p)
        return p, int(hz)
    cands = sorted(sync_dir.glob("synced_*Hz.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"No synced_*Hz.parquet in {sync_dir}")
    p = cands[0]
    try:
        hz = int(p.stem.split("_")[1].replace("Hz", ""))
    except Exception:
        hz = 10
    return p, int(hz)

def normalize_quat_arrays(qw, qx, qy, qz):
    n = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    n[n == 0.0] = 1.0
    return qw/n, qx/n, qy/n, qz/n

def rotate_vec_with_quat(qw, qx, qy, qz, vx, vy, vz):
    """Vectorized rotation of v by unit quaternion q (active). Base->World."""
    qv = np.stack([qx, qy, qz], axis=1)
    v  = np.tile(np.array([vx, vy, vz], dtype=np.float64), (len(qw), 1))
    t  = 2.0 * np.cross(qv, v)
    v2 = v + (qw[:, None] * t) + np.cross(qv, t)
    return v2  # (N,3) world

def build_compare_plot(times_s: np.ndarray, dem_deg: np.ndarray, quat_deg: np.ndarray,
                       y_label: str, width_px: int, height_px: int):
    dpi = 100
    fig_w = max(1, width_px) / dpi
    fig_h = max(1, height_px) / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.plot(times_s, dem_deg,  lw=1.4, label=f"DEM {y_label}", color="C0")
    ax.plot(times_s, quat_deg, lw=1.2, label=f"Quat {y_label}", color="C1")
    vline = ax.axvline(0.0, color="red", lw=1.0)
    ax.set_xlim(times_s.min(), times_s.max())
    ax.set_xlabel("time [s]")
    ax.set_ylabel(f"{y_label} [deg]")
    ax.set_title(f"{y_label}: DEM vs Quaternion")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax, vline, fig.canvas

def render_plot_to_array(fig, canvas, vline, cursor_t: float, target_h: int, target_w: int) -> np.ndarray:
    vline.set_xdata([cursor_t, cursor_t])
    canvas.draw()
    buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
    w, h = canvas.get_width_height()
    img = buf.reshape((h, w, 4))[:, :, 1:4]  # ARGB -> RGB
    if h != target_h or w != target_w:
        img = cv2.resize(img, (target_w, target_h))
    return img

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
    a = transform.a   # pixel size East
    e = transform.e   # pixel size North (negative for north-up)
    p = dz_dcol / a   # ∂z/∂E
    q = dz_drow / e   # ∂z/∂N
    return p, q, transform, crs


# ------------------ main ------------------

def main():
    ap = argparse.ArgumentParser(description="Video with camera on left, DEM-vs-Quat pitch/roll plots on right.")
    ap.add_argument("--mission", required=True)
    ap.add_argument("--hz", type=int, default=None)
    ap.add_argument("--camera-pattern", default="*_hdr_front.bag")
    ap.add_argument("--camera-topic",   default="/boxi/hdr/front/image_raw/compressed")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--dem", type=str, default=None, help="Override DEM path; default: <mission>/maps/swisstopo/alti3d_chip512_gsd050.tif")
    ap.add_argument("--smooth", type=int, default=1, help="Optional rolling mean (samples) for display.")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    P = get_paths()
    mp = resolve_mission(args.mission, P)
    raw_dir, sync_dir, display_name, map_dir = mp.raw, mp.synced, mp.display, mp.maps

    synced_path, hz = pick_synced(sync_dir, args.hz)
    df = pd.read_parquet(synced_path).sort_values("t").reset_index(drop=True)
    time_col = guess_time_col(df)

    # Time array (nanoseconds -> seconds relative to first)
    t_raw = df[time_col]
    if np.issubdtype(t_raw.dtype, np.integer):
        t_ns = t_raw.to_numpy(dtype=np.int64)
    elif np.issubdtype(t_raw.dtype, np.datetime64):
        t_ns = t_raw.view("int64").to_numpy()
    else:
        t_ns = (t_raw.to_numpy(dtype=np.float64) * 1e9).astype(np.int64)
    t_rel = (t_ns - t_ns.min()) / 1e9

    # Quaternion block
    qcols = ("qw_WB", "qx_WB", "qy_WB", "qz_WB")
    if not all(c in df.columns for c in qcols):
        alt = ("qw", "qx", "qy", "qz")
        if all(c in df.columns for c in alt):
            print("[warn] qw_WB..qz_WB not found — using (qw,qx,qy,qz). Ensure it's world frame.")
            qcols = alt
        else:
            raise KeyError("Quaternion columns not found (need qw_WB..qz_WB or qw..qz).")
    qw = df[qcols[0]].to_numpy(np.float64)
    qx = df[qcols[1]].to_numpy(np.float64)
    qy = df[qcols[2]].to_numpy(np.float64)
    qz = df[qcols[3]].to_numpy(np.float64)
    qw, qx, qy, qz = normalize_quat_arrays(qw, qx, qy, qz)

    # Quaternion-derived pitch/roll
    f_w = rotate_vec_with_quat(qw, qx, qy, qz, 1.0, 0.0, 0.0)  # forward in world
    l_w = rotate_vec_with_quat(qw, qx, qy, qz, 0.0, 1.0, 0.0)  # left in world
    pitch_quat = np.degrees(np.arcsin(np.clip(f_w[:, 2], -1.0, 1.0)))  # nose up +
    roll_quat  = np.degrees(np.arcsin(np.clip(l_w[:, 2], -1.0, 1.0)))  # left up  +

    # DEM-derived slopes: read from metrics if present, else compute
    dem_long = dem_lat = None
    metrics_path = sync_dir / f"synced_{hz}Hz_metrics.parquet"
    if metrics_path.exists() and {"dem_slope_long_deg", "dem_slope_lat_deg"}.issubset(pd.read_parquet(metrics_path).columns):
        m = pd.read_parquet(metrics_path)[["t", "dem_slope_long_deg", "dem_slope_lat_deg"]]
        df = df.merge(m, on="t", how="left")
        dem_long = df["dem_slope_long_deg"].to_numpy()
        dem_lat  = df["dem_slope_lat_deg"].to_numpy()
    else:
        # compute from DEM on-the-fly
        dem_path = Path(args.dem) if args.dem else (map_dir / "swisstopo" / "alti3d_chip512_gsd050.tif")
        if not dem_path.exists():
            raise FileNotFoundError(f"DEM not found: {dem_path}")

        # Need lat/lon
        lat_col = next(c for c in df.columns if "lat" in c.lower())
        lon_col = next(c for c in df.columns if "lon" in c.lower())
        lat = df[lat_col].to_numpy()
        lon = df[lon_col].to_numpy()

        p_grid, q_grid, transform, dem_crs = compute_dem_gradients(dem_path)

        # Heading yaw from quaternion: ψ = atan2(N,E) of forward vector
        psi = np.arctan2(f_w[:, 1], f_w[:, 0])
        cosψ, sinψ = np.cos(psi), np.sin(psi)

        transformer = Transformer.from_crs("EPSG:4326", dem_crs, always_xy=True)
        x_e, y_n = transformer.transform(lon, lat)
        row_f, col_f = world_to_rowcol(transform, x_e, y_n)
        p_s = bilinear_sample(p_grid, row_f, col_f)
        q_s = bilinear_sample(q_grid, row_f, col_f)

        g_long = p_s * cosψ + q_s * sinψ
        g_lat  = -p_s * sinψ + q_s * cosψ
        dem_long = np.degrees(np.arctan(g_long))
        dem_lat  = np.degrees(np.arctan(g_lat))

    # Optional smoothing for display
    if args.smooth and args.smooth > 1:
        def smooth(a): return pd.Series(a).rolling(args.smooth, center=True, min_periods=1).mean().to_numpy()
        pitch_quat = smooth(pitch_quat); roll_quat = smooth(roll_quat)
        dem_long   = smooth(dem_long);   dem_lat   = smooth(dem_lat)

    # Camera input
    cam_bag = find_camera_bag(mp.raw, args.camera_pattern, args.camera_topic)
    out_dir = Path(get_paths()["REPO_ROOT"]) / "reports" / mp.display
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else (out_dir / f"{mp.display}_camera_vs_dem_quat.mp4")

    with AnyReader([cam_bag]) as reader:
        conns = [c for c in reader.connections if c.topic == args.camera_topic or c.topic == args.camera_topic + "/compressed"]
        if not conns:
            avail = {c.topic for c in reader.connections}
            raise KeyError(f"Topic {args.camera_topic!r} not in {cam_bag.name}. Available: {sorted(avail)}")
        msg_iter = reader.messages(connections=conns)

        # first frame
        conn0, ts0, raw0 = next(msg_iter)
        msg0 = reader.deserialize(raw0, conn0.msgtype)
        if "CompressedImage" in conn0.msgtype:
            arr0 = np.frombuffer(msg0.data, dtype=np.uint8)
            frame0 = cv2.imdecode(arr0, cv2.IMREAD_COLOR)
        else:
            frame0 = message_to_cvimage(msg0)
        if frame0 is None:
            raise RuntimeError("Could not decode first camera frame.")

        cam_h, cam_w = frame0.shape[:2]
        cam_h_out = cam_h  # keep native height
        cam_w_out = cam_w

        # Right column width (as requested)
        right_w = int(round(cam_w_out / 1.5))
        plot_h_each = cam_h_out // 2  # two stacked plots to match camera height

        # Build right plots
        fig_p, ax_p, vline_p, canvas_p = build_compare_plot(t_rel, dem_long, pitch_quat,
                                                            "Pitch", width_px=right_w, height_px=plot_h_each)
        fig_r, ax_r, vline_r, canvas_r = build_compare_plot(t_rel, dem_lat,  roll_quat,
                                                            "Roll",  width_px=right_w, height_px=plot_h_each)

        # Global frame
        total_w = cam_w_out + right_w
        total_h = cam_h_out

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(out_path), fourcc, args.fps, (total_w, total_h))

        t0 = min(t_ns.min(), ts0)

        # Writer helpers
        def compose(frame_bgr: np.ndarray, cursor_t_s: float) -> np.ndarray:
            left = cv2.resize(frame_bgr, (cam_w_out, cam_h_out))
            pitch_img = render_plot_to_array(fig_p, canvas_p, vline_p, cursor_t_s, plot_h_each, right_w)
            roll_img  = render_plot_to_array(fig_r, canvas_r, vline_r, cursor_t_s, plot_h_each, right_w)
            right_rgb = np.vstack([pitch_img, roll_img])
            if right_rgb.shape[0] != left.shape[0]:
                right_rgb = cv2.resize(right_rgb, (right_rgb.shape[1], left.shape[0]))
            right = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2BGR)
            return np.hstack([left, right])

        # first frame
        combined = compose(frame0, (ts0 - t0) / 1e9)
        vw.write(combined)
        written = 1
        if args.max_frames is not None and written >= args.max_frames:  # noqa: E225 (robustness)
            vw.release(); print(f"[done] wrote {written} frames to {out_path}"); return

        # remaining frames
        for conn, ts, raw in msg_iter:
            msg = reader.deserialize(raw, conn.msgtype)
            if "CompressedImage" in conn.msgtype:
                arr = np.frombuffer(msg.data, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
            else:
                frame = message_to_cvimage(msg)
            t_s = (ts - t0) / 1e9
            combined = compose(frame, t_s)
            vw.write(combined)
            written += 1
            if args.max_frames is not None and written >= args.max_frames:
                break

    vw.release()
    print(f"[ok] wrote {written} frames to {out_path}")
    print(f"     synced: {synced_path}")
    print(f"     camera: {cam_bag}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[interrupted]")
