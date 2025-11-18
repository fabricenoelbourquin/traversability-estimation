"""
Combine camera video with metric plot into a single video (vertical stack),
and optionally add yaw/pitch/roll time-series to the RIGHT (three stacked plots).

Usage:
    python src/visualization/video_metric_viewer.py --mission TRIM-1
    python src/visualization/video_metric_viewer.py --mission TRIM-1 --overlay-pitch
    python src/visualization/video_metric_viewer.py --mission TRIM-1 --no-plot-orientation
"""
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rosbags.highlevel import AnyReader
from rosbags.image import message_to_cvimage

# make src/ importable
THIS = Path(__file__).resolve()
SRC_ROOT = THIS.parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.paths import get_paths
from utils.missions import resolve_mission
from utils.ros_time import message_time_ns
from utils.rosbag_tools import filter_valid_rosbags
from utils.filtering import filter_signal, load_metrics_config
from visualization.cluster_shading import (
    ClusterShading,
    apply_cluster_shading,
    prepare_cluster_shading,
)

def pick_synced_metrics(synced_dir: Path, hz: int | None) -> Path:
    if hz is not None:
        p = synced_dir / f"synced_{hz}Hz_metrics.parquet"
        if p.exists():
            return p
        p = synced_dir / f"synced_{hz}Hz.parquet"
        if p.exists():
            return p
        raise FileNotFoundError(f"Neither synced_{hz}Hz_metrics.parquet nor synced_{hz}Hz.parquet in {synced_dir}")

    cands = sorted(synced_dir.glob("synced_*Hz_metrics.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    if cands:
        return cands[0]
    cands = sorted(synced_dir.glob("synced_*Hz.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"No synced parquet in {synced_dir}")
    return cands[0]

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
            # auto-fallback to /compressed if user asked for /image_raw
            if topic.endswith("/image_raw") and topic + "/compressed" in topics:
                return p
            tried.append(f"{p.name}: {sorted(topics)}")
        except UnicodeDecodeError:
            tried.append(f"{p.name}: <cannot open>")
    raise FileNotFoundError(
        f"Could not find bag in {raw_dir} matching {pattern!r} that has topic {topic!r}.\n"
        "Tried:\n" + "\n".join(tried)
    )

def build_plot_figure(times_s: np.ndarray, values: np.ndarray, y_label: str, width_px: int, height_px: int):
    dpi = 100
    fig_w = max(1, width_px) / dpi
    fig_h = max(1, height_px) / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.plot(times_s, values, lw=1.2)
    vline = ax.axvline(0.0, color="red", lw=1.0)
    ax.set_xlim(times_s.min(), times_s.max())
    ax.set_xlabel("time [s]")
    ax.set_ylabel(y_label)
    ax.set_title(y_label)
    fig.tight_layout()
    return fig, ax, vline, fig.canvas

def build_metric_plot_with_optional_pitch(
    times_s: np.ndarray,
    metric_vals: np.ndarray,
    metric_label: str,
    width_px: int,
    height_px: int,
    pitch_vals: np.ndarray | None = None,
    pitch_label: str = "pitch [deg]",
    cluster_shading: ClusterShading | None = None,
):
    """Build the bottom metric plot. If pitch_vals is provided, overlay it on a twin Y-axis."""
    dpi = 100
    fig_w = max(1, width_px) / dpi
    fig_h = max(1, height_px) / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    if cluster_shading is not None:
        apply_cluster_shading(ax, times_s, cluster_shading)

    # primary: metric
    ax.plot(times_s, metric_vals, lw=1.2, label=metric_label, zorder=2)
    vline = ax.axvline(0.0, color="red", lw=1.0, zorder=4)  # time cursor
    ax.set_xlim(times_s.min(), times_s.max())
    ax.set_xlabel("time [s]")
    ax.set_ylabel(metric_label)

    if pitch_vals is not None:
        ax2 = ax.twinx()
        ax2.set_zorder(ax.get_zorder() + 1)
        ax2.patch.set_alpha(0.0)
        ax2.plot(times_s, pitch_vals, lw=1.4, color="orange", label=pitch_label, zorder=3)
        ax2.set_ylabel(pitch_label, color="orange")
        ax2.tick_params(axis="y", colors="orange")
        ax2.spines["right"].set_color("orange")
        # Optional: one combined title
        ax.set_title(f"{metric_label}  +  {pitch_label}")
    else:
        ax.set_title(metric_label)

    fig.tight_layout()
    return fig, ax, vline, fig.canvas

def render_plot_to_array(fig, canvas, vline, cursor_t: float, target_h: int, target_w: int) -> np.ndarray:
    # move cursor
    vline.set_xdata([cursor_t, cursor_t])
    canvas.draw()

    # ARGB -> RGB
    buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
    w, h = canvas.get_width_height()
    img = buf.reshape((h, w, 4))[:, :, 1:4]

    # resize to exact target
    if h != target_h or w != target_w:
        img = cv2.resize(img, (target_w, target_h))

    return img

def get_quaternion_block(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prefer qw_WB..qz_WB; else fall back to qw..qz if present.
    Returns float64 arrays (normalized later by caller).
    """
    if all(c in df.columns for c in ("qw_WB", "qx_WB", "qy_WB", "qz_WB")):
        qw = df["qw_WB"].to_numpy(dtype=np.float64)
        qx = df["qx_WB"].to_numpy(dtype=np.float64)
        qy = df["qy_WB"].to_numpy(dtype=np.float64)
        qz = df["qz_WB"].to_numpy(dtype=np.float64)
        return qw, qx, qy, qz
    elif all(c in df.columns for c in ("qw", "qx", "qy", "qz")):
        print("[warn] Using (qw,qx,qy,qz) instead of qw_WB..qz_WB — make sure this is world, not odom.")
        qw = df["qw"].to_numpy(dtype=np.float64)
        qx = df["qx"].to_numpy(dtype=np.float64)
        qy = df["qy"].to_numpy(dtype=np.float64)
        qz = df["qz"].to_numpy(dtype=np.float64)
        return qw, qx, qy, qz
    else:
        raise KeyError("Quaternion columns not found (need qw_WB..qz_WB or qw..qz).")


def normalize_quat_arrays(qw, qx, qy, qz):
    n = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    n[n == 0.0] = 1.0
    return qw/n, qx/n, qy/n, qz/n


def rotate_vec_with_quat(qw, qx, qy, qz, vx, vy, vz):
    """Vectorized rotation of v by unit quaternion q (active). Base->World."""
    qv = np.stack([qx, qy, qz], axis=1)           # (N,3)
    v  = np.tile(np.array([vx, vy, vz], dtype=np.float64), (len(qw), 1))
    t  = 2.0 * np.cross(qv, v)
    v2 = v + (qw[:, None] * t) + np.cross(qv, t)
    return v2  # (N,3) world


def euler_zyx_from_qWB(qw: np.ndarray, qx: np.ndarray, qy: np.ndarray, qz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert quaternion q_WB (body→world, active) to yaw-pitch-roll (ZYX) in degrees.
    World: ENU. Body: x-forward, y-left, z-up.
    Returns navigation-friendly yaw/pitch (north = 0°, clockwise yaw positive, nose-up pitch positive).
    """
    # ensure unit quaternions
    qw, qx, qy, qz = normalize_quat_arrays(qw, qx, qy, qz)

    # rotation matrix R_WB (maps body vectors into world)
    # vectorized components
    xx = qx * qx; yy = qy * qy; zz = qz * qz
    xy = qx * qy; xz = qx * qz; yz = qy * qz
    wx = qw * qx; wy = qw * qy; wz = qw * qz

    r00 = 1.0 - 2.0 * (yy + zz)
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)

    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (xx + zz)
    r12 = 2.0 * (yz - wx)

    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (xx + yy)

    # ZYX extraction (yaw about +Z, pitch about +Y, roll about +X)
    yaw   = np.arctan2(r10, r00)
    pitch = np.arctan2(-r20, np.clip(np.sqrt(r00*r00 + r10*r10), 1e-12, None))
    roll  = np.arctan2(r21, r22)

    yaw_deg = np.rad2deg(yaw)
    pitch_deg = np.rad2deg(pitch)
    roll_deg = np.rad2deg(roll)

    # Align with intuitive navigation convention (clockwise heading, nose-up positive).
    yaw_deg *= -1.0
    pitch_deg *= -1.0

    return yaw_deg, pitch_deg, roll_deg


def main():
    ap = argparse.ArgumentParser(description="Render video + metric vertically (camera top, plot bottom) with optional yaw/pitch/roll plots on the right.")
    ap.add_argument("--mission", required=True)
    ap.add_argument("--hz", type=int, default=None)
    ap.add_argument("--metric", default="power_mech")
    ap.add_argument("--camera-pattern", default="*_hdr_front.bag")
    ap.add_argument("--camera-topic", default="/boxi/hdr/front/image_raw/compressed")
    ap.add_argument("--out", help="Output video path, e.g. /tmp/out.mp4")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--max-frames", type=int, default=None)
    # orientation plots
    ap.add_argument("--plot-orientation", dest="plot_orientation", action="store_true", default=True,
                    help="Add yaw/pitch/roll plots to the right (default: on).")
    ap.add_argument("--no-plot-orientation", dest="plot_orientation", action="store_false",
                    help="Disable yaw/pitch/roll plots.")
    ap.add_argument(
        "--overlay-pitch",
        action="store_true",
        help="Overlay pitch [deg] on the metric plot with a secondary Y-axis.",
    )
    ap.add_argument(
        "--shade-clusters",
        action="store_true",
        help="Color the metric plot background by cluster assignments (default: off).",
    )
    ap.add_argument(
        "--cluster-embedding",
        choices=["dino", "stego"],
        default="dino",
        help="Embedding head to use for cluster shading (default: dino).",
    )
    ap.add_argument(
        "--cluster-kmeans",
        type=int,
        default=None,
        help="Explicit KMeans cluster count (default: autodetect latest).",
    )
    ap.add_argument(
        "--cluster-alpha",
        type=float,
        default=0.12,
        help="Alpha for cluster background shading.",
    )
    ap.add_argument("--tmin", type=float, default=None, help="Start time [s] relative to the video time zero (min).")
    ap.add_argument("--tmax", type=float, default=None, help="End time [s] relative to the video time zero (max).")
    args = ap.parse_args()

    P = get_paths()
    metrics_cfg = load_metrics_config(Path(P["REPO_ROOT"]) / "config" / "metrics.yaml")
    filters_cfg = metrics_cfg.get("filters", {})

    mp = resolve_mission(args.mission, P)
    raw_dir, synced_dir, display_name = mp.raw, mp.synced, mp.display
    out_base = Path(P["REPO_ROOT"]) / "reports" / display_name
    out_base.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else (out_base / f"{display_name}_metric_visual_camera.mp4")

    # 1) metrics (+ quaternions if available)
    metrics_path = pick_synced_metrics(synced_dir, args.hz)
    df = pd.read_parquet(metrics_path)
    if args.metric not in df.columns:
        raise KeyError(f"Metric {args.metric!r} not in {metrics_path}. Available: {list(df.columns)}")
    time_col = "t"
    df = df.sort_values(time_col).reset_index(drop=True) # ensure sorted by time

    metric_raw = df[time_col]
    # convert time column to ns
    if np.issubdtype(metric_raw.dtype, np.integer):
        metric_ns = metric_raw.to_numpy(dtype=np.int64)
    elif np.issubdtype(metric_raw.dtype, np.datetime64):
        metric_ns = metric_raw.view("int64").to_numpy()
    else:
        metric_ns = (metric_raw.to_numpy(dtype=np.float64) * 1e9).astype(np.int64)
    metric_vals_raw = df[args.metric].to_numpy(dtype=np.float64)
    metric_vals = filter_signal(
        metric_vals_raw, args.metric, filters_cfg=filters_cfg, log_fn=print
    )
    if metric_vals is None:
        raise RuntimeError(f"Failed to obtain values for metric '{args.metric}'.")

    # Try grabbing quaternion columns if orientation needed
    need_orientation = args.plot_orientation or args.overlay_pitch
    have_orientation = False
    yaw_deg = pitch_deg = roll_deg = None 
    if need_orientation:
        try:
            qw, qx, qy, qz = get_quaternion_block(df)
            qw, qx, qy, qz = normalize_quat_arrays(qw, qx, qy, qz)
            yaw_deg, pitch_deg, roll_deg = euler_zyx_from_qWB(qw, qx, qy, qz)
            have_orientation = True
            """
            yaw_deg = filter_signal(yaw_deg, "yaw_deg", filters_cfg=filters_cfg, log_fn=print)
            pitch_deg = filter_signal(pitch_deg, "pitch_deg", filters_cfg=filters_cfg, log_fn=print)
            roll_deg = filter_signal(roll_deg, "roll_deg", filters_cfg=filters_cfg, log_fn=print)
            """
        except Exception as e:
            print(f"[warn] Orientation unavailable: {e}")
            have_orientation = False
            if args.overlay_pitch:
                print("[warn] --overlay-pitch requested, but pitch cannot be computed. Proceeding without overlay.")

    # 2) camera bag
    cam_bag = find_camera_bag(raw_dir, args.camera_pattern, args.camera_topic)
    with AnyReader([cam_bag]) as reader:
        # choose camera topic connection(s)
        conns = [c for c in reader.connections if c.topic == args.camera_topic or c.topic == args.camera_topic + "/compressed"]
        if not conns:
            avail = {c.topic for c in reader.connections}
            raise KeyError(f"Topic {args.camera_topic!r} not in {cam_bag.name}. Available: {sorted(avail)}")

        msg_iter = reader.messages(connections=conns)

        # first message
        try:
            conn0, ts0, raw0 = next(msg_iter)
        except StopIteration:
            raise RuntimeError(f"No messages on topic {args.camera_topic!r} in {cam_bag}")

        msg0 = reader.deserialize(raw0, conn0.msgtype)
        if "CompressedImage" in conn0.msgtype:
            arr0 = np.frombuffer(msg0.data, dtype=np.uint8)
            frame0 = cv2.imdecode(arr0, cv2.IMREAD_COLOR)
        else:
            frame0 = message_to_cvimage(msg0)

        if frame0 is None:
            raise RuntimeError("Could not decode first camera frame.")

        cam_h, cam_w = frame0.shape[:2]

        # Base layout heights: camera = 2/3, metric plot = 1/3 (ratio 2:1)
        plot_h_out = max(1, cam_h // 3)
        cam_h_out  = 2 * plot_h_out
        final_h_left = cam_h_out + plot_h_out

        scale     = cam_h_out / cam_h
        cam_w_out = int(round(cam_w * scale))
        plot_w_out = cam_w_out  # keep plot as wide as camera stack

        # Optional RIGHT column (yaw/pitch/roll), each plot height == plot_h_out
        right_w = int(round(cam_w_out / 1.5)) if have_orientation else 0  # narrower than camera, ~half width

        # Global frame size (left stack plus optional right stack)
        total_w = cam_w_out + right_w
        total_h = final_h_left  # right column stacks to the same total height (3 * plot_h_out == final_h_left)

        # ROS-time alignment (prefer header.stamp; fallback to log time with a one-time warning)
        cam_ns0 = message_time_ns(msg0, ts0, stream="camera")
        t0 = min(metric_ns.min(), cam_ns0)
        metric_t = (metric_ns - t0) / 1e9

        # optional focus window in seconds relative to t0
        win_min = args.tmin if args.tmin is not None else float(metric_t.min())
        win_max = args.tmax if args.tmax is not None else float(metric_t.max())
        if win_max < win_min:
            win_min, win_max = win_max, win_min

        mask_window = (metric_t >= win_min) & (metric_t <= win_max)
        if not np.any(mask_window):
            print("[warn] --tmin/--tmax window empty; using full time range.")
            mask_window = np.ones_like(metric_t, dtype=bool)
            win_min = float(metric_t.min())
            win_max = float(metric_t.max())

        cluster_shading = None
        if args.shade_clusters:
            mission_maps = Path(mp.maps)
            try:
                cluster_shading = prepare_cluster_shading(
                    df,
                    mission_maps,
                    args.cluster_embedding,
                    args.cluster_kmeans,
                    args.cluster_alpha,
                    mask=mask_window,
                )
            except Exception as exc:
                print(f"[warn] Failed to prepare cluster shading: {exc}")
                cluster_shading = None

        metric_t_plot = metric_t[mask_window]
        metric_vals_plot = metric_vals[mask_window]
        if args.overlay_pitch and have_orientation and pitch_deg is not None:
            pitch_plot = pitch_deg[mask_window]
        else:
            pitch_plot = None
        if have_orientation:
            yaw_plot = yaw_deg[mask_window]
            pitch_full_plot = pitch_deg[mask_window]
            roll_plot = roll_deg[mask_window]

        # Build metric plot (bottom-left) — possibly with pitch overlay
        if args.overlay_pitch and have_orientation and pitch_deg is not None:
            fig_m, ax_m, vline_m, canvas_m = build_metric_plot_with_optional_pitch(
                metric_t_plot,
                metric_vals_plot,
                args.metric,
                width_px=plot_w_out,
                height_px=plot_h_out,
                pitch_vals=pitch_plot,
                pitch_label="pitch [deg]",
                cluster_shading=cluster_shading,
            )
        else:
            fig_m, ax_m, vline_m, canvas_m = build_metric_plot_with_optional_pitch(
                metric_t_plot,
                metric_vals_plot,
                args.metric,
                width_px=plot_w_out,
                height_px=plot_h_out,
                pitch_vals=None,
                cluster_shading=cluster_shading,
            )

        # Build orientation plots if requested
        if have_orientation:
            fig_y, ax_y, vline_y, canvas_y = build_plot_figure(
                metric_t_plot, yaw_plot, "yaw [deg]", width_px=right_w, height_px=plot_h_out
            )
            fig_p, ax_p, vline_p, canvas_p = build_plot_figure(
                metric_t_plot, pitch_full_plot, "pitch [deg]", width_px=right_w, height_px=plot_h_out
            )
            fig_r, ax_r, vline_r, canvas_r = build_plot_figure(
                metric_t_plot, roll_plot, "roll [deg]", width_px=right_w, height_px=plot_h_out
            )

        # video writer (stack left; optional right)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(out_path), fourcc, args.fps, (total_w, total_h))

        # --- Helper to compose one frame
        def compose_frame(frame_bgr: np.ndarray, cursor_t_s: float) -> np.ndarray:
            # Left column
            frame_small = cv2.resize(frame_bgr, (cam_w_out, cam_h_out))
            plot_img = render_plot_to_array(
                fig_m, canvas_m, vline_m, cursor_t_s, target_h=plot_h_out, target_w=plot_w_out
            )
            plot_bgr = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)
            left = np.vstack([frame_small, plot_bgr])  # (final_h_left, cam_w_out, 3)

            if right_w > 0:
                # Right column: yaw / pitch / roll stacked
                yaw_img = render_plot_to_array(fig_y, canvas_y, vline_y, cursor_t_s, plot_h_out, right_w)
                pit_img = render_plot_to_array(fig_p, canvas_p, vline_p, cursor_t_s, plot_h_out, right_w)
                rol_img = render_plot_to_array(fig_r, canvas_r, vline_r, cursor_t_s, plot_h_out, right_w)
                yaw_bgr = cv2.cvtColor(yaw_img, cv2.COLOR_RGB2BGR)
                pit_bgr = cv2.cvtColor(pit_img, cv2.COLOR_RGB2BGR)
                rol_bgr = cv2.cvtColor(rol_img, cv2.COLOR_RGB2BGR)
                right = np.vstack([yaw_bgr, pit_bgr, rol_bgr])  # (final_h_left, right_w, 3)
                if right.shape[0] != left.shape[0]:
                    right = cv2.resize(right, (right.shape[1], left.shape[0]))
                return np.hstack([left, right])
            else:
                return left

        # first frame
        t_s0 = (cam_ns0 - t0) / 1e9
        written = 0
        if win_min <= t_s0 <= win_max:
            combined = compose_frame(frame0, t_s0)
            vw.write(combined)
            written = 1
        if args.max_frames is not None and written >= args.max_frames:
            vw.release()
            print(f"[done] wrote {written} frames to {out_path}")
            return

        # rest of frames
        for conn, ts, raw in msg_iter:
            msg = reader.deserialize(raw, conn.msgtype)
            if "CompressedImage" in conn.msgtype:
                arr = np.frombuffer(msg.data, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
            else:
                frame = message_to_cvimage(msg)

            cam_ns = message_time_ns(msg, ts, stream="camera")
            t_s = (cam_ns - t0) / 1e9
            if t_s < win_min:
                continue
            if t_s > win_max:
                break
            combined = compose_frame(frame, t_s)
            vw.write(combined)
            written += 1
            if args.max_frames is not None and written >= args.max_frames:
                break

    vw.release()
    print(f"[ok] wrote {written} frames to {out_path}")
    print(f"     input synced: {metrics_path}")
    print(f"     camera bag:   {cam_bag}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[interrupted]")
