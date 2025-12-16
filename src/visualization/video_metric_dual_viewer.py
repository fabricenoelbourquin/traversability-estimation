"""
Render a dual-camera video with synchronized metric plots:

- Top row: two camera streams (default: forward HDR and downward view).
- Bottom left: metric vs time with optional pitch overlay and cluster shading.
- Bottom right: same metric vs distance so cursor speed varies with actual motion.

Usage example:
  python src/visualization/video_metric_dual_viewer.py --mission ETH-1 \
         --camera-topic-1 /boxi/hdr/front/image_raw/compressed \
         --camera-topic-2 /boxi/downward/image_raw/compressed \
         --metric power_mech --overlay-pitch --shade-clusters

python src/visualization/video_metric_dual_viewer.py --mission ETH-1 \
    --overlay-pitch --shade-clusters --add-depth-cam
"""

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rosbags.highlevel import AnyReader
from rosbags.image import message_to_cvimage

from utils.paths import get_paths
from utils.cli import add_mission_arguments, add_hz_argument, resolve_mission_from_args
from utils.ros_time import message_time_ns
from utils.rosbag_tools import expand_bag_patterns, filter_valid_rosbags
from utils.filtering import filter_signal, load_metrics_config
from utils.topics import load_topic_candidates
from visualization.cluster_shading import (
    add_cluster_background,
    prepare_cluster_shading,
)

from video_metric_viewer import pick_synced_metrics

METRIC_LINE_ZORDER = 3.0
PITCH_LINE_ZORDER = 2.2


def make_display_image(frame: np.ndarray) -> np.ndarray:
    """Ensure frame is 3-channel uint8 for visualization."""
    if frame is None:
        raise RuntimeError("Frame decoding produced None")
    arr = np.asarray(frame)
    if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1):
        f = arr.astype(np.float32)
        finite = np.isfinite(f)
        if finite.any():
            vals = f[finite]
            lo, hi = np.percentile(vals, [2, 98])
            if not np.isfinite(lo):
                lo = float(np.nanmin(vals)) if vals.size else 0.0
            if not np.isfinite(hi):
                hi = lo + 1.0
            if hi - lo < 1e-6:
                hi = lo + 1.0
            f = np.clip((f - lo) / (hi - lo), 0.0, 1.0)
        else:
            f = np.zeros_like(f, dtype=np.float32)
        gray = (f * 255.0).astype(np.uint8)
        if gray.ndim == 3 and gray.shape[2] == 1:
            gray = gray[:, :, 0]
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def find_camera_bag(raw_dir: Path, pattern: str, topics: list[str] | str) -> tuple[Path, str]:
    candidates = [topics] if isinstance(topics, str) else list(topics)
    candidates = [c for c in candidates if c]
    patterns = expand_bag_patterns(pattern)
    matches: list[Path] = []
    for pat in patterns:
        matches.extend(sorted(raw_dir.glob(pat)))
    seen: set[Path] = set()
    unique_matches = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            unique_matches.append(m)

    matches = filter_valid_rosbags(unique_matches)
    tried: list[str] = []
    for p in matches:
        try:
            with AnyReader([p]) as r:
                topics = {c.topic for c in r.connections}
            for cand in candidates:
                if cand in topics:
                    return p, cand
                if cand.endswith("/image_raw") and cand + "/compressed" in topics:
                    return p, cand + "/compressed"
            tried.append(f"{p.name}: {sorted(topics)}")
        except UnicodeDecodeError:
            tried.append(f"{p.name}: <cannot open>")
    raise FileNotFoundError(
        f"Could not find bag in {raw_dir} matching any of {patterns!r} containing topics {candidates!r}.\n" +
        ("Tried:\n" + "\n".join(tried) if tried else "No candidates found.")
    )


def build_metric_panel(
    x_values: np.ndarray,
    metric_vals: np.ndarray,
    metric_label: str,
    x_label: str,
    width_px: int,
    height_px: int,
    pitch_vals: Optional[np.ndarray],
    cluster_shading,
):
    dpi = 100
    fig_w = max(1, width_px) / dpi
    fig_h = max(1, height_px) / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    add_cluster_background(ax, x_values, cluster_shading)
    if pitch_vals is not None:
        ax.set_facecolor("none")
        ax2 = ax.twinx()
        ax2.set_zorder(ax.get_zorder() - 1.0)
        ax2.patch.set_alpha(0.0)
        ax2.plot(x_values, pitch_vals, lw=1.2, color="orange", label="pitch [deg]", zorder=PITCH_LINE_ZORDER)
        ax2.set_ylabel("pitch [deg]", color="orange")
        ax2.tick_params(axis="y", colors="orange")
        ax2.spines["right"].set_color("orange")
    ax.plot(x_values, metric_vals, lw=1.2, color="tab:blue", label=metric_label, zorder=METRIC_LINE_ZORDER)
    vline = ax.axvline(0.0, color="red", lw=1.0, zorder=4)
    ax.set_xlim(x_values.min(), x_values.max())
    ax.set_xlabel(x_label)
    ax.set_ylabel(metric_label)
    ax.set_title(metric_label)
    fig.tight_layout()
    return fig, ax, vline, fig.canvas


def render_plot_to_array(fig, canvas, vline, cursor_x: float, target_h: int, target_w: int) -> np.ndarray:
    vline.set_xdata([cursor_x, cursor_x])
    canvas.draw()
    buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
    w, h = canvas.get_width_height()
    img = buf.reshape((h, w, 4))[:, :, 1:4]
    if h != target_h or w != target_w:
        img = cv2.resize(img, (target_w, target_h))
    return img


def grab_first_frame(reader: AnyReader, topic: str):
    conns = [c for c in reader.connections if c.topic == topic or c.topic == topic + "/compressed"]
    if not conns:
        raise KeyError(f"Topic {topic!r} not in bag")
    iterator = reader.messages(connections=conns)
    conn0, ts0, raw0 = next(iterator)
    msg0 = reader.deserialize(raw0, conn0.msgtype)
    if "CompressedImage" in conn0.msgtype:
        arr = np.frombuffer(msg0.data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        frame = message_to_cvimage(msg0)
    frame = make_display_image(frame)
    t_ns = message_time_ns(msg0, ts0, stream=conn0.topic)
    return frame, t_ns, iterator, conns


def next_frame(reader: AnyReader, iterator):
    try:
        conn, ts, raw = next(iterator)
    except StopIteration:
        return None, None
    msg = reader.deserialize(raw, conn.msgtype)
    if "CompressedImage" in conn.msgtype:
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return next_frame(reader, iterator)
    else:
        frame = message_to_cvimage(msg)
    frame = make_display_image(frame)
    t_ns = message_time_ns(msg, ts, stream=conn.topic)
    return frame, t_ns


def main():
    ap = argparse.ArgumentParser(description="Dual-camera metric video (time + distance plots).")
    add_mission_arguments(ap)
    add_hz_argument(ap)
    ap.add_argument("--metric", default="cost_of_transport")
    ap.add_argument("--camera-pattern-1", default="*_hdr_front*.bag")
    ap.add_argument("--camera-topic-1", default="/boxi/hdr/front/image_raw/compressed")
    ap.add_argument("--camera-pattern-2", default="*_anymal_depth_cameras*.bag")
    ap.add_argument(
        "--camera-topic-2",
        default="/anymal/depth_camera/front_lower/depth/image_rect_raw",
        help="Secondary camera topic (default: front_lower depth camera)",
    )
    ap.add_argument("--add-depth-cam", action="store_true",
                    help="Include secondary depth camera pane (default off).")
    ap.add_argument("--primary-camera", choices=["1", "2"], default="1",
                    help="Select which camera drives the video timeline (default: 1). Requires --add-depth-cam when choosing camera 2.")
    ap.add_argument("--out", help="Output video path")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--overlay-pitch", action="store_true", help="Overlay pitch on plots")
    ap.add_argument("--shade-clusters", action="store_true", help="Apply cluster shading to plots")
    ap.add_argument("--cluster-embedding", choices=["dino", "stego"], default="dino")
    ap.add_argument("--cluster-kmeans", type=int, default=None)
    ap.add_argument("--cluster-alpha", type=float, default=0.12)
    ap.add_argument("--tmin", type=float, default=None)
    ap.add_argument("--tmax", type=float, default=None)
    args = ap.parse_args()

    P = get_paths()
    metrics_cfg = load_metrics_config(Path(P["REPO_ROOT"]) / "config" / "metrics.yaml")
    filters_cfg = metrics_cfg.get("filters", {})
    mp = resolve_mission_from_args(args, P)

    raw_dir, synced_dir, display_name = mp.raw, mp.synced, mp.display
    out_base = Path(P["REPO_ROOT"]) / "reports" / display_name
    out_base.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else (out_base / f"{display_name}_dual_metric_video.mp4")

    default_camera_topics_1 = load_topic_candidates(
        "camera_front",
        [
            "/boxi/hdr/front/image_raw/compressed",
            "/gt_box/hdr_front/image_raw/compressed",
        ],
    )
    if args.camera_topic_1:
        camera_topics_1: list[str] = []
        for t in [args.camera_topic_1] + default_camera_topics_1:
            if t and t not in camera_topics_1:
                camera_topics_1.append(t)
    else:
        camera_topics_1 = default_camera_topics_1

    metrics_path = pick_synced_metrics(synced_dir, args.hz)
    df = pd.read_parquet(metrics_path)
    if args.metric not in df.columns:
        raise KeyError(f"Metric {args.metric!r} not in {metrics_path}.")
    if "dist_m" not in df.columns:
        raise KeyError("dist_m column required for distance plot. Re-run sync_streams.py to add it.")
    df = df.sort_values("t").reset_index(drop=True)

    metric_raw = df["t"]
    if np.issubdtype(metric_raw.dtype, np.integer):
        metric_ns = metric_raw.to_numpy(dtype=np.int64)
    elif np.issubdtype(metric_raw.dtype, np.datetime64):
        metric_ns = metric_raw.view("int64").to_numpy()
    else:
        metric_ns = (metric_raw.to_numpy(dtype=np.float64) * 1e9).astype(np.int64)
    metric_vals_raw = df[args.metric].to_numpy(dtype=np.float64)
    metric_vals = filter_signal(metric_vals_raw, args.metric, filters_cfg=filters_cfg, log_fn=print)
    if metric_vals is None:
        raise RuntimeError(f"Failed to obtain values for metric '{args.metric}'.")

    metric_vals = metric_vals.astype(np.float64)
    t0_ns = metric_ns.min()
    metric_t = (metric_ns - t0_ns) / 1e9

    dist = df["dist_m"].to_numpy(dtype=np.float64)
    dist = np.nan_to_num(dist - dist[0], nan=0.0)

    win_min = args.tmin if args.tmin is not None else float(metric_t.min())
    win_max = args.tmax if args.tmax is not None else float(metric_t.max())
    if win_max < win_min:
        win_min, win_max = win_max, win_min
    mask_window = (metric_t >= win_min) & (metric_t <= win_max)
    if not mask_window.any():
        mask_window = np.ones_like(metric_t, dtype=bool)
        win_min = float(metric_t.min())
        win_max = float(metric_t.max())

    metric_t_plot = metric_t[mask_window]
    metric_vals_plot = metric_vals[mask_window]
    dist_plot = dist[mask_window]
    dist_plot = np.maximum.accumulate(np.nan_to_num(dist_plot, nan=0.0))

    pitch_plot = None
    if args.overlay_pitch:
        try:
            from video_metric_viewer import get_quaternion_block, normalize_quat_arrays, euler_zyx_from_qWB

            qw, qx, qy, qz = get_quaternion_block(df)
            qw, qx, qy, qz = normalize_quat_arrays(qw, qx, qy, qz)
            _, pitch_deg, _ = euler_zyx_from_qWB(qw, qx, qy, qz)
            pitch_plot = pitch_deg[mask_window]
        except Exception as exc:
            print(f"[warn] Unable to compute pitch overlay: {exc}")
            pitch_plot = None

    cluster_shading = None
    if args.shade_clusters:
        try:
            mission_maps = Path(mp.maps)
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

    fig_time, ax_time, vline_time, canvas_time = build_metric_panel(
        metric_t_plot,
        metric_vals_plot,
        args.metric,
        "time [s]",
        width_px=800,
        height_px=300,
        pitch_vals=pitch_plot,
        cluster_shading=cluster_shading,
    )

    fig_dist, ax_dist, vline_dist, canvas_dist = build_metric_panel(
        dist_plot,
        metric_vals_plot,
        args.metric,
        "distance [m]",
        width_px=800,
        height_px=300,
        pitch_vals=pitch_plot,
        cluster_shading=cluster_shading,
    )

    def time_to_dist(t_s: float) -> float:
        if metric_t_plot.size == 0:
            return 0.0
        return float(np.interp(t_s, metric_t_plot, dist_plot, left=dist_plot[0], right=dist_plot[-1]))

    bag1, topic1 = find_camera_bag(raw_dir, args.camera_pattern_1, camera_topics_1)
    bag2 = None
    topic2 = args.camera_topic_2
    if args.add_depth_cam:
        bag2, topic2 = find_camera_bag(raw_dir, args.camera_pattern_2, args.camera_topic_2)

    if not args.add_depth_cam and args.primary_camera == "2":
        raise SystemExit("--primary-camera=2 requires --add-depth-cam")

    primary_idx = 1 if args.primary_camera == "1" else 2
    if not args.add_depth_cam:
        primary_idx = 1

    bag_primary = bag1 if primary_idx == 1 else bag2
    bag_secondary = bag2 if primary_idx == 1 else bag1
    topic_primary = topic1 if primary_idx == 1 else topic2
    topic_secondary = topic2 if primary_idx == 1 else topic1

    if args.add_depth_cam:
        with AnyReader([bag_primary]) as reader_p, AnyReader([bag_secondary]) as reader_s:
            frame_p0, ts_p0, iter_p, _ = grab_first_frame(reader_p, topic_primary)
            frame_s0, ts_s0, iter_s, _ = grab_first_frame(reader_s, topic_secondary)

            cam_h_out = 480
            scale_p = cam_h_out / frame_p0.shape[0]
            scale_s = cam_h_out / frame_s0.shape[0]
            cam_w_p = int(frame_p0.shape[1] * scale_p)
            cam_w_s = int(frame_s0.shape[1] * scale_s)
            plot_h_out = int(round(max(240, cam_h_out // 2) * 1.2))

            total_w = cam_w_p + cam_w_s
            total_h = cam_h_out + plot_h_out

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(str(out_path), fourcc, args.fps, (total_w, total_h))

            def resize_frame(frame, target_w):
                return cv2.resize(frame, (target_w, cam_h_out))

            sec_current_frame = cv2.resize(frame_s0, (cam_w_s, cam_h_out))
            sec_current_frame = cv2.flip(sec_current_frame, 0)
            sec_current_ts = ts_s0
            sec_future = None  # holds the first frame after target

            def advance_secondary(target_ns: int):
                nonlocal sec_current_frame, sec_current_ts, sec_future
                # If we already peeked a future frame and it's now behind target, promote it
                if sec_future is not None and sec_future[1] <= target_ns:
                    sec_current_frame, sec_current_ts = sec_future
                    sec_future = None
                while True:
                    nxt = next_frame(reader_s, iter_s)
                    if nxt[0] is None:
                        break
                    frame, ts = nxt
                    if frame is None:
                        continue
                    frame_resized = cv2.resize(frame, (cam_w_s, cam_h_out))
                    frame_resized = cv2.flip(frame_resized, 0)
                    if ts <= target_ns:
                        sec_current_frame = frame_resized
                        sec_current_ts = ts
                    else:
                        sec_future = (frame_resized, ts)
                        break

            # Precompute camera1 frame array for first frame
            frame_p_resized = resize_frame(frame_p0, cam_w_p)

            def compose(cursor_t: float, frame_primary: np.ndarray) -> np.ndarray:
                cursor_dist = time_to_dist(cursor_t)
                plot_time = render_plot_to_array(fig_time, canvas_time, vline_time, cursor_t, plot_h_out, cam_w_p)
                plot_dist = render_plot_to_array(fig_dist, canvas_dist, vline_dist, cursor_dist, plot_h_out, cam_w_s)
                plot_time_bgr = cv2.cvtColor(plot_time, cv2.COLOR_RGB2BGR)
                plot_dist_bgr = cv2.cvtColor(plot_dist, cv2.COLOR_RGB2BGR)
                top = np.hstack([frame_primary, sec_current_frame])
                bottom = np.hstack([plot_time_bgr, plot_dist_bgr])
                return np.vstack([top, bottom])

            written = 0

            def process_frame(frame, ts_ns):
                nonlocal written
                cursor_t = (ts_ns - t0_ns) / 1e9
                if cursor_t < win_min or cursor_t > win_max:
                    return
                advance_secondary(ts_ns)
                composed = compose(cursor_t, resize_frame(frame, cam_w_p))
                vw.write(composed)
                written += 1

            process_frame(frame_p0, ts_p0)
            if args.max_frames is not None and written >= args.max_frames:
                vw.release()
                print(f"[done] wrote {written} frames to {out_path}")
                return

            for conn, ts, raw in iter_p:
                msg = reader_p.deserialize(raw, conn.msgtype)
                if "CompressedImage" in conn.msgtype:
                    arr = np.frombuffer(msg.data, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                else:
                    frame = message_to_cvimage(msg)
                cam_ts_ns = message_time_ns(msg, ts, stream=conn.topic)
                process_frame(frame, cam_ts_ns)
                if args.max_frames is not None and written >= args.max_frames:
                    break

        vw.release()
        plt.close(fig_time)
        plt.close(fig_dist)
        print(f"[ok] wrote {written} frames to {out_path}")
        print(f"     input synced: {metrics_path}")
        print(f"     camera1 bag: {bag1} (topic {topic1})")
        print(f"     camera2 bag: {bag2} (topic {topic2})")
        return

    # Single camera layout
    with AnyReader([bag_primary]) as reader_p:
        frame_p0, ts_p0, iter_p, _ = grab_first_frame(reader_p, topic_primary)

        cam_h_out = 480
        scale_p = cam_h_out / frame_p0.shape[0]
        cam_w_p = int(frame_p0.shape[1] * scale_p)
        plot_col_w = int(round(cam_w_p * 0.9))
        time_plot_h = int(round(cam_h_out * 0.55))
        dist_plot_h = cam_h_out - time_plot_h

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(out_path), fourcc, args.fps, (cam_w_p + plot_col_w, cam_h_out))

        def resize_frame(frame, target_w):
            return cv2.resize(frame, (target_w, cam_h_out))

        def compose(cursor_t: float, frame_primary: np.ndarray) -> np.ndarray:
            cursor_dist = time_to_dist(cursor_t)
            plot_time = render_plot_to_array(fig_time, canvas_time, vline_time, cursor_t, time_plot_h, plot_col_w)
            plot_dist = render_plot_to_array(fig_dist, canvas_dist, vline_dist, cursor_dist, dist_plot_h, plot_col_w)
            plot_time_bgr = cv2.cvtColor(plot_time, cv2.COLOR_RGB2BGR)
            plot_dist_bgr = cv2.cvtColor(plot_dist, cv2.COLOR_RGB2BGR)
            right_col = np.vstack([plot_time_bgr, plot_dist_bgr])
            return np.hstack([frame_primary, right_col])

        written = 0

        def process_frame(frame, ts_ns):
            nonlocal written
            cursor_t = (ts_ns - t0_ns) / 1e9
            if cursor_t < win_min or cursor_t > win_max:
                return
            composed = compose(cursor_t, resize_frame(frame, cam_w_p))
            vw.write(composed)
            written += 1

        process_frame(frame_p0, ts_p0)
        if args.max_frames is not None and written >= args.max_frames:
            vw.release()
            print(f"[done] wrote {written} frames to {out_path}")
            return

        for conn, ts, raw in iter_p:
            msg = reader_p.deserialize(raw, conn.msgtype)
            if "CompressedImage" in conn.msgtype:
                arr = np.frombuffer(msg.data, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
            else:
                frame = message_to_cvimage(msg)
            cam_ts_ns = message_time_ns(msg, ts, stream=conn.topic)
            process_frame(frame, cam_ts_ns)
            if args.max_frames is not None and written >= args.max_frames:
                break

    vw.release()
    plt.close(fig_time)
    plt.close(fig_dist)
    print(f"[ok] wrote {written} frames to {out_path}")
    print(f"     input synced: {metrics_path}")
    print(f"     camera1 bag: {bag1} (topic {topic1})")
    if args.add_depth_cam:
        print(f"     camera2 bag: {bag2} (topic {topic2})")


if __name__ == "__main__":
    main()
