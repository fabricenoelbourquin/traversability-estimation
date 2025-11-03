"""
Combine camera video with metric plot into a single video (vertical stack).
Usage:
    python src/visualization/video_metric_viewer.py --mission <mission>
    e.g. python src/visualization/video_metric_viewer.py --mission ETH-1
"""
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
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

# make src/ importable
THIS = Path(__file__).resolve()
SRC_ROOT = THIS.parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.paths import get_paths


def filter_valid_rosbags(paths: list[Path]) -> list[Path]:
    keep: list[Path] = []
    for p in paths:
        if not p.is_file():
            continue
        if p.name.startswith("._"):
            continue
        try:
            with p.open("rb") as f:
                head = f.read(13)
            if head.startswith(b"#ROSBAG V2.0"):
                keep.append(p)
        except OSError:
            continue
    return keep


def _resolve_mission(mission: str, P):
    mj = Path(P["REPO_ROOT"]) / "config" / "missions.json"
    meta = json.loads(mj.read_text())
    alias_map = meta.get("_aliases", {})
    mission_id = alias_map.get(mission, mission)
    entry = meta.get(mission_id)
    if not entry:
        raise KeyError(f"Mission '{mission}' not found in missions.json")
    folder = entry["folder"]

    # what's shown in filename
    if mission in alias_map and alias_map[mission] == mission_id:
        display = mission
    else:
        rev = [a for a, mid in alias_map.items() if mid == mission_id]
        display = rev[0] if rev else folder

    tables_dir = Path(P["TABLES"]) / folder
    synced_dir = Path(P["SYNCED"]) / folder
    raw_dir = Path(P["RAW"]) / folder
    return tables_dir, synced_dir, raw_dir, mission_id, folder, display


def pick_synced_metrics(synced_dir: Path, hz: Optional[int]) -> Path:
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


def guess_time_col(df: pd.DataFrame) -> str:
    for c in ("stamp", "timestamp", "t", "time", "time_s", "ros_time"):
        if c in df.columns:
            return c
    raise KeyError("No time column found in metrics parquet.")


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


def build_plot_figure(times_s: np.ndarray, values: np.ndarray, metric_name: str, width_px: int, height_px: int):
    dpi = 100
    fig_w = width_px / dpi
    fig_h = height_px / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.plot(times_s, values, lw=1.2)
    vline = ax.axvline(0.0, color="red", lw=1.0)
    ax.set_xlim(times_s.min(), times_s.max())
    ax.set_xlabel("time [s]")
    ax.set_ylabel(metric_name)
    ax.set_title(metric_name)
    fig.tight_layout()
    return fig, ax, vline, fig.canvas


def render_plot_to_array(fig, canvas, vline, cursor_t: float, target_h: int, target_w: int) -> np.ndarray:
    # move cursor
    vline.set_xdata([cursor_t, cursor_t])
    canvas.draw()

    # we stay with ARGB, but drop alpha
    buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
    w, h = canvas.get_width_height()
    img = buf.reshape((h, w, 4))[:, :, 1:4]  # ARGB -> RGB

    # resize to exact target (we MUST keep width == camera width)
    if h != target_h or w != target_w:
        img = cv2.resize(img, (target_w, target_h))

    return img


def main():
    ap = argparse.ArgumentParser(description="Render video + metric vertically (camera top, plot bottom).")
    ap.add_argument("--mission", required=True)
    ap.add_argument("--hz", type=int, default=None)
    ap.add_argument("--metric", default="speed_error_abs")
    ap.add_argument("--camera-pattern", default="*_hdr_front.bag")
    ap.add_argument("--camera-topic", default="/boxi/hdr/front/image_raw/compressed")
    ap.add_argument("--out", help="Output video path, e.g. /tmp/out.mp4")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    P = get_paths()
    _, synced_dir, raw_dir, _, _, display_name = _resolve_mission(args.mission, P)
    out_base = Path(P["REPO_ROOT"]) / "reports" / display_name
    out_base.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else (out_base / f"{display_name}_metric_visual_camera.mp4")

    # 1) metrics
    metrics_path = pick_synced_metrics(synced_dir, args.hz)
    df = pd.read_parquet(metrics_path)
    if args.metric not in df.columns:
        raise KeyError(f"Metric {args.metric!r} not in {metrics_path}. Available: {list(df.columns)}")
    time_col = guess_time_col(df)

    metric_raw = df[time_col]
    if np.issubdtype(metric_raw.dtype, np.integer):
        metric_ns = metric_raw.to_numpy(dtype=np.int64)
    elif np.issubdtype(metric_raw.dtype, np.datetime64):
        metric_ns = metric_raw.view("int64").to_numpy()
    else:
        metric_ns = (metric_raw.to_numpy(dtype=np.float64) * 1e9).astype(np.int64)
    metric_vals = df[args.metric].to_numpy()

    # 2) camera bag
    cam_bag = find_camera_bag(raw_dir, args.camera_pattern, args.camera_topic)

    with AnyReader([cam_bag]) as reader:
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

        # final layout: camera = 2/3, plot = 1/3
        final_h = cam_h
        cam_h_out = int(final_h * (2.0 / 3.0))
        plot_h_out = final_h - cam_h_out

        scale = cam_h_out / cam_h
        cam_w_out = int(cam_w * scale)
        plot_w_out = cam_w_out

        # ROS-time alignment
        t0 = min(metric_ns.min(), ts0)
        metric_t = (metric_ns - t0) / 1e9

        # plot figure
        fig, ax, vline, canvas = build_plot_figure(
            metric_t,
            metric_vals,
            args.metric,
            width_px=plot_w_out,
            height_px=plot_h_out,
        )

        # video writer (vertical stack)
        total_w = cam_w_out
        total_h = cam_h_out + plot_h_out
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(out_path), fourcc, args.fps, (total_w, total_h))

        # first frame
        frame0_small = cv2.resize(frame0, (cam_w_out, cam_h_out))
        plot_img = render_plot_to_array(
            fig, canvas, vline,
            (ts0 - t0) / 1e9,
            target_h=plot_h_out,
            target_w=plot_w_out,
        )
        plot_bgr = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)
        combined = np.vstack([frame0_small, plot_bgr])
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

            t_s = (ts - t0) / 1e9

            # resize cam
            frame_small = cv2.resize(frame, (cam_w_out, cam_h_out))

            # render plot for this time
            plot_img = render_plot_to_array(
                fig, canvas, vline,
                t_s,
                target_h=plot_h_out,
                target_w=plot_w_out,
            )
            plot_bgr = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)

            combined = np.vstack([frame_small, plot_bgr])
            vw.write(combined)
            written += 1

            if args.max_frames is not None and written >= args.max_frames:
                break

    vw.release()
    print(f"[ok] wrote {written} frames to {out_path}")


if __name__ == "__main__":
    main()
