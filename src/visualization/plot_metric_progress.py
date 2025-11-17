#!/usr/bin/env python3
"""
Plot mission metrics against time and/or cumulative distance.

Default behavior:
  - Creates one subplot per metric listed under `metrics.names` in config/metrics.yaml (when available).
  - Generates a time-based plot unless --plot-distance only is requested.

Examples:
  python src/visualization/plot_metric_progress.py --mission ETH-1
  python src/visualization/plot_metric_progress.py --mission ETH-1 --plot-distance --with-speeds
  python src/visualization/plot_metric_progress.py --mission ETH-1 --plot-time --plot-distance --overlay-pitch
"""

from __future__ import annotations
import argparse
import colorsys
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import rasterio
    from pyproj import Transformer
    from rasterio.transform import rowcol
except ImportError:
    rasterio = None
    Transformer = None
    rowcol = None

# --- path helper ---
import sys
THIS_FILE = Path(__file__).resolve()
SRC_ROOT = THIS_FILE.parents[1]    # .../src
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from utils.paths import get_paths
from utils.missions import resolve_mission
from utils.filtering import filter_signal, load_metrics_config

PITCH_LABEL = "pitch [deg]"
PITCH_COLOR = "tab:orange"
ZERO_LINE_STYLE = {"color": "0.3", "linewidth": 0.7, "alpha": 0.55, "linestyle": "--"}


@dataclass
class ClusterShading:
    segments: list[tuple[int, int, int]]
    colors: dict[int, tuple[float, float, float]]
    alpha: float
    description: str


def _cluster_palette(n_clusters: int) -> list[tuple[float, float, float]]:
    n = max(1, int(n_clusters))
    palette = []
    for i in range(n):
        h = (i / float(n)) % 1.0
        s, v = 0.65, 0.95
        palette.append(colorsys.hsv_to_rgb(h, s, v))
    return palette


def _compute_step_edges(x_values: np.ndarray) -> np.ndarray:
    x = np.asarray(x_values, dtype=np.float64)
    if x.size == 0:
        return np.asarray([], dtype=np.float64)
    if x.size == 1:
        step = 0.5
        return np.array([x[0] - step, x[0] + step], dtype=np.float64)
    diffs = np.diff(x)
    mids = x[:-1] + 0.5 * diffs
    edges = np.empty(x.size + 1, dtype=np.float64)
    edges[1:-1] = mids
    edges[0] = x[0] - 0.5 * diffs[0]
    edges[-1] = x[-1] + 0.5 * diffs[-1]
    return edges


def _segment_cluster_ids(cluster_ids: np.ndarray) -> list[tuple[int, int, int]]:
    segments: list[tuple[int, int, int]] = []
    start = None
    current = None
    for idx, val in enumerate(cluster_ids):
        if not np.isfinite(val):
            if current is not None and start is not None:
                segments.append((start, idx, current))
                start = None
                current = None
            continue
        cid = int(val)
        if current is None:
            start = idx
            current = cid
            continue
        if cid != current:
            segments.append((start, idx, current))
            start = idx
            current = cid
    if current is not None and start is not None:
        segments.append((start, cluster_ids.shape[0], current))
    return segments


def _apply_cluster_shading(ax, x_values: np.ndarray, shading: ClusterShading | None) -> None:
    if shading is None or not shading.segments:
        return
    edges = _compute_step_edges(x_values)
    if edges.size == 0:
        return
    for start, end, cid in shading.segments:
        if end <= start:
            continue
        if end >= edges.size:
            end = edges.size - 1
        color = shading.colors.get(cid)
        if color is None:
            continue
        x0 = edges[start]
        x1 = edges[end]
        if not np.isfinite(x0) or not np.isfinite(x1):
            continue
        ax.axvspan(x0, x1, facecolor=color, alpha=shading.alpha, zorder=0.02, edgecolor="none", linewidth=0)


def _locate_cluster_raster(mission_maps: Path, kmeans: int | None, emb: str) -> tuple[Path, int | None]:
    clusters_dir = mission_maps / "clusters"
    if not clusters_dir.exists():
        raise FileNotFoundError(f"Cluster directory not found: {clusters_dir}")
    if kmeans is not None:
        cand = clusters_dir / f"cluster_kmeans{kmeans}_{emb}.tif"
        if not cand.exists():
            raise FileNotFoundError(f"{cand} not found (needed for cluster shading).")
        return cand, kmeans
    cands = sorted(clusters_dir.glob(f"cluster_kmeans*_{emb}.tif"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"No cluster_kmeans*_{emb}.tif files in {clusters_dir}")
    picked = cands[0]
    match = re.search(r"cluster_kmeans(\d+)", picked.name)
    detected = int(match.group(1)) if match else None
    return picked, detected


def _sample_cluster_labels(lat: np.ndarray, lon: np.ndarray, raster_path: Path) -> tuple[np.ndarray, int]:
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    labels = np.full(lat.shape, np.nan, dtype=np.float64)
    with rasterio.open(raster_path) as ds:
        band = ds.read(1)
        max_label = int(np.nanmax(band)) if band.size else -1
        ok = np.isfinite(lat) & np.isfinite(lon)
        if not ok.any():
            return labels, max_label
        tf = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
        x, y = tf.transform(lon[ok], lat[ok])
        rows, cols = rowcol(ds.transform, x, y, op=np.round)
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        inside = (rows >= 0) & (rows < ds.height) & (cols >= 0) & (cols < ds.width)
        if inside.any():
            idx_ok = np.nonzero(ok)[0]
            valid_idx = idx_ok[inside]
            labels_vals = band[rows[inside], cols[inside]].astype(np.float64)
            labels[valid_idx] = labels_vals
    return labels, max_label


def _prepare_cluster_shading(
    df: pd.DataFrame,
    mission_maps: Path,
    emb: str,
    kmeans: int | None,
    alpha: float,
) -> ClusterShading | None:
    if rasterio is None or Transformer is None or rowcol is None:
        print("[warn] Cluster shading requires rasterio + pyproj; install them to enable this feature.")
        return None
    if "lat" not in df.columns or "lon" not in df.columns:
        print("[warn] Cannot shade clusters — 'lat'/'lon' columns missing.")
        return None
    try:
        raster_path, inferred_k = _locate_cluster_raster(mission_maps, kmeans, emb)
    except FileNotFoundError as exc:
        print(f"[warn] {exc}")
        return None
    try:
        labels, max_label = _sample_cluster_labels(df["lat"].to_numpy(), df["lon"].to_numpy(), raster_path)
    except Exception as exc:
        print(f"[warn] Unable to sample cluster raster ({raster_path}): {exc}")
        return None
    segments = _segment_cluster_ids(labels)
    if not segments:
        print("[warn] Cluster shading requested but no valid cluster samples were found.")
        return None
    palette_len = inferred_k if inferred_k is not None else (max_label + 1 if max_label >= 0 else 0)
    if palette_len <= 0:
        palette_len = int(max(seg[2] for seg in segments) + 1)
    palette = _cluster_palette(palette_len)
    colors = {cid: palette[cid % len(palette)] for cid in {seg[2] for seg in segments}}
    desc = f"k={palette_len}, {emb} ({raster_path.name})"
    print(f"[info] Cluster shading enabled using {desc}")
    return ClusterShading(segments=segments, colors=colors, alpha=alpha, description=desc)


def _get_quaternion_block(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return quaternion columns if present, preferring *_WB."""
    if all(c in df.columns for c in ("qw_WB", "qx_WB", "qy_WB", "qz_WB")):
        qw = df["qw_WB"].to_numpy(dtype=np.float64)
        qx = df["qx_WB"].to_numpy(dtype=np.float64)
        qy = df["qy_WB"].to_numpy(dtype=np.float64)
        qz = df["qz_WB"].to_numpy(dtype=np.float64)
        return qw, qx, qy, qz
    if all(c in df.columns for c in ("qw", "qx", "qy", "qz")):
        print("[warn] Using (qw,qx,qy,qz) instead of qw_WB..qz_WB — confirm frame is world-aligned.")
        qw = df["qw"].to_numpy(dtype=np.float64)
        qx = df["qx"].to_numpy(dtype=np.float64)
        qy = df["qy"].to_numpy(dtype=np.float64)
        qz = df["qz"].to_numpy(dtype=np.float64)
        return qw, qx, qy, qz
    raise KeyError("Quaternion columns not found (need qw_WB..qz_WB or qw..qz).")


def _normalize_quat_arrays(qw: np.ndarray, qx: np.ndarray, qy: np.ndarray, qz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    n[n == 0.0] = 1.0
    return qw / n, qx / n, qy / n, qz / n


def _euler_zyx_from_qWB(qw: np.ndarray, qx: np.ndarray, qy: np.ndarray, qz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert unit quaternion q_WB to yaw/pitch/roll (ZYX ordering) in degrees."""
    qw, qx, qy, qz = _normalize_quat_arrays(qw, qx, qy, qz)

    xx = qx * qx; yy = qy * qy; zz = qz * qz
    xy = qx * qy; xz = qx * qz; yz = qy * qz
    wx = qw * qx; wy = qw * qy; wz = qw * qz

    r00 = 1.0 - 2.0 * (yy + zz)
    r10 = 2.0 * (xy + wz)
    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (xx + yy)

    yaw = np.arctan2(r10, r00)
    pitch = np.arctan2(-r20, np.clip(np.sqrt(r00 * r00 + r10 * r10), 1e-12, None))
    roll = np.arctan2(r21, r22)
    return np.rad2deg(yaw), np.rad2deg(pitch), np.rad2deg(roll)


def _compute_pitch_deg(df: pd.DataFrame) -> np.ndarray:
    qw, qx, qy, qz = _get_quaternion_block(df)
    _, pitch_deg, _ = _euler_zyx_from_qWB(qw, qx, qy, qz)
    return -pitch_deg  # align with navigation convention (nose-up positive)


def _overlay_pitch(ax, xx: np.ndarray, pitch_deg: np.ndarray):
    """Overlay pitch on a twin axis so metrics can be compared to slope."""
    ax2 = ax.twinx()
    ax2.plot(xx, pitch_deg, color=PITCH_COLOR, linewidth=1.2, label=PITCH_LABEL)
    ax2.set_ylabel(PITCH_LABEL, color=PITCH_COLOR)
    ax2.tick_params(axis="y", colors=PITCH_COLOR)
    ax2.spines["right"].set_color(PITCH_COLOR)
    ax2.grid(False)
    return ax2


def _add_zero_line(ax) -> None:
    """Draw a faint horizontal line at zero for quick visual reference."""
    ax.axhline(0.0, **ZERO_LINE_STYLE)

def _pick_synced(sync_dir: Path, hz: int | None) -> Path:
    # If Hz specified, try metrics file first
    if hz is not None:
        p_metrics = sync_dir / f"synced_{hz}Hz_metrics.parquet"
        if p_metrics.exists():
            return p_metrics
        p_plain = sync_dir / f"synced_{hz}Hz.parquet"
        if p_plain.exists():
            return p_plain
        raise FileNotFoundError(f"Neither {p_metrics} nor {p_plain} found")

    metrics = sorted(sync_dir.glob("synced_*Hz_metrics.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    if metrics:
        return metrics[0]

    plains = sorted(sync_dir.glob("synced_*Hz.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not plains:
        raise FileNotFoundError(f"No synced parquets in {sync_dir}")
    return plains[0]

def _detect_metric_cols(df: pd.DataFrame, include: Sequence[str] | None, whitelist: Iterable[str]) -> List[str]:
    if include:
        missing = [c for c in include if c not in df.columns]
        if missing:
            raise KeyError(f"Requested metrics not present: {missing}")
        return list(include)

    wl = [c for c in (whitelist or []) if isinstance(c, str)]
    if wl:
        missing = [c for c in wl if c not in df.columns]
        if missing:
            print(f"[warn] Metrics listed in config but missing from parquet: {missing}")
        cols = [c for c in wl if c in df.columns]
        if not cols:
            raise RuntimeError("None of the metrics listed in config/metrics.yaml are present in the dataframe.")
        return cols

    # fallback: keep previous heuristic so script still works if config lacks names
    num = df.select_dtypes(include="number").columns
    cols = [c for c in num if c not in {"t"}]
    return cols


def _build_metric_figure(
    df: pd.DataFrame,
    x_values: np.ndarray,
    x_label: str,
    metric_cols: Sequence[str],
    cumulative_set: set[str],
    filters_cfg: dict,
    add_speed_panel: bool,
    pitch_deg: np.ndarray | None,
    title: str,
    cluster_shading: ClusterShading | None,
) -> plt.Figure:
    """
    Build a stacked metric figure for a particular X-axis (time or distance).
    """
    n_extra = 1 if add_speed_panel else 0
    n_panels = len(metric_cols) + n_extra
    if n_panels == 0:
        raise RuntimeError("No metric columns detected. Run compute_metrics or specify --metrics.")

    fig_h = max(3.0, 2.3 * n_panels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, fig_h), sharex=True)
    if n_panels == 1:
        axes = [axes]

    ax_i = 0
    if add_speed_panel:
        if cluster_shading is not None:
            _apply_cluster_shading(axes[ax_i], x_values, cluster_shading)
        if "v_cmd" in df.columns:
            v_cmd = df["v_cmd"]
        elif {"v_cmd_x", "v_cmd_y"}.issubset(df.columns):
            v_cmd = np.hypot(df["v_cmd_x"], df["v_cmd_y"])
        else:
            v_cmd = None

        if "v_actual" in df.columns:
            v_act = df["v_actual"]
        elif {"vx", "vy"}.issubset(df.columns):
            v_act = np.hypot(df["vx"], df["vy"])
        else:
            v_act = None

        if v_cmd is not None:
            axes[ax_i].plot(x_values, v_cmd, label="v_cmd")
        if v_act is not None:
            axes[ax_i].plot(x_values, v_act, label="v_actual")
        axes[ax_i].set_ylabel("speed [m/s]")
        axes[ax_i].legend(loc="upper right")
        axes[ax_i].set_title("Commanded vs actual speed")
        _add_zero_line(axes[ax_i])
        if pitch_deg is not None:
            _overlay_pitch(axes[ax_i], x_values, pitch_deg)
        ax_i += 1

    for c in metric_cols:
        ax = axes[ax_i]
        if cluster_shading is not None:
            _apply_cluster_shading(ax, x_values, cluster_shading)
        raw_vals = df[c].to_numpy(dtype=np.float64)
        filt_vals = filter_signal(raw_vals, c, filters_cfg=filters_cfg, log_fn=print)
        series = filt_vals if filt_vals is not None else raw_vals
        ax.plot(x_values, series, label=c)
        ax.set_ylabel(c)
        ax.grid(True, alpha=0.25)
        _add_zero_line(ax)
        ax.legend(loc="upper right", frameon=False)
        if pitch_deg is not None and c not in cumulative_set:
            _overlay_pitch(ax, x_values, pitch_deg)
        ax_i += 1

    axes[-1].set_xlabel(x_label)
    fig.suptitle(title, y=0.995, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig

def main():
    ap = argparse.ArgumentParser(description="Plot metrics over time and/or distance for a mission.")
    ap.add_argument("--mission", required=True, help="Mission alias or UUID")
    ap.add_argument("--hz", type=int, default=None, help="Pick synced_<Hz>Hz.parquet (default: latest)")
    ap.add_argument("--metrics", nargs="*", default=None, help="Subset of metric columns to plot")
    ap.add_argument("--with-speeds", action="store_true", help="Add a top panel with v_cmd vs v_actual")
    ap.add_argument(
        "--overlay-pitch",
        action="store_true",
        help="Overlay pitch [deg] on each subplot using a secondary Y-axis.",
    )
    ap.add_argument(
        "--shade-clusters",
        action="store_true",
        help="Color subplot backgrounds by the mission's cluster assignments (default: off).",
    )
    ap.add_argument(
        "--cluster-embedding",
        choices=["dino", "stego"],
        default="dino",
        help="Embedding head to use when shading clusters (default: dino).",
    )
    ap.add_argument(
        "--cluster-kmeans",
        type=int,
        default=None,
        help="Explicit KMeans cluster count to locate (defaults to newest available for the embedding).",
    )
    ap.add_argument(
        "--cluster-alpha",
        type=float,
        default=0.12,
        help="Alpha for cluster background shading.",
    )
    ap.add_argument("--plot-time", action="store_true", default=True, help="Also render metrics vs time (default if no axis specified).")
    ap.add_argument("--plot-distance", action="store_true", default=True, help="Render metrics vs cumulative distance when dist_m is available.")
    ap.add_argument(
        "--out",
        default=None,
        help="Output path. Use a file path when generating a single plot or a directory when producing multiple plots.",
    )
    args = ap.parse_args()

    P = get_paths()
    metrics_cfg = load_metrics_config(Path(P["REPO_ROOT"]) / "config" / "metrics.yaml")
    filters_cfg = metrics_cfg.get("filters", {})
    mp = resolve_mission(args.mission, P)
    sync_dir, mission_id, display_name = mp.synced, mp.mission_id, mp.display

    synced_path = _pick_synced(sync_dir, args.hz)
    df = pd.read_parquet(synced_path)
    if "t" not in df.columns:
        raise KeyError(f"'t' column missing in {synced_path}")
    t0 = float(df["t"].iloc[0])
    tt = (df["t"] - t0).to_numpy(dtype=np.float64)

    pitch_deg = None
    if args.overlay_pitch:
        try:
            pitch_deg = _compute_pitch_deg(df)
        except Exception as exc:
            print(f"[warn] Unable to overlay pitch: {exc}")
            pitch_deg = None

    cluster_shading = None
    if args.shade_clusters:
        mission_maps = Path(mp.maps)
        cluster_shading = _prepare_cluster_shading(
            df,
            mission_maps,
            args.cluster_embedding,
            args.cluster_kmeans,
            args.cluster_alpha,
        )

    metrics_section = metrics_cfg.get("metrics", {}) if metrics_cfg else {}
    whitelist = metrics_section.get("names", []) if metrics_section else []
    cumulative_list = metrics_section.get("cumulative", []) if metrics_section else []
    cumulative_set = {c for c in cumulative_list if isinstance(c, str)}

    metric_cols = _detect_metric_cols(df, args.metrics, whitelist)
    if args.metrics is None and cumulative_set:
        ordered = [c for c in metric_cols if c not in cumulative_set]
        ordered.extend([c for c in metric_cols if c in cumulative_set])
        metric_cols = ordered

    # Decide whether to add the speed panel (explicit flag + available columns)
    has_cmd_speed = ("v_cmd" in df.columns) or ({"v_cmd_x", "v_cmd_y"}.issubset(df.columns))
    has_act_speed = ("v_actual" in df.columns) or ({"vx", "vy"}.issubset(df.columns))
    add_speed_panel = bool(args.with_speeds and (has_cmd_speed or has_act_speed))

    # Determine which axes to render. Default to time when nothing specified.
    plot_time = args.plot_time or not args.plot_distance
    plot_distance = args.plot_distance
    requests: list[tuple[str, np.ndarray, str, str]] = []
    if plot_time:
        requests.append(("time", tt, "time [s]", "timeseries"))
    if plot_distance:
        if "dist_m" not in df.columns:
            raise KeyError("dist_m column missing in synced parquet. Re-run sync_streams.py to add it.")
        dist = df["dist_m"].to_numpy(dtype=np.float64)
        dist = np.nan_to_num(dist, nan=0.0)
        dist = dist - dist[0]
        requests.append(("distance", dist, "distance traveled [m]", "distance"))

    out_arg = Path(args.out).expanduser() if args.out else None
    default_dir = P["REPO_ROOT"] / "reports" / display_name
    default_dir.mkdir(parents=True, exist_ok=True)
    multi = len(requests) > 1

    def resolve_out_path(suffix: str) -> Path:
        if multi:
            out_dir = out_arg if out_arg is not None else default_dir
            if out_dir.suffix:
                raise ValueError("When plotting multiple axes, --out must be a directory path.")
            out_dir.mkdir(parents=True, exist_ok=True)
            return out_dir / f"{display_name}_{suffix}.png"
        if out_arg is not None:
            parent = out_arg.parent
            parent.mkdir(parents=True, exist_ok=True)
            return out_arg
        return default_dir / f"{display_name}_{suffix}.png"

    title = f"{display_name} — {mission_id}"
    for axis_key, x_values, x_label, suffix in requests:
        fig = _build_metric_figure(
            df,
            x_values,
            x_label,
            metric_cols,
            cumulative_set,
            filters_cfg,
            add_speed_panel,
            pitch_deg,
            title=f"{title} ({axis_key})",
            cluster_shading=cluster_shading,
        )
        out_path = resolve_out_path(suffix)
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        print(f"[ok] saved {out_path}")

if __name__ == "__main__":
    main()
