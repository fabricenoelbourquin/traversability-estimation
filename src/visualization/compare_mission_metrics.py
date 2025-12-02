#!/usr/bin/env python3
"""
Compare a metric between two missions along distance-aligned segments.

Example:
  python src/visualization/compare_mission_metrics.py \
      --mission-1 ETH-1 --mission-2 ICE-1 \
      --segment-distance 10 --start-1 20 --start-2 5 \
      --metric power_mech
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow `import utils.*`
import sys

THIS_FILE = Path(__file__).resolve()
SRC_ROOT = THIS_FILE.parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.paths import get_paths
from utils.missions import resolve_mission
from utils.filtering import filter_signal, load_metrics_config
from utils.synced import resolve_synced_parquet


@dataclass
class MissionSegment:
    mission_display: str
    metric_name: str
    start_distance_m: float
    distances_m: np.ndarray
    values: np.ndarray

    def legend_label(self) -> str:
        start_str = f"{self.start_distance_m:g} m"
        metric = self.metric_name
        return f"{self.mission_display} ({metric}, start {start_str})"


def extract_metric_segment(
    df: pd.DataFrame,
    metric_name: str,
    start_m: float,
    length_m: float,
    filters_cfg: dict,
    cumulative_metrics: set[str],
) -> tuple[np.ndarray, np.ndarray]:
    if "dist_m" not in df.columns:
        raise KeyError("dist_m column missing. Run sync_streams.py with distance support enabled.")
    if metric_name not in df.columns:
        raise KeyError(f"Metric '{metric_name}' not present in dataframe columns.")
    if length_m <= 0.0:
        raise ValueError("Segment length must be positive.")

    dist = df["dist_m"].to_numpy(dtype=np.float64)
    metric = df[metric_name].to_numpy(dtype=np.float64)
    metric = filter_signal(metric, metric_name, filters_cfg=filters_cfg, log_fn=None)
    if metric is None:
        raise RuntimeError(f"Unable to compute signal '{metric_name}'.")

    mask = np.isfinite(dist) & np.isfinite(metric)
    dist = dist[mask]
    metric = metric[mask]
    if dist.size == 0:
        raise RuntimeError("No valid distance samples available after filtering NaNs.")

    end_m = start_m + length_m
    seg_mask = (dist >= start_m) & (dist <= end_m)
    if not np.any(seg_mask):
        raise ValueError(
            f"No samples exist between {start_m} m and {end_m} m "
            f"for metric '{metric_name}'."
        )

    seg_dist = dist[seg_mask] - start_m
    seg_vals = metric[seg_mask].copy()

    # Ensure distances are monotonically increasing within the window.
    order = np.argsort(seg_dist)
    seg_dist = seg_dist[order]
    seg_vals = seg_vals[order]

    if metric_name in cumulative_metrics and seg_vals.size:
        seg_vals = seg_vals - seg_vals[0]

    seg_dist = np.clip(seg_dist, 0.0, None)
    return seg_dist, seg_vals


def load_segment_for_mission(
    mission_name: str,
    metric_name: str,
    start_m: float,
    window_m: float,
    hz: int | None,
    filters_cfg: dict,
    cumulative_metrics: set[str],
) -> MissionSegment:
    paths = get_paths()
    mission = resolve_mission(mission_name, paths)
    parquet = resolve_synced_parquet(mission.synced, hz, prefer_metrics=True)
    df = pd.read_parquet(parquet)

    distances, values = extract_metric_segment(
        df, metric_name, start_m, window_m, filters_cfg, cumulative_metrics
    )

    if distances.size and distances[-1] < window_m - 1e-2:
        print(
            f"[warn] {mission.display} provides "
            f"{distances[-1]:.2f} m of data within the requested window of {window_m:.2f} m."
        )

    return MissionSegment(
        mission_display=mission.display,
        metric_name=metric_name,
        start_distance_m=start_m,
        distances_m=distances,
        values=values,
    )


def plot_segments(
    segments: Sequence[MissionSegment],
    window_m: float,
    title: str | None,
    output: Path | None,
):
    if not segments:
        raise ValueError("No segments provided for plotting.")

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for idx, seg in enumerate(segments):
        color = colors[idx % len(colors)]
        ax.plot(
            seg.distances_m,
            seg.values,
            color=color,
            linewidth=1.6,
            label=seg.legend_label(),
        )

    ax.set_xlim(0.0, window_m)
    ax.set_xlabel("distance from start [m]")
    metric_names = {seg.metric_name for seg in segments}
    if len(metric_names) == 1:
        ax.set_ylabel(next(iter(metric_names)))
    else:
        ax.set_ylabel("metric value")

    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)
    ax.legend()
    if title:
        ax.set_title(title)
    else:
        mission_titles = " vs ".join(seg.mission_display for seg in segments)
        ax.set_title(f"{mission_titles} (window {window_m:g} m)")
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=200)
        print(f"[info] Figure saved to {output}")
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compare metrics vs distance between two missions."
    )
    ap.add_argument("--mission-1", required=True, help="First mission alias/UUID.")
    ap.add_argument("--mission-2", required=True, help="Second mission alias/UUID.")
    ap.add_argument(
        "--metric",
        default="power_mech",
        help="Metric to plot for both missions (overridden by --metric-1/--metric-2).",
    )
    ap.add_argument(
        "--metric-1",
        dest="metric_1",
        default=None,
        help="Metric override for mission 1.",
    )
    ap.add_argument(
        "--metric-2",
        dest="metric_2",
        default=None,
        help="Metric override for mission 2.",
    )
    ap.add_argument(
        "--segment-distance",
        "--distance",
        dest="segment_distance",
        type=float,
        default=10.0,
        help="Length of the distance window to compare (meters).",
    )
    ap.add_argument(
        "--start-1",
        type=float,
        default=0.0,
        help="Distance along mission 1 where the comparison window should start.",
    )
    ap.add_argument(
        "--start-2",
        type=float,
        default=0.0,
        help="Distance along mission 2 where the comparison window should start.",
    )
    ap.add_argument(
        "--hz",
        type=int,
        default=None,
        help="Force a specific synced_*Hz parquet if multiple exist.",
    )
    ap.add_argument(
        "--title",
        default=None,
        help="Optional plot title. Defaults to 'mission vs mission'.",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save the figure to this path instead of displaying it.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    metrics_cfg = load_metrics_config(SRC_ROOT.parent / "config" / "metrics.yaml")
    cumulative_metrics = set(metrics_cfg.get("metrics", {}).get("cumulative", []))
    filters_cfg = metrics_cfg.get("filters", {})

    metric_1 = args.metric_1 or args.metric
    metric_2 = args.metric_2 or args.metric

    seg1 = load_segment_for_mission(
        args.mission_1,
        metric_1,
        args.start_1,
        args.segment_distance,
        args.hz,
        filters_cfg,
        cumulative_metrics,
    )
    seg2 = load_segment_for_mission(
        args.mission_2,
        metric_2,
        args.start_2,
        args.segment_distance,
        args.hz,
        filters_cfg,
        cumulative_metrics,
    )

    plot_segments([seg1, seg2], args.segment_distance, args.title, args.output)


if __name__ == "__main__":
    main()
