#!/usr/bin/env python3
"""
Visualize per-cluster statistics for a metric (default: power_mech) across missions.

Features:
  - Aggregates metric samples + cluster assignments for every mission with GPS data.
  - Computes mean/std/count per cluster and reports how many clusters had no samples.
  - Builds a 2x2 dashboard:
        [0,0] Mean metric per cluster with ±std error bars
        [0,1] Scatter: mean vs. sample count (color encodes std)
        [1,0] Sample count per cluster (sorted)
        [1,1] Box plot for the clusters with the most samples (configurable)

Example:
    python src/visualization/plot_cluster_metric_stats.py --metric power_mech --cluster-embedding dino
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -- repo imports -------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
SRC_ROOT = THIS_FILE.parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.paths import get_paths
from utils.missions import resolve_mission, MissionPaths
from utils.synced import resolve_synced_parquet
from visualization.cluster_shading import sample_cluster_labels_for_dataframe


def _load_requested_missions(args: argparse.Namespace, base_paths: dict) -> list[MissionPaths]:
    requested: list[str] = []
    if args.mission_file:
        mf = Path(args.mission_file)
        if not mf.exists():
            raise FileNotFoundError(f"Mission file not found: {mf}")
        for line in mf.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            requested.append(line)
    if args.mission:
        requested.extend(args.mission)

    if not requested:
        meta = json.loads(Path(base_paths["MISSIONS_JSON"]).read_text())
        requested = [mid for mid in meta.keys() if not mid.startswith("_")]

    seen_ids: set[str] = set()
    missions: list[MissionPaths] = []
    for name in requested:
        try:
            mp = resolve_mission(name, base_paths)
        except Exception as exc:
            print(f"[warn] skipping mission {name}: {exc}")
            continue
        if mp.mission_id in seen_ids:
            continue
        seen_ids.add(mp.mission_id)
        missions.append(mp)
    return missions


def _collect_cluster_samples(
    missions: list[MissionPaths],
    args: argparse.Namespace,
) -> tuple[dict[int, list[np.ndarray]], int, int, int | None]:
    """
    Returns:
      - values_by_cluster: cluster -> list of numpy arrays with metric samples
      - total_points: aggregated sample count
      - used_missions: number of missions that contributed data
      - expected_k: inferred cluster count (None if unknown)
    """
    values_by_cluster: dict[int, list[np.ndarray]] = defaultdict(list)
    total_points = 0
    used_missions = 0
    expected_k: int | None = None

    for mp in missions:
        synced_dir = Path(mp.synced)
        try:
            parquet_path = resolve_synced_parquet(synced_dir, args.hz, prefer_metrics=True)
        except FileNotFoundError as exc:
            print(f"[warn] {mp.display}: {exc}")
            continue

        try:
            df = pd.read_parquet(parquet_path)
        except Exception as exc:
            print(f"[warn] {mp.display}: failed to load {parquet_path.name}: {exc}")
            continue

        if args.metric not in df.columns:
            print(f"[warn] {mp.display}: metric '{args.metric}' not present in {parquet_path.name}")
            continue
        if "lat" not in df.columns or "lon" not in df.columns:
            print(f"[warn] {mp.display}: missing 'lat'/'lon' — skipping.")
            continue

        metric_vals = df[args.metric].to_numpy(dtype=np.float64)
        lat = df["lat"].to_numpy(dtype=np.float64)
        lon = df["lon"].to_numpy(dtype=np.float64)
        mask_valid = np.isfinite(metric_vals) & np.isfinite(lat) & np.isfinite(lon)
        if not mask_valid.any():
            print(f"[warn] {mp.display}: no finite samples for {args.metric}")
            continue

        metric_vals = metric_vals[mask_valid]
        sample = sample_cluster_labels_for_dataframe(
            df,
            Path(mp.maps),
            args.cluster_embedding,
            args.cluster_kmeans,
            mask=mask_valid,
        )
        if sample is None:
            print(f"[warn] {mp.display}: cluster sampling unavailable; skipping mission.")
            continue
        labels = sample.labels
        if labels.shape[0] != metric_vals.shape[0]:
            print(f"[warn] {mp.display}: cluster labels mismatch metric samples (labels={labels.shape[0]}, data={metric_vals.shape[0]})")
            continue

        valid_clusters = np.isfinite(labels)
        if not valid_clusters.any():
            print(f"[warn] {mp.display}: no valid cluster intersections.")
            continue

        labels = labels[valid_clusters].astype(np.int32, copy=False)
        metric_vals = metric_vals[valid_clusters]
        if labels.size == 0:
            continue

        if sample.n_clusters > 0:
            if expected_k is None:
                expected_k = sample.n_clusters
            elif expected_k != sample.n_clusters:
                print(f"[warn] {mp.display}: cluster count mismatch ({sample.n_clusters} vs {expected_k}). Using max.")
                expected_k = max(expected_k, sample.n_clusters)

        used_missions += 1
        total_points += metric_vals.size

        for cid in np.unique(labels):
            cid_mask = labels == cid
            vals = metric_vals[cid_mask]
            if vals.size == 0:
                continue
            values_by_cluster[cid].append(vals)

    return values_by_cluster, total_points, used_missions, expected_k


def _build_summary_dataframe(
    values_by_cluster: dict[int, list[np.ndarray]],
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """
    Returns:
      - summary DataFrame with columns [cluster, count, mean, std, median]
      - concatenated values per cluster (for box plots)
    """
    rows = []
    merged_values: dict[int, np.ndarray] = {}
    for cid in sorted(values_by_cluster):
        arrays = values_by_cluster[cid]
        if not arrays:
            continue
        data = np.concatenate(arrays)
        merged_values[cid] = data
        count = data.size
        mean = float(np.mean(data)) if count else np.nan
        std = float(np.std(data, ddof=1)) if count > 1 else 0.0
        median = float(np.median(data)) if count else np.nan
        rows.append({"cluster": cid, "count": count, "mean": mean, "std": std, "median": median})

    summary = pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)
    return summary, merged_values


def _set_sparse_xticks(ax, positions: list[float], labels: list[str]) -> None:
    """
    Avoid overcrowded tick labels by down-sampling when necessary.
    """
    if not positions or not labels:
        return
    max_ticks = 30
    n = min(len(positions), len(labels))
    if n <= max_ticks:
        idxs = list(range(n))
    else:
        step = max(1, n // max_ticks)
        idxs = list(range(0, n, step))
    ticks = [positions[i] for i in idxs]
    tick_labels = [labels[i] for i in idxs]
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")


def _plot_dashboard(
    summary: pd.DataFrame,
    merged_values: dict[int, np.ndarray],
    args: argparse.Namespace,
    empty_clusters: int,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    if summary.empty:
        for ax in axes.flat:
            ax.text(0.5, 0.5, "No cluster data", ha="center", va="center", fontsize=14)
            ax.axis("off")
        fig.tight_layout()
        return fig

    # (0,0) Mean metric per cluster with std as error bars
    ax = axes[0, 0]
    summary_by_count = summary.sort_values("count", ascending=False)
    if args.mean_top_k > 0 and args.mean_top_k < len(summary_by_count):
        summary_subset = summary_by_count.head(args.mean_top_k)
    else:
        summary_subset = summary_by_count
    summary_by_mean = summary_subset.sort_values("mean", ascending=False)
    x_mean = list(range(len(summary_by_mean)))
    ax.bar(x_mean, summary_by_mean["mean"], yerr=summary_by_mean["std"], alpha=0.8, capsize=3, color="tab:blue")
    _set_sparse_xticks(ax, x_mean, [str(cid) for cid in summary_by_mean["cluster"]])
    ax.set_ylabel(f"{args.metric} mean")
    ax.set_xlabel("Cluster ID")
    if args.mean_top_k > 0:
        ax.set_title(f"Mean {args.metric} per cluster (±std, top {len(summary_by_mean)} by count)")
    else:
        ax.set_title(f"Mean {args.metric} per cluster (±std)")
    ax.grid(axis="y", alpha=0.3)
    ax.text(0.99, 0.95, f"{empty_clusters} empty clusters", transform=ax.transAxes, ha="right", va="top", fontsize=9)

    # (0,1) Scatter: mean vs count, colored by std
    ax = axes[0, 1]
    scatter = ax.scatter(summary["count"], summary["mean"], c=summary["std"], cmap="viridis", s=40, edgecolor="k", linewidth=0.4)
    ax.set_xlabel("Samples per cluster")
    ax.set_ylabel(f"{args.metric} mean")
    ax.set_title("Mean vs. sample count (color = std)")
    ax.grid(alpha=0.3)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Std dev")

    # (1,0) Sample counts per cluster (sorted descending)
    ax = axes[1, 0]
    sorted_counts = summary.sort_values("count", ascending=False)
    x_counts = list(range(len(sorted_counts)))
    ax.bar(x_counts, sorted_counts["count"], color="tab:gray")
    ax.set_xlabel("Cluster ID (sorted by sample count)")
    ax.set_ylabel("Samples")
    ax.set_title("Sample availability per cluster")
    _set_sparse_xticks(ax, x_counts, [str(cid) for cid in sorted_counts["cluster"]])
    ax.grid(axis="y", alpha=0.3)

    # (1,1) Box plot for top-K clusters by count
    ax = axes[1, 1]
    top_k = args.boxplot_top_k
    if top_k <= 0 or top_k > len(sorted_counts):
        top_k = len(sorted_counts)
    top_subset = sorted_counts.head(top_k)
    if top_subset.empty:
        ax.text(0.5, 0.5, "No data for box plot", ha="center", va="center")
        ax.axis("off")
    else:
        box_data = [merged_values[cid] for cid in top_subset["cluster"]]
        ax.boxplot(box_data, labels=[str(cid) for cid in top_subset["cluster"]], showfliers=False)
        ax.set_xlabel("Cluster ID")
        ax.set_ylabel(args.metric)
        ax.set_title(f"{args.metric} distribution (top {top_k} clusters by count)")
        ax.tick_params(axis="x", rotation=45)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description="Plot aggregated per-cluster metric statistics.")
    ap.add_argument("--metric", default="power_mech", help="Metric column to analyze (default: power_mech).")
    ap.add_argument("--hz", type=int, default=None, help="Preferred synced Hz (default: latest available).")
    ap.add_argument("--cluster-embedding", default="dino", help="Embedding head to locate cluster rasters (default: dino).")
    ap.add_argument("--cluster-kmeans", type=int, default=None, help="Explicit K for cluster map selection.")
    ap.add_argument("--mission", action="append", help="Mission alias/UUID to include (repeatable). Defaults to all in config.")
    ap.add_argument("--mission-file", help="Optional text file with one mission alias/UUID per line.")
    ap.add_argument("--boxplot-top-k", type=int, default=12, help="How many clusters to include in the box plot (default: 12, 0 = all).")
    ap.add_argument(
        "--mean-top-k",
        type=int,
        default=0,
        help="Limit the mean/STD bar plot to clusters with the highest sample counts (0 = show all).",
    )
    ap.add_argument("--save", help="Optional path to save the dashboard figure (e.g., reports/cluster_stats.png).")
    ap.add_argument("--no-show", action="store_true", help="Skip plt.show(); useful for headless runs combined with --save.")
    args = ap.parse_args()

    base_paths = get_paths()
    missions = _load_requested_missions(args, base_paths)
    if not missions:
        raise SystemExit("No missions resolved. Provide --mission/--mission-file or update config/missions.json.")

    (
        values_by_cluster,
        total_points,
        used_missions,
        expected_k,
    ) = _collect_cluster_samples(missions, args)

    if not values_by_cluster:
        raise SystemExit("No cluster data collected. Check metric name, cluster embedding, or mission coverage.")

    summary_df, merged_values = _build_summary_dataframe(values_by_cluster)

    observed_clusters = set(summary_df["cluster"].tolist())
    if expected_k is None and observed_clusters:
        expected_k = max(observed_clusters) + 1
    if expected_k is None:
        expected_k = 0
    empty_clusters = max(0, expected_k - len(observed_clusters))

    print(f"[info] Aggregated {total_points} samples from {used_missions} missions.")
    print(f"[info] Found {len(observed_clusters)} populated clusters, {empty_clusters} with zero samples.")

    fig = _plot_dashboard(summary_df, merged_values, args, empty_clusters)

    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        print(f"[info] Saved figure to {out_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
