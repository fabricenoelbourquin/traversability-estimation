#!/usr/bin/env python3
"""
Compare cumulative metrics across multiple missions with optional per-mission time windows.

Examples:
  python src/visualization/compare_cumulative_metrics.py --missions ETH-1 ETH-2
  python src/visualization/compare_cumulative_metrics.py --missions ETH-1 TRIM-3 \\
      --metrics energy_pos_cum joint_travel_cum_L1 \\
      --windows ETH-1:60:120 TRIM-3:40:100
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_FILE = Path(__file__).resolve()
SRC_ROOT = THIS_FILE.parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.paths import get_paths
from utils.missions import MissionPaths, resolve_mission
from utils.filtering import load_metrics_config
from utils.synced import resolve_synced_parquet


@dataclass
class MissionSeries:
    """Container for one mission's sliced time-series."""

    key: str
    display: str
    mission_id: str
    folder: str
    time_rel_s: np.ndarray
    metric_data: dict[str, np.ndarray]
    window_requested: tuple[float | None, float | None] | None
    window_applied: tuple[float, float]
    time_range_full: tuple[float, float]
    source_path: Path


def parse_time_windows(specs: Sequence[str]) -> dict[str, tuple[float | None, float | None]]:
    """
    Parse window specs like `MISSION:START:END`.
    Missing START/END (empty string) indicates open-ended bounds.
    """
    windows: dict[str, tuple[float | None, float | None]] = {}
    for raw in specs:
        spec = raw.strip()
        if not spec:
            continue
        mission_key, sep, remainder = spec.partition(":")
        if not sep:
            raise ValueError(f"Window spec '{spec}' is missing ':' separators (expected MISSION:START:END)")
        parts = remainder.split(":")
        if len(parts) == 1:
            start_str, end_str = parts[0], ""
        elif len(parts) == 2:
            start_str, end_str = parts
        else:
            raise ValueError(f"Window spec '{spec}' has too many ':' separators; use MISSION:START:END")

        def _to_float(txt: str) -> float | None:
            txt = txt.strip()
            if not txt:
                return None
            try:
                return float(txt)
            except ValueError as exc:
                raise ValueError(f"Could not parse '{txt}' in window spec '{spec}' as a float.") from exc

        start = _to_float(start_str)
        end = _to_float(end_str)
        if start is not None and end is not None and end < start:
            start, end = end, start
        windows[mission_key] = (start, end)
    return windows


def resolve_window_for_mission(
    window_specs: Mapping[str, tuple[float | None, float | None]],
    mission_arg: str,
    mp: MissionPaths,
) -> tuple[float | None, float | None] | None:
    """Accept multiple aliases when matching a window spec to a mission."""
    candidates = [mission_arg, mp.mission_id, mp.folder, mp.display]
    for key in candidates:
        if key in window_specs:
            return window_specs[key]
    return None


def _zero_relative(values: np.ndarray) -> np.ndarray:
    """Shift cumulative values so each mission starts at zero for direct comparison."""
    if values.size == 0:
        return values
    finite = np.isfinite(values)
    if not np.any(finite):
        return values
    offset = values[finite][0]
    return values - offset


def load_mission_series(
    mission_arg: str,
    mp: MissionPaths,
    metric_names: Sequence[str],
    window: tuple[float | None, float | None] | None,
    hz: int | None,
) -> tuple[MissionSeries, list[str]]:
    """Load the selected mission parquet and slice the requested window."""
    synced_path = resolve_synced_parquet(mp.synced, hz, prefer_metrics=True)
    df = pd.read_parquet(synced_path)
    if "t" not in df.columns:
        raise KeyError(f"'t' column missing in {synced_path}")
    df = df.sort_values("t").reset_index(drop=True)
    t_vals = df["t"].to_numpy(dtype=np.float64)
    if t_vals.size == 0:
        raise RuntimeError(f"No samples found in {synced_path}")
    t_rel = t_vals - float(t_vals[0])
    full_range = (float(t_rel.min()), float(t_rel.max()))

    mask = np.ones_like(t_rel, dtype=bool)
    start_req = window[0] if window else None
    end_req = window[1] if window else None
    if start_req is not None:
        mask &= t_rel >= start_req
    if end_req is not None:
        mask &= t_rel <= end_req
    if not np.any(mask):
        print(f"[warn] Window {window} on mission '{mp.display}' is empty; using full time range.")
        mask[:] = True
        start_req = None
        end_req = None
        window = None

    t_window = t_rel[mask]
    t_zero = t_window - float(t_window[0])
    window_applied = (float(t_window[0]), float(t_window[-1]))

    metric_data: dict[str, np.ndarray] = {}
    missing_cols: list[str] = []
    for name in metric_names:
        if name not in df.columns:
            missing_cols.append(name)
            continue
        raw = df.loc[mask, name].to_numpy(dtype=np.float64)
        metric_data[name] = _zero_relative(raw)

    series = MissionSeries(
        key=mission_arg,
        display=mp.display,
        mission_id=mp.mission_id,
        folder=mp.folder,
        time_rel_s=t_zero,
        metric_data=metric_data,
        window_requested=window,
        window_applied=window_applied,
        time_range_full=full_range,
        source_path=synced_path,
    )
    return series, missing_cols


def format_window_suffix(ms: MissionSeries) -> str:
    if ms.window_requested is None:
        return ""
    start, end = ms.window_applied
    if start == end:
        return f" ({start:.1f}s)"
    return f" ({start:.1f}-{end:.1f}s)"


def build_output_path(repo_root: Path, missions: Sequence[MissionSeries], metrics: Sequence[str]) -> Path:
    mission_slug = "--".join(ms.folder for ms in missions)
    metric_slug = "--".join(metrics)
    reports_dir = repo_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir / f"compare_cumulative_{metric_slug}_{mission_slug}.png"


def plot_series(
    missions: Sequence[MissionSeries],
    metric_names: Sequence[str],
    *,
    width: float,
    height_per_metric: float,
    linewidth: float,
    legend_loc: str,
    title: str | None = None,
):
    rows = len(metric_names)
    fig, axes = plt.subplots(rows, 1, figsize=(width, height_per_metric * rows), sharex=False)
    if rows == 1:
        axes = [axes]

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not colors:
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for idx, metric in enumerate(metric_names):
        ax = axes[idx]
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)
        for midx, ms in enumerate(missions):
            values = ms.metric_data.get(metric)
            if values is None or values.size == 0:
                continue
            color = colors[midx % len(colors)]
            label = f"{ms.display}{format_window_suffix(ms)}" if idx == 0 else None
            ax.plot(ms.time_rel_s, values, color=color, linewidth=linewidth, label=label)
        if idx == rows - 1:
            ax.set_xlabel("time [s] (window-relative)")
    if missions and metric_names:
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(handles, labels, loc=legend_loc)
    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        fig.tight_layout()
    return fig


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Overlay cumulative metrics from multiple missions.")
    ap.add_argument(
        "--missions",
        nargs="+",
        required=True,
        help="List of mission aliases or IDs (>=2). Each mission becomes one line per subplot.",
    )
    ap.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Cumulative metrics to plot. Default: metrics.metrics.cumulative from config, else energy_pos_cum.",
    )
    ap.add_argument(
        "--windows",
        nargs="*",
        default=None,
        metavar="MISSION:START:END",
        help="Per-mission time windows in seconds relative to mission start (e.g. ETH-1:60:120).",
    )
    ap.add_argument("--hz", type=int, default=None, help="Pick a specific synced_<Hz>Hz[_metrics].parquet.")
    ap.add_argument("--out", type=str, default=None, help="Output image path. Default: reports/compare_cumulative_*.png")
    ap.add_argument("--width", type=float, default=10.0, help="Figure width in inches.")
    ap.add_argument("--height-per", type=float, default=3.0, help="Height per metric subplot in inches.")
    ap.add_argument("--linewidth", type=float, default=2.0, help="Line width for each mission trace.")
    ap.add_argument("--legend-loc", default="upper left", help="Legend location code (matplotlib style).")
    ap.add_argument("--title", default=None, help="Optional figure title.")
    ap.add_argument("--show", action="store_true", help="Show the plot interactively (in addition to saving if --out).")
    args = ap.parse_args(argv)

    if len(args.missions) < 2:
        raise SystemExit("Provide at least two missions via --missions.")

    try:
        windows = parse_time_windows(args.windows or [])
    except ValueError as exc:
        raise SystemExit(str(exc))

    P = get_paths()
    repo_root = Path(P["REPO_ROOT"])
    metrics_cfg = load_metrics_config(repo_root / "config" / "metrics.yaml")
    cfg_metrics = metrics_cfg.get("metrics", {}) if metrics_cfg else {}
    cfg_cumulative = [m for m in cfg_metrics.get("cumulative", []) if isinstance(m, str)]

    metric_names = args.metrics if args.metrics else (cfg_cumulative or ["energy_pos_cum"])

    missions_data: list[MissionSeries] = []
    missing_global: dict[str, list[str]] = {}
    for mission_arg in args.missions:
        mp = resolve_mission(mission_arg, P)
        window = resolve_window_for_mission(windows, mission_arg, mp)
        series, missing = load_mission_series(mission_arg, mp, metric_names, window, args.hz)
        missions_data.append(series)
        for name in missing:
            missing_global.setdefault(name, []).append(mp.display)

    # Filter out metrics that are missing everywhere.
    available_metrics = []
    for name in metric_names:
        if any(name in ms.metric_data for ms in missions_data):
            available_metrics.append(name)
        else:
            print(f"[warn] Metric '{name}' missing from all selected missions; skipping.")
    if not available_metrics:
        raise SystemExit("None of the requested metrics are present in the selected missions.")

    for metric_name, mission_displays in sorted(missing_global.items()):
        print(f"[info] Metric '{metric_name}' missing in missions: {', '.join(mission_displays)}")

    fig = plot_series(
        missions_data,
        available_metrics,
        width=args.width,
        height_per_metric=args.height_per,
        linewidth=args.linewidth,
        legend_loc=args.legend_loc,
        title=args.title or f"Cumulative comparison ({', '.join(available_metrics)})",
    )

    out_path = Path(args.out) if args.out else build_output_path(repo_root, missions_data, available_metrics)
    fig.savefig(out_path, dpi=150)
    print(f"[info] Saved figure to {out_path}")
    if args.show:
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
