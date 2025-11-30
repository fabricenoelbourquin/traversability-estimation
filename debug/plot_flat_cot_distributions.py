#!/usr/bin/env python3
"""
Plot distributions of cost-of-transport metrics on near-flat terrain segments.

The script filters each mission to user-specified time windows and to samples
with |pitch| <= pitch_limit_deg, then shows how cost_of_transport and
cost_of_transport_mech behave. It produces one figure per mission plus an
aggregate figure across all listed missions, with both time-sampled and
distance-weighted histograms.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make src/ importable when running from repo root
import sys
THIS_FILE = Path(__file__).resolve()
SRC_ROOT = THIS_FILE.parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.paths import get_paths
from utils.missions import resolve_mission
from utils.filtering import load_metrics_config
import metrics as metrics_mod


# Missions and time ranges (seconds from mission start) to inspect
MISSION_TIME_WINDOWS: dict[str, list[tuple[float, float]]] = {
    "ETH-1": [(75.0, 220.0)],
    "GRI-1": [(70.0, 180.0), (182.0, 210.0), (227.0, 250.0)],
    "LEICA-1": [(65.0, 95.0), (110.0, 155.0)],
    "KÄB-3": [(355.0, 425.0)],
}

METRICS_TO_PLOT: tuple[str, ...] = ("cost_of_transport", "cost_of_transport_mech")
DEFAULT_PITCH_LIMIT_DEG: float = 1.0
DEFAULT_REPORT_DIR = Path(get_paths()["REPO_ROOT"]) / "reports" / "zz_flat_traversability_analysis"


def _merge_dicts(base: Mapping, override: Mapping) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def _pick_synced(sync_dir: Path, hz: int | None) -> Path:
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


def _get_quaternion_block(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if all(c in df.columns for c in ("qw_WB", "qx_WB", "qy_WB", "qz_WB")):
        return (
            df["qw_WB"].to_numpy(dtype=np.float64),
            df["qx_WB"].to_numpy(dtype=np.float64),
            df["qy_WB"].to_numpy(dtype=np.float64),
            df["qz_WB"].to_numpy(dtype=np.float64),
        )
    if all(c in df.columns for c in ("qw", "qx", "qy", "qz")):
        return (
            df["qw"].to_numpy(dtype=np.float64),
            df["qx"].to_numpy(dtype=np.float64),
            df["qy"].to_numpy(dtype=np.float64),
            df["qz"].to_numpy(dtype=np.float64),
        )
    raise KeyError("Quaternion columns not found (need qw_WB..qz_WB or qw..qz).")


def _normalize_quat_arrays(qw: np.ndarray, qx: np.ndarray, qy: np.ndarray, qz: np.ndarray):
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    n[n == 0.0] = 1.0
    return qw / n, qx / n, qy / n, qz / n


def _euler_zyx_from_qWB(qw: np.ndarray, qx: np.ndarray, qy: np.ndarray, qz: np.ndarray):
    qw, qx, qy, qz = _normalize_quat_arrays(qw, qx, qy, qz)

    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    r00 = 1.0 - 2.0 * (yy + zz)
    r10 = 2.0 * (xy + wz)
    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (xx + yy)

    yaw = np.arctan2(r10, r00)
    pitch = np.arctan2(-r20, np.clip(np.sqrt(r00 * r00 + r10 * r10), 1e-12, None))
    roll = np.arctan2(r21, r22)
    return np.rad2deg(yaw), np.rad2deg(pitch), np.rad2deg(roll)


def _compute_pitch_deg(df: pd.DataFrame) -> np.ndarray | None:
    try:
        qw, qx, qy, qz = _get_quaternion_block(df)
    except KeyError as exc:
        print(f"[warn] {exc}")
        return None
    _, pitch_deg, _ = _euler_zyx_from_qWB(qw, qx, qy, qz)
    return -pitch_deg  # align with navigation convention (nose-up positive)


def _build_time_mask(times_s: np.ndarray, windows: Sequence[tuple[float, float]]) -> np.ndarray:
    mask = np.zeros_like(times_s, dtype=bool)
    for t0, t1 in windows:
        mask |= (times_s >= float(t0)) & (times_s <= float(t1))
    return mask


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "t" not in out.columns:
        raise KeyError("'t' column missing in dataframe.")
    t0 = float(out["t"].iloc[0])
    out["time_rel_s"] = out["t"] - t0

    if "dist_m" in out.columns:
        dist = out["dist_m"].to_numpy(dtype=np.float64)
        dist = np.nan_to_num(dist - dist[0], nan=0.0)
        out["dist_rel_m"] = dist
        if "dist_m_per_step" not in out.columns:
            step = np.diff(dist, prepend=dist[0])
            out["dist_m_per_step"] = step

    pitch_deg = _compute_pitch_deg(out)
    if pitch_deg is not None:
        out["pitch_deg"] = pitch_deg
    return out


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    sorter = np.argsort(values)
    v = values[sorter]
    w = weights[sorter]
    cum = np.cumsum(w)
    if cum[-1] == 0:
        return float(np.nan)
    target = percentile / 100.0 * cum[-1]
    return float(np.interp(target, cum, v))


def _box_stats(values: np.ndarray, weights: np.ndarray | None = None) -> dict:
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0:
        return {}
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != vals.shape or not np.any(w > 0):
            return {}
        q1 = _weighted_percentile(vals, w, 25.0)
        med = _weighted_percentile(vals, w, 50.0)
        q3 = _weighted_percentile(vals, w, 75.0)
        lo = _weighted_percentile(vals, w, 5.0)
        hi = _weighted_percentile(vals, w, 95.0)
    else:
        q1, med, q3, lo, hi = np.percentile(vals, [25, 50, 75, 5, 95])
    return {"med": med, "q1": q1, "q3": q3, "whislo": lo, "whishi": hi, "fliers": []}


def _slugify(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)


def _plot_metric_distribution(
    metric: str,
    values: np.ndarray,
    dist_weights: np.ndarray | None,
    title: str,
    output_path: Path,
) -> None:
    if values.size == 0:
        print(f"[warn] No samples for {metric}; skipping {output_path.name}")
        return

    has_dist_weights = dist_weights is not None and dist_weights.shape == values.shape and np.any(dist_weights > 0)

    fig, axes = plt.subplots(2, 2, figsize=(11, 6))
    axes = axes.reshape(2, 2)

    axes[0, 0].hist(values, bins=50, color="tab:blue", alpha=0.85)
    axes[0, 0].set_ylabel("samples (time-weighted)")
    axes[0, 0].set_xlabel(metric)
    axes[0, 0].grid(alpha=0.2)

    if has_dist_weights:
        axes[1, 0].hist(values, bins=50, weights=dist_weights, color="tab:green", alpha=0.8)
        axes[1, 0].set_ylabel("distance-weighted (m)")
    else:
        axes[1, 0].text(0.5, 0.5, "distance weights unavailable", ha="center", va="center", transform=axes[1, 0].transAxes)
        axes[1, 0].set_ylabel("distance-weighted (m)")
    axes[1, 0].set_xlabel(metric)
    axes[1, 0].grid(alpha=0.2)

    stats = [_box_stats(values)]
    labels = ["time-sampled"]
    if has_dist_weights:
        stats.append(_box_stats(values, dist_weights))
        labels.append("distance-weighted")
    axes[0, 1].bxp(stats, vert=False, showmeans=False, patch_artist=True)
    axes[0, 1].set_yticklabels(labels)
    axes[0, 1].set_xlabel(metric)
    axes[0, 1].set_title("Box summary")
    axes[0, 1].grid(alpha=0.2)

    axes[1, 1].hist(values, bins=50, density=True, color="0.6", alpha=0.6)
    axes[1, 1].set_xlabel(metric)
    axes[1, 1].set_ylabel("density (time-sampled)")
    axes[1, 1].grid(alpha=0.2)

    fig.suptitle(title, y=0.995, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"[ok] wrote {output_path}")
    plt.close(fig)


def _ensure_metrics(df: pd.DataFrame, cfg: dict, metric_names: Iterable[str]) -> pd.DataFrame:
    missing = [m for m in metric_names if m not in df.columns]
    if not missing:
        return df
    print(f"[info] Computing missing metrics: {missing}")
    return metrics_mod.compute(df, missing, cfg)


def _collect_values(
    df: pd.DataFrame,
    windows: Sequence[tuple[float, float]],
    pitch_limit_deg: float,
) -> dict[str, tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    if "pitch_deg" not in df.columns:
        raise KeyError("pitch_deg column missing; cannot filter by slope.")

    time_mask = _build_time_mask(df["time_rel_s"].to_numpy(dtype=np.float64), windows)
    pitch = df["pitch_deg"].to_numpy(dtype=np.float64)
    pitch_mask = np.isfinite(pitch) & (np.abs(pitch) <= pitch_limit_deg)
    mask = time_mask & pitch_mask

    if not np.any(mask):
        print("[warn] No samples within requested time windows + pitch limit.")
        return {}

    dist_weights_full = None
    if "dist_m_per_step" in df.columns:
        dist_weights_full = df["dist_m_per_step"].to_numpy(dtype=np.float64)
        dist_weights_full = np.clip(np.nan_to_num(dist_weights_full, nan=0.0), 0.0, None)

    results: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
    for metric in METRICS_TO_PLOT:
        if metric not in df.columns:
            print(f"[warn] Metric {metric} missing in dataframe; skipping.")
            continue
        vals_full = df[metric].to_numpy(dtype=np.float64)
        flat_mask = mask & np.isfinite(vals_full) & (vals_full != 0.0)
        rest_mask = (~mask) & np.isfinite(vals_full) & (vals_full != 0.0)

        vals_flat = vals_full[flat_mask]

        vals_rest = vals_full[rest_mask]

        w_dist = None
        if dist_weights_full is not None:
            w_dist = dist_weights_full[flat_mask]
        results[metric] = (vals_flat, w_dist, vals_rest)
    return results


def analyze_mission(
    mission: str,
    hz: int | None,
    pitch_limit_deg: float,
    cfg: dict,
    output_dir: Path,
) -> tuple[str, dict[str, tuple[np.ndarray, np.ndarray | None, np.ndarray]]]:
    P = get_paths()
    mp = resolve_mission(mission, P)
    synced_path = _pick_synced(mp.synced, hz)
    df = pd.read_parquet(synced_path)
    df = _ensure_metrics(df, cfg, METRICS_TO_PLOT)
    df = _prepare_df(df)

    windows = MISSION_TIME_WINDOWS.get(mp.display) or MISSION_TIME_WINDOWS.get(mission) or [(0.0, float("inf"))]
    results = _collect_values(df, windows, pitch_limit_deg)

    if not results:
        print(f"[warn] No results for {mp.display}")
        return mp.display, {}

    for metric, (vals, dist_w, _) in results.items():
        out_name = f"{_slugify(mp.display)}_{metric}_flat_pitch{pitch_limit_deg:.1f}.png"
        title = f"{mp.display} — {metric} (|pitch|<= {pitch_limit_deg}°, time windows)"
        _plot_metric_distribution(metric, vals, dist_w, title, output_dir / out_name)

    return mp.display, results


def _stack_samples(samples: Sequence[tuple[np.ndarray, np.ndarray | None]]) -> tuple[np.ndarray, np.ndarray | None]:
    val_list = [vals for vals, _ in samples if vals.size]
    if not val_list:
        return np.array([]), None
    all_vals = np.concatenate(val_list)
    weight_list = [w for _, w in samples if w is not None and w.size]
    all_weights = np.concatenate(weight_list) if weight_list else None
    return all_vals, all_weights


def _stack_arrays(samples: Sequence[np.ndarray]) -> np.ndarray:
    if not samples:
        return np.array([])
    non_empty = [s for s in samples if s.size]
    if not non_empty:
        return np.array([])
    return np.concatenate(non_empty)


def _plot_box_by_mission(metric: str, samples_by_mission: list[tuple[str, np.ndarray]], output_path: Path) -> None:
    samples = [(label, vals) for label, vals in samples_by_mission if vals.size]
    if not samples:
        print(f"[warn] No data to plot mission comparison for {metric}")
        return

    labels = [s[0] for s in samples]
    data = [s[1] for s in samples]
    means = [np.mean(d) for d in data]

    fig, ax = plt.subplots(figsize=(max(7.0, 1.4 * len(data)), 4.5))
    bp = ax.boxplot(data, labels=labels, showmeans=True, patch_artist=True)
    for mean, x in zip(means, range(1, len(data) + 1)):
        ax.plot(x, mean, marker="o", color="tab:red", markersize=4, zorder=3)
    ax.set_title(f"{metric} — flat segments (per mission)")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"[ok] wrote {output_path}")
    plt.close(fig)


def _plot_flat_vs_rest(metric: str, flat_vals: np.ndarray, rest_vals: np.ndarray, pitch_limit_deg: float, output_path: Path) -> None:
    if flat_vals.size == 0 or rest_vals.size == 0:
        print(f"[warn] Need both flat and rest samples for {metric}; skipping flat-vs-rest plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 6))
    axes = axes.reshape(2, 2)

    axes[0, 0].hist(flat_vals, bins=50, color="tab:blue", alpha=0.8)
    axes[0, 0].set_title(f"Flat only (|pitch|<= {pitch_limit_deg}°)")
    axes[0, 0].set_ylabel("samples (time)")
    axes[0, 0].set_xlabel(metric)
    axes[0, 0].grid(alpha=0.2)

    axes[0, 1].hist(rest_vals, bins=50, color="tab:orange", alpha=0.8)
    axes[0, 1].set_title("Non-flat (rest)")
    axes[0, 1].set_ylabel("samples (time)")
    axes[0, 1].set_xlabel(metric)
    axes[0, 1].grid(alpha=0.2)

    stats = [_box_stats(flat_vals), _box_stats(rest_vals)]
    axes[1, 0].bxp(stats, vert=False, showmeans=False, patch_artist=True)
    axes[1, 0].set_yticklabels(["flat", "rest"])
    axes[1, 0].set_xlabel(metric)
    axes[1, 0].set_title("Box summary")
    axes[1, 0].grid(alpha=0.2)

    axes[1, 1].hist([flat_vals, rest_vals], bins=50, density=True, label=["flat", "rest"], alpha=0.6)
    axes[1, 1].set_xlabel(metric)
    axes[1, 1].set_ylabel("density")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.2)

    fig.suptitle(f"{metric}: flat vs rest (all asphalt missions, |pitch|<= {pitch_limit_deg}°)", y=0.995, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"[ok] wrote {output_path}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot cost-of-transport distributions on near-flat terrain.")
    ap.add_argument(
        "--missions",
        nargs="*",
        default=list(MISSION_TIME_WINDOWS.keys()),
        help="Mission aliases or UUIDs to include (default: asphalt list).",
    )
    ap.add_argument("--hz", type=int, default=None, help="Pick synced_<Hz>Hz[_metrics].parquet (default: newest).")
    ap.add_argument(
        "--pitch-limit",
        type=float,
        default=DEFAULT_PITCH_LIMIT_DEG,
        help="Absolute pitch [deg] defining 'flat' (default: 1.0).",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory for saved figures (default: reports/flat_traversability_analysis).",
    )
    args = ap.parse_args()

    cfg_base = load_metrics_config(Path(get_paths()["REPO_ROOT"]) / "config" / "metrics.yaml")
    cfg_private = load_metrics_config(Path(get_paths()["REPO_ROOT"]) / "config" / "metrics.private.yaml")
    metrics_cfg = _merge_dicts(cfg_base, cfg_private)

    aggregated: dict[str, list[tuple[np.ndarray, np.ndarray | None]]] = {m: [] for m in METRICS_TO_PLOT}
    aggregated_rest: dict[str, list[np.ndarray]] = {m: [] for m in METRICS_TO_PLOT}
    per_mission_samples: dict[str, list[tuple[str, np.ndarray]]] = {m: [] for m in METRICS_TO_PLOT}

    for mission in args.missions:
        label, mission_results = analyze_mission(
            mission=mission,
            hz=args.hz,
            pitch_limit_deg=args.pitch_limit,
            cfg=metrics_cfg,
            output_dir=args.output_dir,
        )
        for metric, pair in mission_results.items():
            vals_flat, dist_w, vals_rest = pair
            aggregated[metric].append((vals_flat, dist_w))
            aggregated_rest[metric].append(vals_rest)
            per_mission_samples[metric].append((label, vals_flat))

    for metric, samples in aggregated.items():
        all_vals, all_weights = _stack_samples(samples)
        if all_vals.size == 0:
            print(f"[warn] No aggregate data for {metric}; skipping.")
            continue
        out_name = f"ALL_{metric}_flat_pitch{args.pitch_limit:.1f}.png"
        title = f"Aggregate flat segments — {metric} (|pitch|<= {args.pitch_limit}°)"
        _plot_metric_distribution(metric, all_vals, all_weights, title, args.output_dir / out_name)

    for metric, samples in per_mission_samples.items():
        out_name = f"COMPARE_MISSIONS_{metric}_flat_pitch{args.pitch_limit:.1f}.png"
        _plot_box_by_mission(metric, samples, args.output_dir / out_name)

    for metric in METRICS_TO_PLOT:
        flat_vals, _ = _stack_samples(aggregated[metric])
        rest_vals = _stack_arrays(aggregated_rest[metric])
        if flat_vals.size == 0 or rest_vals.size == 0:
            print(f"[warn] Missing flat/rest data for {metric}; skipping flat-vs-rest plot.")
            continue
        out_name = f"ALL_{metric}_flat_vs_rest_pitch{args.pitch_limit:.1f}.png"
        _plot_flat_vs_rest(metric, flat_vals, rest_vals, args.pitch_limit, args.output_dir / out_name)


if __name__ == "__main__":
    main()
