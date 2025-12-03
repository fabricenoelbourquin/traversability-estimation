#!/usr/bin/env python3
"""
Plot distributions of cost-of-transport metrics on near-flat patches from the
HDF5 dataset produced by build_patch_dataset.py. The plotting layout matches
debug/plot_flat_cot_distributions.py; only the data source changes.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Sequence

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


METRICS_TO_PLOT: tuple[tuple[str, str], ...] = (
    ("metric_cost_of_transport_mean", "cost_of_transport"),
    ("metric_cost_of_transport_mech_mean", "cost_of_transport_mech"),
)
DEFAULT_PITCH_LIMIT_DEG: float = 1.0
DEFAULT_SLOPE_LIMIT_DEG: float = 1.0
DEFAULT_PATCH_SIZE_M: float = 5.0
DEFAULT_REPORT_DIR = Path(get_paths()["REPO_ROOT"]) / "reports" / "zz_incline_patch_analysis"


def _slugify(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)


def _normalize_quat_arrays(qw: np.ndarray, qx: np.ndarray, qy: np.ndarray, qz: np.ndarray):
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    n[n == 0.0] = 1.0
    return qw / n, qx / n, qy / n, qz / n


def _euler_zyx_from_qWB(qw: np.ndarray, qx: np.ndarray, qy: np.ndarray, qz: np.ndarray):
    # ensure unit quaternions
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
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)

    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (xx + zz)
    r12 = 2.0 * (yz - wx)

    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (xx + yy)

    yaw = np.arctan2(r10, r00)
    pitch = np.arctan2(-r20, np.clip(np.sqrt(r00 * r00 + r10 * r10), 1e-12, None))
    roll = np.arctan2(r21, r22)

    yaw_deg = np.rad2deg(yaw) * -1.0  # navigation-friendly (clockwise positive)
    pitch_deg = np.rad2deg(pitch) * -1.0  # nose-up positive
    roll_deg = np.rad2deg(roll)
    return yaw_deg, pitch_deg, roll_deg


def _get_quaternion_block(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    candidates = [
        ("bearing_qw", "bearing_qx", "bearing_qy", "bearing_qz"),
        ("qw_WB", "qx_WB", "qy_WB", "qz_WB"),
        ("qw", "qx", "qy", "qz"),
    ]
    for cols in candidates:
        if all(c in df.columns for c in cols):
            return tuple(df[c].to_numpy(dtype=np.float64) for c in cols)  # type: ignore
    raise KeyError("Quaternion columns not found (need bearing_qw..qz or qw_WB..qz_WB or qw..qz).")


def _compute_pitch_deg(df: pd.DataFrame) -> np.ndarray | None:
    try:
        qw, qx, qy, qz = _get_quaternion_block(df)
    except KeyError as exc:
        print(f"[warn] {exc}")
        return None
    _, pitch_deg, _ = _euler_zyx_from_qWB(qw, qx, qy, qz)
    return pitch_deg


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


def _plot_flat_vs_rest(metric: str, flat_vals: np.ndarray, rest_vals: np.ndarray, flatness_desc: str, output_path: Path) -> None:
    if flat_vals.size == 0 or rest_vals.size == 0:
        print(f"[warn] Need both flat and rest samples for {metric}; skipping flat-vs-rest plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 6))
    axes = axes.reshape(2, 2)

    axes[0, 0].hist(flat_vals, bins=50, color="tab:blue", alpha=0.8)
    axes[0, 0].set_title(f"Flat only ({flatness_desc})")
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

    fig.suptitle(f"{metric}: flat vs rest (patches, {flatness_desc})", y=0.995, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"[ok] wrote {output_path}")
    plt.close(fig)


def format_patch_size_label(size_m: float) -> str:
    return f"{size_m:.3f}".rstrip("0").rstrip(".")


def _decode_attr_val(val):
    if isinstance(val, (bytes, bytearray)):
        try:
            return val.decode("utf-8")
        except Exception:
            return val
    return val


def _load_patch_groups(h5_path: Path, missions: Sequence[str] | None) -> list[tuple[str, str, pd.DataFrame]]:
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise SystemExit("h5py is required to read the patch dataset (pip install h5py).") from exc

    if not h5_path.exists():
        raise SystemExit(f"Dataset not found: {h5_path}")

    requested = set(missions) if missions else None
    groups: list[tuple[str, str, pd.DataFrame]] = []
    with h5py.File(h5_path, "r") as f:
        for grp_name in sorted(f.keys()):
            grp = f[grp_name]
            attrs = {k: _decode_attr_val(v) for k, v in grp.attrs.items()}
            display = str(attrs.get("mission_display") or grp_name)
            if requested and grp_name not in requested and display not in requested:
                continue
            ds = grp["patches"]
            records = ds[:]
            df = pd.DataFrame.from_records(records)
            df.columns = [c.decode("utf-8") if isinstance(c, (bytes, bytearray)) else str(c) for c in df.columns]
            col_order_attr = ds.attrs.get("column_order")
            col_order: list[str] = []
            if isinstance(col_order_attr, (bytes, str)):
                try:
                    col_order = list(json.loads(col_order_attr))
                except Exception:
                    col_order = []
            if col_order:
                missing_cols = [c for c in col_order if c not in df.columns]
                if not missing_cols:
                    df = df[col_order]
            groups.append((grp_name, display, df))
    if requested:
        missing = requested - {g for g, _, _ in groups}
        if missing:
            print(f"[warn] Requested missions not found in dataset: {sorted(missing)}")
    return groups


def _prepare_patch_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pitch_deg = _compute_pitch_deg(out)
    if pitch_deg is not None:
        out["pitch_deg"] = pitch_deg
    if "slope_mag_mean" in out.columns:
        slope_mag = out["slope_mag_mean"].to_numpy(dtype=np.float64)
        out["slope_mag_mean_deg"] = np.rad2deg(np.arctan(slope_mag))
    return out


def _build_flat_mask(df: pd.DataFrame, column: str, limit_deg: float, use_abs: bool) -> np.ndarray:
    if column not in df.columns:
        print(f"[warn] {column} column missing; cannot filter flat patches.")
        return np.zeros(len(df), dtype=bool)
    vals = df[column].to_numpy(dtype=np.float64)
    mask = np.isfinite(vals)
    if use_abs:
        mask &= (np.abs(vals) <= limit_deg)
    else:
        mask &= (vals <= limit_deg)
    return mask


def _collect_patch_values(df: pd.DataFrame, flat_mask: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    results: dict[str, tuple[np.ndarray, np.ndarray | None, np.ndarray]] = {}
    dist_weights_full = None
    if "distance_traveled_m" in df.columns:
        dist_weights_full = df["distance_traveled_m"].to_numpy(dtype=np.float64)
        dist_weights_full = np.clip(np.nan_to_num(dist_weights_full, nan=0.0), 0.0, None)

    for metric_key, label in METRICS_TO_PLOT:
        if metric_key not in df.columns:
            print(f"[warn] Metric column {metric_key} missing; skipping.")
            continue
        vals_full = df[metric_key].to_numpy(dtype=np.float64)
        base_mask = np.isfinite(vals_full) & (vals_full != 0.0)
        flat = flat_mask & base_mask
        rest = (~flat_mask) & base_mask
        vals_flat = vals_full[flat]
        vals_rest = vals_full[rest]
        w_dist = dist_weights_full[flat] if dist_weights_full is not None else None
        results[label] = (vals_flat, w_dist, vals_rest)
    return results


def _default_dataset_path(patch_size_m: float | None) -> Path:
    size = DEFAULT_PATCH_SIZE_M if patch_size_m is None else patch_size_m
    label = format_patch_size_label(size)
    return Path(get_paths()["DATASETS"]) / f"patches_{label}m.h5"


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot cost-of-transport distributions on near-flat patches.")
    ap.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to patch HDF5 dataset (default: DATASETS/patches_<patch-size>m.h5).",
    )
    ap.add_argument(
        "--patch-size",
        type=float,
        default=None,
        help=f"Patch size to build default dataset path (meters, default: {DEFAULT_PATCH_SIZE_M}).",
    )
    ap.add_argument(
        "--missions",
        nargs="*",
        default=None,
        help="Optional mission ids/displays to include (default: all missions in the dataset).",
    )
    ap.add_argument(
        "--pitch-limit",
        type=float,
        default=DEFAULT_PITCH_LIMIT_DEG,
        help="Absolute pitch [deg] defining 'flat' (default: 1.0).",
    )
    ap.add_argument(
        "--slope-limit",
        type=float,
        default=DEFAULT_SLOPE_LIMIT_DEG,
        help="Mean slope magnitude [deg] defining 'flat' (default: 1.0).",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory for saved figures (default: reports/zz_incline_patch_analysis).",
    )
    args = ap.parse_args()

    dataset_path = args.dataset if args.dataset is not None else _default_dataset_path(args.patch_size)
    patch_groups = _load_patch_groups(dataset_path, args.missions)
    if not patch_groups:
        raise SystemExit("No missions found in dataset (after filtering).")

    prepared_groups = [(name, display, _prepare_patch_df(df)) for name, display, df in patch_groups]
    metric_labels = [label for _, label in METRICS_TO_PLOT]

    flatness_modes = [
        ("pitch", "pitch_deg", True, float(args.pitch_limit), f"|pitch|<= {args.pitch_limit}°"),
        ("slope", "slope_mag_mean_deg", False, float(args.slope_limit), f"mean slope magnitude <= {args.slope_limit}°"),
    ]

    for mode_name, column, use_abs, limit_deg, desc in flatness_modes:
        aggregated: dict[str, list[tuple[np.ndarray, np.ndarray | None]]] = {m: [] for m in metric_labels}
        aggregated_rest: dict[str, list[np.ndarray]] = {m: [] for m in metric_labels}
        per_mission_samples: dict[str, list[tuple[str, np.ndarray]]] = {m: [] for m in metric_labels}

        for _, display, df in prepared_groups:
            flat_mask = _build_flat_mask(df, column, limit_deg, use_abs)
            if not np.any(flat_mask):
                print(f"[warn] No flat patches for {display} ({desc}).")
            mission_results = _collect_patch_values(df, flat_mask)
            for metric, (vals_flat, dist_w, vals_rest) in mission_results.items():
                aggregated[metric].append((vals_flat, dist_w))
                aggregated_rest[metric].append(vals_rest)
                per_mission_samples[metric].append((display, vals_flat))
                out_name = f"{_slugify(display)}_{metric}_flat_{mode_name}{limit_deg:.1f}.png"
                title = f"{display} — {metric} ({desc}, patches)"
                _plot_metric_distribution(metric, vals_flat, dist_w, title, args.output_dir / out_name)

        for metric, samples in aggregated.items():
            all_vals, all_weights = _stack_samples(samples)
            if all_vals.size == 0:
                print(f"[warn] No aggregate data for {metric} ({mode_name}); skipping.")
                continue
            out_name = f"ALL_{metric}_flat_{mode_name}{limit_deg:.1f}.png"
            title = f"Aggregate flat patches — {metric} ({desc})"
            _plot_metric_distribution(metric, all_vals, all_weights, title, args.output_dir / out_name)

        for metric, samples in per_mission_samples.items():
            if not samples:
                continue
            out_name = f"COMPARE_MISSIONS_{metric}_flat_{mode_name}{limit_deg:.1f}.png"
            _plot_box_by_mission(metric, samples, args.output_dir / out_name)

        for metric in metric_labels:
            flat_vals, _ = _stack_samples(aggregated[metric])
            rest_vals = _stack_arrays(aggregated_rest[metric])
            if flat_vals.size == 0 or rest_vals.size == 0:
                print(f"[warn] Missing flat/rest data for {metric} ({mode_name}); skipping flat-vs-rest plot.")
                continue
            out_name = f"ALL_{metric}_flat_vs_rest_{mode_name}{limit_deg:.1f}.png"
            _plot_flat_vs_rest(metric, flat_vals, rest_vals, desc, args.output_dir / out_name)


if __name__ == "__main__":
    main()
