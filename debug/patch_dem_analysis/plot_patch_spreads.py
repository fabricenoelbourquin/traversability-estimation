#!/usr/bin/env python3
"""
Plot spreads of patch-level slope magnitudes and cost-of-transport metrics
from the HDF5 dataset produced by build_patch_dataset.py.

Two figures (slope + cost_of_transport), each with two subplots:
  - Left: min/mean/max overlaid with transparency (range optionally clamped).
  - Right: spread histogram (max - min) as in the earlier bottom-right plot.
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


def _resolve_repo_root(file_path: Path) -> Path:
    for parent in file_path.parents:
        if (parent / "src").exists():
            return parent
    raise SystemExit("Could not find repository root (missing 'src' directory).")


REPO_ROOT = _resolve_repo_root(THIS_FILE)
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.paths import get_paths


DEFAULT_PATCH_SIZE_M: float = 5.0
DEFAULT_REPORT_DIR = Path(get_paths()["REPO_ROOT"]) / "reports" / "zz_incline_patch_analysis"

SLOPE_COLS = ("slope_mag_min", "slope_mag_mean", "slope_mag_max")
COT_COLS = (
    "metric_cost_of_transport_min",
    "metric_cost_of_transport_mean",
    "metric_cost_of_transport_max",
)

def _decode_attr_val(val):
    if isinstance(val, (bytes, bytearray)):
        try:
            return val.decode("utf-8")
        except Exception:
            return val
    return val


def _default_dataset_path(patch_size_m: float | None) -> Path:
    size = DEFAULT_PATCH_SIZE_M if patch_size_m is None else patch_size_m
    label = f"{size:.3f}".rstrip("0").rstrip(".")
    return Path(get_paths()["DATASETS"]) / f"patches_{label}m.h5"


def _patch_label(patch_size_m: float | None) -> str:
    size = DEFAULT_PATCH_SIZE_M if patch_size_m is None else patch_size_m
    label_num = f"{size:.3f}".rstrip("0").rstrip(".")
    return f"{label_num}m"


def _parse_patch_size(arg: str | None) -> float | None:
    if arg is None:
        return None
    # accept "5", "5.0", "5m"
    txt = str(arg).lower().rstrip()
    if txt.endswith("m"):
        txt = txt[:-1]
    try:
        return float(txt)
    except Exception:
        return None


def _resolve_dataset_and_size(dataset_arg: str | None) -> tuple[Path, float | None]:
    if dataset_arg:
        p = Path(dataset_arg)
        if p.exists():
            # try to infer size from filename, else None
            size = None
            stem = p.stem
            for token in stem.split("_"):
                if token.endswith("m"):
                    try:
                        size = float(token[:-1])
                        break
                    except Exception:
                        continue
            return p, size
        size = _parse_patch_size(dataset_arg)
        if size is not None:
            return _default_dataset_path(size), size
    # fallback to default
    return _default_dataset_path(DEFAULT_PATCH_SIZE_M), DEFAULT_PATCH_SIZE_M


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


def _finite(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _rad_slope_to_deg(vals: np.ndarray) -> np.ndarray:
    return np.rad2deg(np.arctan(vals.astype(np.float64)))


def _collect_slope_data(groups: list[tuple[str, str, pd.DataFrame]]) -> dict[str, np.ndarray]:
    data: dict[str, list[np.ndarray]] = {k: [] for k in SLOPE_COLS}
    spread_list: list[np.ndarray] = []
    for _, _, df in groups:
        if not all(col in df.columns for col in SLOPE_COLS):
            continue
        s_min = _rad_slope_to_deg(df["slope_mag_min"].to_numpy())
        s_mean = _rad_slope_to_deg(df["slope_mag_mean"].to_numpy())
        s_max = _rad_slope_to_deg(df["slope_mag_max"].to_numpy())
        spread = s_max - s_min
        data["slope_mag_min"].append(_finite(s_min))
        data["slope_mag_mean"].append(_finite(s_mean))
        data["slope_mag_max"].append(_finite(s_max))
        spread_list.append(_finite(spread))
    out: dict[str, np.ndarray] = {}
    for k, parts in data.items():
        out[k] = np.concatenate(parts) if parts else np.array([])
    out["slope_spread_deg"] = np.concatenate(spread_list) if spread_list else np.array([])
    return out


def _collect_cot_data(groups: list[tuple[str, str, pd.DataFrame]]) -> dict[str, np.ndarray]:
    data: dict[str, list[np.ndarray]] = {k: [] for k in COT_COLS}
    spread_list: list[np.ndarray] = []
    for _, _, df in groups:
        if not any(col in df.columns for col in COT_COLS):
            continue
        cols_available = {c for c in COT_COLS if c in df.columns}
        if not {"metric_cost_of_transport_min", "metric_cost_of_transport_max"}.issubset(cols_available):
            continue
        c_min = df["metric_cost_of_transport_min"].to_numpy(dtype=np.float64)
        c_mean = df["metric_cost_of_transport_mean"].to_numpy(dtype=np.float64) if "metric_cost_of_transport_mean" in df.columns else np.array([], dtype=np.float64)
        c_max = df["metric_cost_of_transport_max"].to_numpy(dtype=np.float64)
        spread = c_max - c_min
        data["metric_cost_of_transport_min"].append(_finite(c_min))
        if c_mean.size:
            data["metric_cost_of_transport_mean"].append(_finite(c_mean))
        data["metric_cost_of_transport_max"].append(_finite(c_max))
        spread_list.append(_finite(spread))
    out: dict[str, np.ndarray] = {}
    for k, parts in data.items():
        out[k] = np.concatenate(parts) if parts else np.array([])
    out["cot_spread"] = np.concatenate(spread_list) if spread_list else np.array([])
    return out


def _plot_overlaid(ax, values: dict[str, np.ndarray], keys: tuple[str, str, str], title: str, xlabel: str) -> None:
    colors = ["tab:blue", "tab:orange", "tab:green"]
    labels = ["min", "max", "mean"]
    # Draw mean last so it appears on top
    order = [0, 1, 2]
    for idx in order:
        key = keys[idx]
        arr = _finite(values.get(key, np.array([])))
        if not arr.size:
            continue
        alpha = 0.55 if idx < 2 else 0.8
        ax.hist(arr, bins=50, color=colors[idx], alpha=alpha, label=f"{labels[idx]} ({key})")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("patch count")
    ax.grid(alpha=0.2)
    ax.legend()


def _plot_spread(ax, spread_values: np.ndarray, title: str, xlabel: str) -> None:
    arr = _finite(spread_values)
    if arr.size:
        ax.hist(arr, bins=50, color="0.5", alpha=0.85)
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("patch count")
    ax.grid(alpha=0.2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot spreads of patch-level slopes and cost-of-transport.")
    ap.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Patch size (e.g., '5' or '5m') or explicit dataset path. Default uses PATCH_SIZE constant.",
    )
    ap.add_argument(
        "--missions",
        nargs="*",
        default=None,
        help="Optional mission ids/displays to include (default: all missions in the dataset).",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory for saved figures (default: reports/zz_incline_patch_analysis).",
    )
    ap.add_argument(
        "--slope-range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=None,
        help="Optional x-axis range for slope overlay (degrees).",
    )
    ap.add_argument(
        "--cot-range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=None,
        help="Optional x-axis range for cost_of_transport overlay.",
    )
    args = ap.parse_args()

    dataset_path, patch_size_used = _resolve_dataset_and_size(args.dataset)
    patch_label = _patch_label(patch_size_used)
    default_out_dir = DEFAULT_REPORT_DIR / patch_label
    out_dir = default_out_dir if args.output_dir == DEFAULT_REPORT_DIR else args.output_dir
    patch_groups = _load_patch_groups(dataset_path, args.missions)
    if not patch_groups:
        raise SystemExit("No missions found in dataset (after filtering).")

    slope_data = _collect_slope_data(patch_groups)
    cot_data = _collect_cot_data(patch_groups)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Slope figure
    fig_s, axes_s = plt.subplots(1, 2, figsize=(12, 5))
    fig_s.suptitle("Patch slope magnitudes and spread", y=0.995, fontsize=11)
    _plot_overlaid(
        axes_s[0],
        slope_data,
        ("slope_mag_min", "slope_mag_max", "slope_mag_mean"),
        "Slope magnitude per patch (deg)",
        "slope magnitude [deg]",
    )
    if args.slope_range is not None:
        axes_s[0].set_xlim(args.slope_range)
    _plot_spread(
        axes_s[1],
        slope_data.get("slope_spread_deg", np.array([])),
        "Slope spread (max - min)",
        "spread [deg]",
    )
    fig_s.tight_layout(rect=[0, 0, 1, 0.98])
    out_s = out_dir / "ALL_patch_slope_spreads.png"
    fig_s.savefig(out_s, dpi=200)
    print(f"[ok] wrote {out_s}")
    plt.close(fig_s)

    # CoT figure
    fig_c, axes_c = plt.subplots(1, 2, figsize=(12, 5))
    fig_c.suptitle("Patch cost of transport and spread", y=0.995, fontsize=11)
    _plot_overlaid(
        axes_c[0],
        cot_data,
        ("metric_cost_of_transport_min", "metric_cost_of_transport_max", "metric_cost_of_transport_mean"),
        "Cost of transport per patch",
        "cost_of_transport",
    )
    if args.cot_range is not None:
        axes_c[0].set_xlim(args.cot_range)
    _plot_spread(
        axes_c[1],
        cot_data.get("cot_spread", np.array([])),
        "CoT spread (max - min)",
        "spread",
    )
    fig_c.tight_layout(rect=[0, 0, 1, 0.98])
    out_c = out_dir / "ALL_patch_cot_spreads.png"
    fig_c.savefig(out_c, dpi=200)
    print(f"[ok] wrote {out_c}")
    plt.close(fig_c)


if __name__ == "__main__":
    main()
