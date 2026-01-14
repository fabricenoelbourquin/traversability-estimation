#!/usr/bin/env python3
"""
Plot patch mean robot pitch (deg) vs patch cot_patch from the
HDF5 dataset produced by build_patch_dataset.py.
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
DEFAULT_REPORT_DIR = Path(get_paths()["REPO_ROOT"]) / "reports" / "zz_patch_analysis_robot_data" / "all"

PITCH_COL = "pitch_deg_mean"
COT_COL = "cot_patch"
COT_P95_COL = "cot_patch_p95"
OUTLIER_PCT_LOW = 2.0
OUTLIER_PCT_HIGH = 98.0


def _patch_label(patch_size_m: float | None) -> str:
    size = DEFAULT_PATCH_SIZE_M if patch_size_m is None else patch_size_m
    label_num = f"{size:.3f}".rstrip("0").rstrip(".")
    return f"{label_num}m"


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


def _resolve_dataset_path(dataset_arg: Path | None, patch_size_m: float | None) -> Path:
    if dataset_arg is None:
        return _default_dataset_path(patch_size_m)
    dataset_path = Path(dataset_arg)
    if dataset_path.is_absolute():
        return dataset_path
    if dataset_path.parent == Path("."):
        if dataset_path.suffix == "":
            dataset_path = dataset_path.with_suffix(".h5")
        return Path(get_paths()["DATASETS"]) / dataset_path.name
    return dataset_path


def _load_patch_groups(h5_path: Path, missions: Sequence[str] | None) -> list[pd.DataFrame]:
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise SystemExit("h5py is required to read the patch dataset (pip install h5py).") from exc

    if not h5_path.exists():
        raise SystemExit(f"Dataset not found: {h5_path}")

    requested = set(missions) if missions else None
    dfs: list[pd.DataFrame] = []
    with h5py.File(h5_path, "r") as f:
        for grp_name in sorted(f.keys()):
            grp = f[grp_name]
            attrs = {k: _decode_attr_val(v) for k, v in grp.attrs.items()}
            display = str(attrs.get("mission_display") or grp_name)
            if requested and grp_name not in requested and display not in requested:
                continue
            if "patches" not in grp:
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
            df["mission_display"] = display
            dfs.append(df)
    if requested:
        missing = requested - {d["mission_display"].iloc[0] for d in dfs} if dfs else requested
        if missing:
            print(f"[warn] Requested missions not found in dataset: {sorted(missing)}")
    return dfs


def _finite(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _prepare(df: pd.DataFrame, cot_col: str) -> tuple[np.ndarray, np.ndarray]:
    if PITCH_COL not in df.columns or cot_col not in df.columns:
        raise SystemExit(f"Required columns missing: need '{PITCH_COL}' and '{cot_col}'.")
    pitch = df[PITCH_COL].to_numpy(dtype=np.float64)
    cot = df[cot_col].to_numpy(dtype=np.float64)
    mask = np.isfinite(pitch) & np.isfinite(cot)
    return pitch[mask], cot[mask]


def _fit_poly(x: np.ndarray, y: np.ndarray, deg: int) -> np.ndarray | None:
    if x.size < deg + 1 or y.size < deg + 1:
        return None
    coeffs = np.polyfit(x, y, deg)
    return coeffs.astype(float)


def _outlier_mask(
    x: np.ndarray, y: np.ndarray, pct_low: float = OUTLIER_PCT_LOW, pct_high: float = OUTLIER_PCT_HIGH
) -> tuple[np.ndarray, dict[str, float | None]]:
    if x.size == 0 or y.size == 0:
        return np.ones_like(x, dtype=bool), {
            "pct_low": pct_low,
            "pct_high": pct_high,
            "x_low": None,
            "x_high": None,
            "y_low": None,
            "y_high": None,
        }
    x_low, x_high = np.percentile(x, [pct_low, pct_high])
    y_low, y_high = np.percentile(y, [pct_low, pct_high])
    mask = (x >= x_low) & (x <= x_high) & (y >= y_low) & (y <= y_high)
    return mask, {
        "pct_low": float(pct_low),
        "pct_high": float(pct_high),
        "x_low": float(x_low),
        "x_high": float(x_high),
        "y_low": float(y_low),
        "y_high": float(y_high),
    }


def _coeffs_payload(coeffs: np.ndarray | None) -> dict[str, float | list[float] | int | str] | None:
    if coeffs is None:
        return None
    coeff_list = [float(c) for c in coeffs.tolist()]
    payload: dict[str, float | list[float] | int | str] = {
        "degree": int(len(coeff_list) - 1),
        "coefficients": coeff_list,
        "order": "highest_degree_first",
    }
    if len(coeff_list) == 3:
        payload["quadratic"] = {"a": coeff_list[0], "b": coeff_list[1], "c": coeff_list[2]}
    return payload


def _residual_metrics(x: np.ndarray, y: np.ndarray, coeffs: np.ndarray | None) -> dict[str, float | int | None]:
    metrics: dict[str, float | int | None] = {
        "count": int(x.size),
        "rmse": None,
        "mae": None,
        "median_abs_error": None,
        "bias": None,
        "residual_variance": None,
        "residual_std": None,
        "r2": None,
        "y_mean": None,
        "y_variance": None,
        "y_std": None,
    }
    if coeffs is None or x.size == 0:
        return metrics
    preds = np.polyval(coeffs, x)
    residuals = y - preds
    metrics.update(
        {
            "rmse": float(np.sqrt(np.mean(residuals**2))),
            "mae": float(np.mean(np.abs(residuals))),
            "median_abs_error": float(np.median(np.abs(residuals))),
            "bias": float(np.mean(residuals)),
            "residual_variance": float(np.var(residuals)),
            "residual_std": float(np.std(residuals)),
            "y_mean": float(np.mean(y)),
            "y_variance": float(np.var(y)),
            "y_std": float(np.std(y)),
        }
    )
    ss_res = float(np.sum(residuals**2))
    y_mean = float(metrics["y_mean"]) if metrics["y_mean"] is not None else 0.0
    ss_tot = float(np.sum((y - y_mean) ** 2))
    metrics["r2"] = None if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return metrics


def _write_metrics(out_path: Path, payload: dict) -> Path:
    metrics_path = out_path.with_name(f"{out_path.stem}_fit_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[ok] wrote {metrics_path}")
    return metrics_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot mean robot pitch (deg) vs cot_patch for patches.")
    ap.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=(
            "Path to patch HDF5 dataset. If a bare filename is provided, it is"
            " resolved under DATASETS (default: DATASETS/patches_<patch-size>m.h5)."
        ),
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
        "--output",
        type=Path,
        default=None,
        help="Output path for figure (default: reports/zz_patch_analysis_robot_data/patch_pitch_vs_cot.png).",
    )
    ap.add_argument(
        "--gridsize",
        type=int,
        default=50,
        help="Hexbin grid size (default: 50).",
    )
    ap.add_argument(
        "--y-range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=None,
        help="Optional y-axis range (cot_patch). When set, saves both unrestricted and restricted plots.",
    )
    ap.add_argument(
        "--p95",
        action="store_true",
        help="Use cot_patch_p95 instead of cot_patch.",
    )
    args = ap.parse_args()

    dataset_path = _resolve_dataset_path(args.dataset, args.patch_size)
    patch_label = _patch_label(args.patch_size)
    cot_col = COT_P95_COL if args.p95 else COT_COL
    default_name = "patch_pitch_vs_cot_p95.png" if args.p95 else "patch_pitch_vs_cot.png"
    default_out = DEFAULT_REPORT_DIR / patch_label / default_name
    base_out = args.output if args.output is not None else default_out
    dfs = _load_patch_groups(dataset_path, args.missions)
    if not dfs:
        raise SystemExit("No missions found in dataset (after filtering).")

    pitch_all: list[np.ndarray] = []
    cot_all: list[np.ndarray] = []
    for df in dfs:
        p, c = _prepare(df, cot_col)
        pitch_all.append(p)
        cot_all.append(c)

    pitch_deg = _finite(np.concatenate(pitch_all)) if pitch_all else np.array([])
    cot_vals = _finite(np.concatenate(cot_all)) if cot_all else np.array([])
    if pitch_deg.size == 0 or cot_vals.size == 0:
        raise SystemExit(f"No finite pitch/{cot_col} data to plot.")

    fit_all = _fit_poly(pitch_deg, cot_vals, deg=2)
    outlier_mask, outlier_bounds = _outlier_mask(pitch_deg, cot_vals)
    pitch_nr, cot_nr = pitch_deg[outlier_mask], cot_vals[outlier_mask]
    fit_no_outliers = _fit_poly(pitch_nr, cot_nr, deg=2)
    x_plot = np.linspace(np.min(pitch_deg), np.max(pitch_deg), 200) if pitch_deg.size else np.array([])

    metrics_payload = {
        "dataset_path": str(dataset_path),
        "missions": args.missions,
        "patch_size_m": float(args.patch_size) if args.patch_size is not None else DEFAULT_PATCH_SIZE_M,
        "cot_col": cot_col,
        "points": {
            "total": int(pitch_deg.size),
            "inliers": int(pitch_nr.size),
            "outliers": int(pitch_deg.size - pitch_nr.size),
        },
        "outlier_filter": outlier_bounds,
        "fit_no_outliers": _coeffs_payload(fit_no_outliers),
        "fit_all": _coeffs_payload(fit_all),
        "metrics_vs_no_outliers_fit": {
            "all_points": _residual_metrics(pitch_deg, cot_vals, fit_no_outliers),
            "inlier_points": _residual_metrics(pitch_nr, cot_nr, fit_no_outliers),
        },
    }

    def _plot(y_range: tuple[float, float] | None, suffix: str) -> Path:
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        hb = ax.hexbin(
            pitch_deg,
            cot_vals,
            gridsize=args.gridsize,
            cmap="viridis",
            mincnt=1,
            linewidths=0.0,
        )
        if y_range is not None:
            ax.set_ylim(y_range)
        ax.set_xlabel("pitch_deg_mean [deg]")
        ax.set_ylabel(cot_col)
        title_suffix = "" if y_range is None else f" (y∈[{y_range[0]}, {y_range[1]}])"
        ax.set_title(f"Patch mean robot pitch vs {cot_col}{title_suffix}")
        if x_plot.size and fit_all is not None:
            ax.plot(x_plot, np.polyval(fit_all, x_plot), color="tab:red", lw=1.4, label="quad fit (all)")
        if x_plot.size and fit_no_outliers is not None:
            ax.plot(x_plot, np.polyval(fit_no_outliers, x_plot), color="tab:orange", lw=1.4, linestyle="--", label="quad fit (no outliers)")
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("patch count")
        ax.grid(alpha=0.25)
        fig.tight_layout()

        out_path = base_out if suffix == "" else base_out.with_name(f"{base_out.stem}{suffix}{base_out.suffix}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"[ok] wrote {out_path}")
        return out_path

    out_path = _plot(None, "")
    _write_metrics(out_path, metrics_payload)
    if args.y_range is not None:
        _plot((args.y_range[0], args.y_range[1]), "_restricted")


if __name__ == "__main__":
    main()
