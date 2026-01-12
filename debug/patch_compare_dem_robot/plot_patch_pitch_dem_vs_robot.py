#!/usr/bin/env python3
"""
Compare patch-level robot pitch vs DEM pitch (multi-scale) from the dataset.

Default: use the middle DEM smoothing scale (lower-middle if even count).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

from utils.paths import get_paths  # noqa: E402


DEFAULT_PATCH_SIZE_M: float = 5.0
DEFAULT_REPORT_DIR = Path(get_paths()["REPO_ROOT"]) / "reports" / "zz_compare_dem_robot" / "rigiblick"

PITCH_ROBOT_COL = "pitch_deg_mean"


def _decode_attr_val(val):
    if isinstance(val, (bytes, bytearray)):
        try:
            return val.decode("utf-8")
        except Exception:
            return val
    return val


def _patch_label(patch_size_m: float | None) -> str:
    size = DEFAULT_PATCH_SIZE_M if patch_size_m is None else patch_size_m
    label_num = f"{size:.3f}".rstrip("0").rstrip(".")
    return f"{label_num}m"


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


def _load_patch_groups(h5_path: Path, missions: Sequence[str] | None) -> list[tuple[pd.DataFrame, dict]]:
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise SystemExit("h5py is required to read the patch dataset (pip install h5py).") from exc

    if not h5_path.exists():
        raise SystemExit(f"Dataset not found: {h5_path}")

    requested = set(missions) if missions else None
    dfs: list[tuple[pd.DataFrame, dict]] = []
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
            dfs.append((df, attrs))
    if requested:
        found = {d["mission_display"].iloc[0] for d, _ in dfs} if dfs else set()
        missing = requested - found
        if missing:
            print(f"[warn] Requested missions not found in dataset: {sorted(missing)}")
    return dfs


def _format_scale_label(scale_m: float) -> str:
    return f"{scale_m:.1f}".rstrip("0").rstrip(".").replace(".", "p") + "m"


def _load_scales_from_attrs(attrs: dict) -> list[float]:
    raw = attrs.get("dem_pitch_roll_scales_m")
    if not raw:
        return []
    try:
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        values = json.loads(raw)
        return [float(v) for v in values]
    except Exception:
        return []


def _load_scales_from_metrics() -> list[float]:
    metrics_path = Path(get_paths()["REPO_ROOT"]) / "config" / "metrics.yaml"
    if not metrics_path.exists():
        return []
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise SystemExit("pyyaml is required to read config/metrics.yaml (pip install pyyaml).") from exc
    cfg = yaml.safe_load(metrics_path.read_text()) or {}
    dem_cfg = cfg.get("dem_pitch_roll") or {}
    scales = dem_cfg.get("smooth_scales_m") or []
    out = []
    for v in scales:
        try:
            out.append(float(v))
        except Exception:
            continue
    return out


def _pick_middle_scale(scales_m: list[float]) -> float:
    if not scales_m:
        raise SystemExit("No DEM scales found in dataset attrs or metrics config.")
    idx = (len(scales_m) - 1) // 2
    return float(scales_m[idx])


def _finite(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _stats(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return (np.nan, np.nan, np.nan)
    diff = y[mask] - x[mask]
    rmse = float(np.sqrt(np.nanmean(diff * diff)))
    bias = float(np.nanmean(diff))
    corr = float(np.corrcoef(x[mask], y[mask])[0, 1]) if np.sum(mask) > 2 else np.nan
    return rmse, bias, corr


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare patch pitch (robot) vs DEM pitch.")
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
        "--scale-m",
        type=float,
        default=None,
        help="Override DEM scale in meters (default: middle scale).",
    )
    ap.add_argument(
        "--gridsize",
        type=int,
        default=60,
        help="Hexbin grid size (default: 60).",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for figure (default: reports/zz_compare_dem_robot/<patch-size>/pitch_dem_vs_robot.png).",
    )
    args = ap.parse_args()

    dataset_path = _resolve_dataset_path(args.dataset, args.patch_size)
    patch_label = _patch_label(args.patch_size)
    default_out = DEFAULT_REPORT_DIR / patch_label / "pitch_dem_vs_robot.png"

    groups = _load_patch_groups(dataset_path, args.missions)
    if not groups:
        raise SystemExit("No missions found in dataset (after filtering).")

    scales_m = []
    for _, attrs in groups:
        scales_m = _load_scales_from_attrs(attrs)
        if scales_m:
            break
    if not scales_m:
        scales_m = _load_scales_from_metrics()
    scale_m = float(args.scale_m) if args.scale_m is not None else _pick_middle_scale(scales_m)
    label = _format_scale_label(scale_m)
    pitch_dem_col = f"pitch_dem_{label}_mean"

    pitch_robot_all: list[np.ndarray] = []
    pitch_dem_all: list[np.ndarray] = []
    for df, _ in groups:
        missing = [c for c in (PITCH_ROBOT_COL, pitch_dem_col) if c not in df.columns]
        if missing:
            raise SystemExit(f"Required columns missing: {missing}")
        robot = df[PITCH_ROBOT_COL].to_numpy(dtype=np.float64)
        dem = df[pitch_dem_col].to_numpy(dtype=np.float64)
        mask = np.isfinite(robot) & np.isfinite(dem)
        pitch_robot_all.append(robot[mask])
        pitch_dem_all.append(dem[mask])

    pitch_robot = _finite(np.concatenate(pitch_robot_all)) if pitch_robot_all else np.array([])
    pitch_dem = _finite(np.concatenate(pitch_dem_all)) if pitch_dem_all else np.array([])
    if pitch_robot.size == 0 or pitch_dem.size == 0:
        raise SystemExit("No finite pitch data to plot.")

    rmse, bias, corr = _stats(pitch_robot, pitch_dem)

    out_path = args.output if args.output is not None else default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_scatter, ax_hist) = plt.subplots(1, 2, figsize=(12, 5))
    hb = ax_scatter.hexbin(
        pitch_robot,
        pitch_dem,
        gridsize=args.gridsize,
        cmap="viridis",
        mincnt=1,
        linewidths=0.0,
    )
    lim = max(np.max(np.abs(pitch_robot)), np.max(np.abs(pitch_dem)), 5.0)
    ax_scatter.plot([-lim, lim], [-lim, lim], color="k", lw=1.0, alpha=0.6)
    ax_scatter.set_xlabel("robot pitch mean [deg]")
    ax_scatter.set_ylabel(f"DEM pitch mean ({label}) [deg]")
    ax_scatter.set_title(f"Pitch agreement (rmse={rmse:.2f}°, bias={bias:+.2f}°, corr={corr:.2f})")
    ax_scatter.grid(alpha=0.25)
    cb = fig.colorbar(hb, ax=ax_scatter)
    cb.set_label("patch count")

    diff = pitch_dem - pitch_robot
    ax_hist.hist(diff[np.isfinite(diff)], bins=60, color="tab:blue", alpha=0.8)
    ax_hist.axvline(bias, color="k", linestyle="--", lw=1.0)
    ax_hist.set_xlabel("DEM - robot [deg]")
    ax_hist.set_ylabel("count")
    ax_hist.set_title("Difference histogram")
    ax_hist.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
