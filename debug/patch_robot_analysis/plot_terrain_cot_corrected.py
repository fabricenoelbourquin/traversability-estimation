#!/usr/bin/env python3
"""
Plot pitch-corrected COT (p95) for terrain categories using patch datasets.

Uses a quadratic fit (without outliers) of robot pitch vs cot_patch_p95 and
removes the fitted pitch influence before aggregating by terrain.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
DEFAULT_REPORT_DIR = Path(get_paths()["REPO_ROOT"]) / "reports" / "zz_patch_analysis_robot_data" / "terrain_cot"

PITCH_COL = "pitch_deg_mean"
COT_P95_COL = "cot_patch_p95"
MISSION_DISPLAY_COL = "mission_display"
MISSION_ID_COL = "mission_id"
TIME_COL_CANDIDATES = ("center_t", "t_start", "t_end")


@dataclass(frozen=True)
class MissionSpec:
    mission: str
    ranges_s: tuple[tuple[float, float | None], ...] | None = None


TERRAIN_MISSIONS: dict[str, list[MissionSpec]] = {
    "asphalt": [
        MissionSpec("ETH-1"),
        MissionSpec("GRI-1", ((0.0, 210.0), (250.0, None))),
        MissionSpec("LEE-1"),
        MissionSpec("LEICA-1"),
        MissionSpec("KÄB-3", ((355.0, 425.0),)),
        MissionSpec("SRB-1"),
        MissionSpec("SRB-2"),
        MissionSpec("SRB-3"),
        MissionSpec("ARC-5", ((0.0, 170.0), (215.0, None))),
    ],
    "gravel": [
        MissionSpec("KÄB-3", ((10.0, 345.0),)),
        MissionSpec("TRIM-1", ((0.0, 350.0),)),
        MissionSpec("ARC-5", ((175.0, 200.0),)),
        MissionSpec("ALB-2", ((0.0, 40.0),)),
        MissionSpec("HÖB-1", ((180.0, 220.0)))
    ],
    "forest": [
        MissionSpec("ALB-1"),
        MissionSpec("ALB-2", ((50.0, 440),)),
        MissionSpec("ALB-3", ((0.0, 160.0),)),
        MissionSpec("LMB-2"),
        MissionSpec("HÖB-1", ((0.0, 180.0), (230.0, None))),
        MissionSpec("HÖB-2"),
        MissionSpec("KÄB-1"),
        MissionSpec("KÄB-2"),
    ],
    "snow": [MissionSpec("SNOW-3")],
    "ice": [MissionSpec("ICE-1")],
    "hiking trail": [
        MissionSpec("CYN-1", ((90.0, 300.0),)),
        MissionSpec("CYN-2"),
        MissionSpec("PIL-2"),
        MissionSpec("ROOT-1"),
        MissionSpec("LMB-1"),
    ],
}


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
    return Path(get_paths()["DATASETS"]) / f"all_patches_{label}m.h5"


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


def _load_patch_groups(h5_path: Path, missions: list[str]) -> list[pd.DataFrame]:
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
            df[MISSION_DISPLAY_COL] = display
            df[MISSION_ID_COL] = grp_name
            dfs.append(df)
    if requested:
        present = set()
        for df in dfs:
            if not df.empty:
                present.add(str(df[MISSION_DISPLAY_COL].iloc[0]))
                present.add(str(df[MISSION_ID_COL].iloc[0]))
        missing = requested - present
        if missing:
            print(f"[warn] Requested missions not found in dataset: {sorted(missing)}")
    return dfs


def _mission_mask(df: pd.DataFrame, mission: str) -> np.ndarray:
    mask = df[MISSION_DISPLAY_COL] == mission
    if MISSION_ID_COL in df.columns:
        mask |= df[MISSION_ID_COL] == mission
    return mask.to_numpy(dtype=bool)


def _select_time_col(df: pd.DataFrame) -> str:
    for col in TIME_COL_CANDIDATES:
        if col in df.columns:
            return col
    raise SystemExit(f"None of the time columns {TIME_COL_CANDIDATES} found in dataset.")


def _apply_time_ranges(df: pd.DataFrame, time_col: str, ranges: tuple[tuple[float, float | None], ...]) -> pd.DataFrame:
    times = df[time_col].to_numpy(dtype=np.float64)
    finite = np.isfinite(times)
    if not finite.any():
        return df.iloc[0:0]
    # Interpret time windows as seconds since mission start (min time in dataset subset).
    base = float(np.nanmin(times[finite]))
    t_rel = times - base
    mask = np.zeros_like(times, dtype=bool)
    for start, end in ranges:
        start_val = float(start)
        if end is None:
            mask |= t_rel >= start_val
        else:
            end_val = float(end)
            mask |= (t_rel >= start_val) & (t_rel <= end_val)
    mask &= finite
    return df.loc[mask].copy()


def _prepare_pitch_cot(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if PITCH_COL not in df.columns or COT_P95_COL not in df.columns:
        raise SystemExit(f"Required columns missing: need '{PITCH_COL}' and '{COT_P95_COL}'.")
    pitch = df[PITCH_COL].to_numpy(dtype=np.float64)
    cot = df[COT_P95_COL].to_numpy(dtype=np.float64)
    mask = np.isfinite(pitch) & np.isfinite(cot)
    return pitch[mask], cot[mask]


def _remove_outliers(x: np.ndarray, y: np.ndarray, pct_low: float = 2.0, pct_high: float = 98.0) -> tuple[np.ndarray, np.ndarray]:
    if x.size == 0 or y.size == 0:
        return x, y
    x_low, x_high = np.percentile(x, [pct_low, pct_high])
    y_low, y_high = np.percentile(y, [pct_low, pct_high])
    mask = (x >= x_low) & (x <= x_high) & (y >= y_low) & (y <= y_high)
    return x[mask], y[mask]


def _fit_quad_no_outliers(pitch: np.ndarray, cot: np.ndarray) -> np.ndarray:
    if pitch.size < 3 or cot.size < 3:
        raise SystemExit("Not enough samples to fit quadratic pitch correction.")
    pitch_nr, cot_nr = _remove_outliers(pitch, cot)
    if pitch_nr.size < 3 or cot_nr.size < 3:
        raise SystemExit("Not enough samples after outlier removal to fit quadratic pitch correction.")
    return np.polyfit(pitch_nr, cot_nr, deg=2).astype(float)


def _pitch_correct(pitch: np.ndarray, cot: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    baseline = float(np.polyval(coeffs, 0.0))
    predicted = np.polyval(coeffs, pitch)
    return cot - (predicted - baseline)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot pitch-corrected COT (p95) by terrain category.")
    ap.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=(
            "Path to patch HDF5 dataset. If a bare filename is provided, it is"
            " resolved under DATASETS (default: DATASETS/all_patches_<patch-size>m.h5)."
        ),
    )
    ap.add_argument(
        "--patch-size",
        type=float,
        default=None,
        help=f"Patch size to build default dataset path (meters, default: {DEFAULT_PATCH_SIZE_M}).",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for figure (default: reports/zz_patch_analysis_robot_data/terrain_cot/<patch>/terrain_cot_corrected.png).",
    )
    args = ap.parse_args()

    dataset_path = _resolve_dataset_path(args.dataset, args.patch_size)
    patch_label = _patch_label(args.patch_size)
    default_out = DEFAULT_REPORT_DIR / patch_label / "terrain_cot_corrected.png"

    mission_names = sorted({spec.mission for specs in TERRAIN_MISSIONS.values() for spec in specs})
    dfs = _load_patch_groups(dataset_path, mission_names)
    if not dfs:
        raise SystemExit("No missions found in dataset (after filtering).")

    df_all = pd.concat(dfs, ignore_index=True)
    need_time_ranges = any(spec.ranges_s for specs in TERRAIN_MISSIONS.values() for spec in specs)
    time_col = _select_time_col(df_all) if need_time_ranges else ""

    terrain_frames: dict[str, list[pd.DataFrame]] = {}
    selected_frames: list[pd.DataFrame] = []
    for terrain, specs in TERRAIN_MISSIONS.items():
        frames: list[pd.DataFrame] = []
        for spec in specs:
            df_m = df_all[_mission_mask(df_all, spec.mission)]
            if df_m.empty:
                print(f"[warn] Mission '{spec.mission}' not found for terrain '{terrain}'.")
                continue
            if spec.ranges_s:
                df_m = _apply_time_ranges(df_m, time_col, spec.ranges_s)
            if df_m.empty:
                print(f"[warn] Mission '{spec.mission}' empty after time filtering for terrain '{terrain}'.")
                continue
            frames.append(df_m)
            selected_frames.append(df_m)
        terrain_frames[terrain] = frames

    if not selected_frames:
        raise SystemExit("No patches found after applying mission/time filters.")

    df_selected = pd.concat(selected_frames, ignore_index=True)
    pitch_all, cot_all = _prepare_pitch_cot(df_selected)
    if pitch_all.size == 0 or cot_all.size == 0:
        raise SystemExit("No finite pitch/cot_patch_p95 data for correction fit.")
    coeffs = _fit_quad_no_outliers(pitch_all, cot_all)

    print(f"[info] Dataset: {dataset_path}")
    print(f"[info] Quad fit (no outliers) coeffs: {coeffs}")

    stats_rows = []
    for terrain in TERRAIN_MISSIONS.keys():
        frames = terrain_frames.get(terrain, [])
        if not frames:
            stats_rows.append({"terrain": terrain, "n": 0, "mean_cot_corr": np.nan, "std_cot_corr": np.nan})
            continue
        df_terrain = pd.concat(frames, ignore_index=True)
        pitch, cot = _prepare_pitch_cot(df_terrain)
        if pitch.size == 0:
            stats_rows.append({"terrain": terrain, "n": 0, "mean_cot_corr": np.nan, "std_cot_corr": np.nan})
            continue
        corr = _pitch_correct(pitch, cot, coeffs)
        n = int(corr.size)
        mean = float(np.nanmean(corr)) if n else np.nan
        std = float(np.nanstd(corr, ddof=1)) if n > 1 else np.nan
        stats_rows.append({"terrain": terrain, "n": n, "mean_cot_corr": mean, "std_cot_corr": std})

    stats_df = pd.DataFrame(stats_rows)
    for _, row in stats_df.iterrows():
        print(f"[info] {row['terrain']}: n={int(row['n'])} mean={row['mean_cot_corr']:.4g} std={row['std_cot_corr']:.4g}")

    plot_df = stats_df[stats_df["n"] > 0].copy()
    if plot_df.empty:
        raise SystemExit("No terrain categories with data to plot.")

    labels = [f"{t}\n(n={int(n)})" for t, n in zip(plot_df["terrain"], plot_df["n"])]
    means = plot_df["mean_cot_corr"].to_numpy(dtype=np.float64)
    stds = plot_df["std_cot_corr"].to_numpy(dtype=np.float64)
    stds_plot = np.where(np.isfinite(stds), stds, 0.0)

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(labels), 1)))
    ax.bar(np.arange(len(labels)), means, yerr=stds_plot, capsize=4, color=colors, alpha=0.9)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Pitch-corrected COT (cot_patch_p95)")
    ax.set_title("Pitch-corrected COT by terrain (quad fit, no outliers)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = args.output if args.output is not None else default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    stats_path = out_path.with_name(f"{out_path.stem}_stats.csv")
    stats_df.to_csv(stats_path, index=False)

    print(f"[ok] wrote {out_path}")
    print(f"[ok] wrote {stats_path}")


if __name__ == "__main__":
    main()
