#!/usr/bin/env python3
"""
Plot patch mean robot pitch vs cot_patch (or cot_patch_p95) for terrain-defined missions.

Hexbin colors indicate the dominant terrain per bin (majority vote with global-count tie-break).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import unicodedata
from pathlib import Path
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from matplotlib import transforms as mtransforms

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
from utils.missions import MissionSpec  # noqa: E402


DEFAULT_PATCH_SIZE_M: float = 5.0
DEFAULT_REPORT_DIR = Path(get_paths()["REPO_ROOT"]) / "reports" / "zz_patch_analysis_robot_data" / "terrain_cot"
DEFAULT_TERRAIN_CONFIG = REPO_ROOT / "config" / "terrain_missions.yaml"

PITCH_COL = "pitch_deg_mean"
COT_COL = "cot_patch"
COT_P95_COL = "cot_patch_p95"
MISSION_DISPLAY_COL = "mission_display"
MISSION_ID_COL = "mission_id"
TIME_COL_CANDIDATES = ("center_t", "t_start", "t_end")
DEFAULT_COLOR_MIX_MIN = 0.15


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


def _parse_ranges(ranges_raw) -> tuple[tuple[float, float | None], ...] | None:
    if not ranges_raw:
        return None
    if not isinstance(ranges_raw, list):
        raise SystemExit("ranges_s must be a list of [start, end] pairs.")
    parsed: list[tuple[float, float | None]] = []
    for pair in ranges_raw:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise SystemExit("ranges_s entries must be [start, end] pairs.")
        start = float(pair[0])
        end = None if pair[1] is None else float(pair[1])
        parsed.append((start, end))
    return tuple(parsed)


def _load_terrain_missions(config_path: Path) -> dict[str, list[MissionSpec]]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise SystemExit("pyyaml is required to read terrain config (pip install pyyaml).") from exc
    if not config_path.exists():
        raise SystemExit(f"Terrain config not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text()) or {}
    terrains_raw = raw.get("terrains", raw)
    if not isinstance(terrains_raw, dict):
        raise SystemExit("Terrain config must map terrain names to mission lists.")
    out: dict[str, list[MissionSpec]] = {}
    for terrain, entries in terrains_raw.items():
        if entries is None:
            out[str(terrain)] = []
            continue
        if not isinstance(entries, list):
            raise SystemExit(f"Terrain '{terrain}' must map to a list of missions.")
        specs: list[MissionSpec] = []
        for entry in entries:
            if isinstance(entry, str):
                specs.append(MissionSpec(entry))
                continue
            if isinstance(entry, dict):
                mission = entry.get("mission")
                if not mission:
                    raise SystemExit(f"Terrain '{terrain}' has a mission entry without 'mission'.")
                ranges = _parse_ranges(entry.get("ranges_s"))
                specs.append(MissionSpec(str(mission), ranges))
                continue
            raise SystemExit(f"Terrain '{terrain}' mission entries must be strings or dicts.")
        out[str(terrain)] = specs
    return out


def _load_patch_groups(h5_path: Path, missions: Sequence[str]) -> list[pd.DataFrame]:
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
    best_col = None
    best_count = -1
    for col in TIME_COL_CANDIDATES:
        if col not in df.columns:
            continue
        count = int(np.isfinite(df[col].to_numpy(dtype=np.float64)).sum())
        if count > best_count:
            best_count = count
            best_col = col
    if best_col is None or best_count <= 0:
        raise SystemExit(f"None of the time columns {TIME_COL_CANDIDATES} have finite values.")
    return best_col


def _apply_time_ranges(df: pd.DataFrame, time_col: str, ranges: tuple[tuple[float, float | None], ...]) -> pd.DataFrame:
    times = df[time_col].to_numpy(dtype=np.float64)
    finite = np.isfinite(times)
    if not finite.any():
        return df.iloc[0:0]
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


def _prepare_pitch_cot(df: pd.DataFrame, cot_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if PITCH_COL not in df.columns or cot_col not in df.columns:
        raise SystemExit(f"Required columns missing: need '{PITCH_COL}' and '{cot_col}'.")
    pitch = df[PITCH_COL].to_numpy(dtype=np.float64)
    cot = df[cot_col].to_numpy(dtype=np.float64)
    mask = np.isfinite(pitch) & np.isfinite(cot)
    return pitch, cot, mask


def _fit_poly(x: np.ndarray, y: np.ndarray, deg: int) -> np.ndarray | None:
    if x.size < deg + 1 or y.size < deg + 1:
        return None
    coeffs = np.polyfit(x, y, deg)
    return coeffs.astype(float)


def _remove_outliers(x: np.ndarray, y: np.ndarray, pct_low: float = 2.0, pct_high: float = 98.0) -> tuple[np.ndarray, np.ndarray]:
    if x.size == 0 or y.size == 0:
        return x, y
    x_low, x_high = np.percentile(x, [pct_low, pct_high])
    y_low, y_high = np.percentile(y, [pct_low, pct_high])
    mask = (x >= x_low) & (x <= x_high) & (y >= y_low) & (y <= y_high)
    return x[mask], y[mask]


def _terrain_cmap(n: int) -> ListedColormap:
    base = matplotlib.colormaps.get_cmap("tab10" if n <= 10 else "tab20")
    if n <= 0:
        return ListedColormap([], name="terrain")
    denom = max(n - 1, 1)
    colors = [base(i / denom) for i in range(n)]
    return ListedColormap(colors, name="terrain")


def _slugify(name: str) -> str:
    ascii_name = (
        unicodedata.normalize("NFKD", name)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    out = []
    for ch in ascii_name:
        if ch.isalnum():
            out.append(ch)
        elif ch in (" ", "-", "_"):
            out.append("_")
    slug = "".join(out).strip("_")
    return slug or "terrain"


def _compute_hexbin_extent(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    if x.size == 0 or y.size == 0:
        return (0.0, 1.0, 0.0, 1.0)
    xmin, xmax = mtransforms.nonsingular(float(np.min(x)), float(np.max(x)), expander=0.1)
    ymin, ymax = mtransforms.nonsingular(float(np.min(y)), float(np.max(y)), expander=0.1)
    padding = 1.0e-9 * (xmax - xmin)
    xmin -= padding
    xmax += padding
    return (xmin, xmax, ymin, ymax)


def _hexbin_indices(
    x: np.ndarray,
    y: np.ndarray,
    gridsize: int | tuple[int, int],
    extent: tuple[float, float, float, float],
    mincnt: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if np.iterable(gridsize):
        nx, ny = gridsize  # type: ignore[misc]
    else:
        nx = int(gridsize)
        ny = int(nx / math.sqrt(3))
    xmin, xmax, ymin, ymax = extent
    sx = (xmax - xmin) / nx
    sy = (ymax - ymin) / ny

    ix = (x - xmin) / sx
    iy = (y - ymin) / sy
    ix1 = np.round(ix).astype(int)
    iy1 = np.round(iy).astype(int)
    ix2 = np.floor(ix).astype(int)
    iy2 = np.floor(iy).astype(int)

    nx1 = nx + 1
    ny1 = ny + 1
    nx2 = nx
    ny2 = ny
    n_bins = nx1 * ny1 + nx2 * ny2

    i1 = np.where((0 <= ix1) & (ix1 < nx1) & (0 <= iy1) & (iy1 < ny1), ix1 * ny1 + iy1 + 1, 0)
    i2 = np.where((0 <= ix2) & (ix2 < nx2) & (0 <= iy2) & (iy2 < ny2), ix2 * ny2 + iy2 + 1, 0)

    d1 = (ix - ix1) ** 2 + 3.0 * (iy - iy1) ** 2
    d2 = (ix - ix2 - 0.5) ** 2 + 3.0 * (iy - iy2 - 0.5) ** 2
    bdist = d1 < d2

    counts1 = np.bincount(i1[bdist], minlength=1 + nx1 * ny1)[1:]
    counts2 = np.bincount(i2[~bdist], minlength=1 + nx2 * ny2)[1:]
    accum = np.concatenate([counts1, counts2]).astype(float)
    if mincnt is not None:
        accum[accum < mincnt] = np.nan
    good_idxs = ~np.isnan(accum)

    bin_index = np.full(len(x), -1, dtype=int)
    mask1 = bdist & (i1 > 0)
    mask2 = (~bdist) & (i2 > 0)
    bin_index[mask1] = i1[mask1] - 1
    bin_index[mask2] = nx1 * ny1 + (i2[mask2] - 1)

    return bin_index, good_idxs, accum, n_bins


def _mix_weights(counts: np.ndarray, min_mix: float) -> np.ndarray:
    total = float(np.sum(counts))
    if total <= 0.0:
        return np.zeros_like(counts, dtype=np.float64)
    present = counts > 0
    k = int(present.sum())
    weights = counts.astype(np.float64) / total
    if k <= 1 or min_mix <= 0.0:
        weights[~present] = 0.0
        return weights
    floor = min(min_mix, 1.0 / k)
    base = np.full(k, floor, dtype=np.float64)
    remaining = 1.0 - floor * k
    extras = weights[present] - floor
    extras[extras < 0.0] = 0.0
    if remaining > 0.0:
        if extras.sum() > 0.0:
            base += remaining * (extras / extras.sum())
        else:
            base += remaining / k
    mixed = np.zeros_like(weights, dtype=np.float64)
    mixed[present] = base
    return mixed


def _legend_columns(count: int) -> int:
    if count <= 6:
        return 1
    if count <= 12:
        return 2
    return 3


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot mean robot pitch vs cot, colored by terrain (hexbin).")
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
        help="Output path for figure (default: reports/zz_patch_analysis_robot_data/terrain_cot/<patch>/patch_pitch_vs_cot_terrain.png).",
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
    ap.add_argument(
        "--color-mix",
        action="store_true",
        help="Mix terrain colors within each hexbin instead of choosing the dominant terrain.",
    )
    ap.add_argument(
        "--color-mix-min",
        type=float,
        default=DEFAULT_COLOR_MIX_MIN,
        help=f"Minimum mix fraction for present terrains when using --color-mix (default: {DEFAULT_COLOR_MIX_MIN}).",
    )
    ap.add_argument(
        "--terrain-config",
        type=Path,
        default=None,
        help=f"Terrain mission config YAML (default: {DEFAULT_TERRAIN_CONFIG}).",
    )
    ap.add_argument(
        "--add-terrain-plots",
        action="store_true",
        help="Also save per-terrain patch-count hexbins in a subfolder.",
    )
    args = ap.parse_args()

    dataset_path = _resolve_dataset_path(args.dataset, args.patch_size)
    patch_label = _patch_label(args.patch_size)
    cot_col = COT_P95_COL if args.p95 else COT_COL
    default_name = "patch_pitch_vs_cot_p95_terrain.png" if args.p95 else "patch_pitch_vs_cot_terrain.png"
    default_out = DEFAULT_REPORT_DIR / patch_label / default_name
    base_out = args.output if args.output is not None else default_out

    terrain_cfg_path = args.terrain_config or DEFAULT_TERRAIN_CONFIG
    terrain_missions = _load_terrain_missions(terrain_cfg_path)
    if not terrain_missions:
        raise SystemExit(f"No terrain missions found in {terrain_cfg_path}.")

    mission_names = sorted({spec.mission for specs in terrain_missions.values() for spec in specs})
    dfs = _load_patch_groups(dataset_path, mission_names)
    if not dfs:
        raise SystemExit("No missions found in dataset (after filtering).")

    df_all = pd.concat(dfs, ignore_index=True)
    selected_frames: list[pd.DataFrame] = []
    assigned_idx: set[int] = set()
    for terrain, specs in terrain_missions.items():
        for spec in specs:
            df_m = df_all[_mission_mask(df_all, spec.mission)]
            if df_m.empty:
                print(f"[warn] Mission '{spec.mission}' not found for terrain '{terrain}'.")
                continue
            if spec.ranges_s:
                time_col = _select_time_col(df_m)
                df_m = _apply_time_ranges(df_m, time_col, spec.ranges_s)
            if df_m.empty:
                print(f"[warn] Mission '{spec.mission}' empty after time filtering for terrain '{terrain}'.")
                continue
            dup_mask = df_m.index.isin(assigned_idx)
            if dup_mask.any():
                print(f"[warn] Mission '{spec.mission}' has overlapping terrain ranges; skipping {dup_mask.sum()} patches.")
                df_m = df_m.loc[~dup_mask]
            if df_m.empty:
                continue
            assigned_idx.update(df_m.index.astype(int).tolist())
            df_m = df_m.copy()
            df_m["terrain"] = terrain
            selected_frames.append(df_m)

    if not selected_frames:
        raise SystemExit("No patches found after applying terrain filters.")

    df_selected = pd.concat(selected_frames, ignore_index=True)
    terrain_counts = df_selected["terrain"].value_counts().to_dict()
    terrain_order = [t for t in terrain_missions.keys() if terrain_counts.get(t, 0) > 0]
    if not terrain_order:
        raise SystemExit("No terrain categories with data to plot.")
    terrain_to_idx = {t: i for i, t in enumerate(terrain_order)}
    total_counts = np.array([terrain_counts.get(t, 0) for t in terrain_order], dtype=np.int64)
    pitch_raw, cot_raw, base_mask = _prepare_pitch_cot(df_selected, cot_col)
    terrain_idx = df_selected["terrain"].map(terrain_to_idx).to_numpy(dtype=np.float64)
    finite_mask = base_mask & np.isfinite(terrain_idx)
    pitch_all = pitch_raw[finite_mask]
    cot_all = cot_raw[finite_mask]
    terrain_idx = terrain_idx[finite_mask]
    if pitch_all.size == 0 or cot_all.size == 0:
        raise SystemExit("No finite pitch/cot data after terrain filtering.")

    fit_all = _fit_poly(pitch_all, cot_all, deg=2)
    pitch_nr, cot_nr = _remove_outliers(pitch_all, cot_all)
    fit_no_outliers = _fit_poly(pitch_nr, cot_nr, deg=2)
    x_plot = np.linspace(np.min(pitch_all), np.max(pitch_all), 200) if pitch_all.size else np.array([])
    xlim = (float(np.min(pitch_all)), float(np.max(pitch_all))) if pitch_all.size else (0.0, 1.0)
    ylim = (float(np.min(cot_all)), float(np.max(cot_all))) if cot_all.size else (0.0, 1.0)

    def _dominant_terrain(c_vals: np.ndarray) -> float:
        if len(c_vals) == 0:
            return np.nan
        vals = np.asarray(c_vals, dtype=int)
        counts = np.bincount(vals, minlength=len(terrain_order))
        max_count = counts.max() if counts.size else 0
        if max_count == 0:
            return np.nan
        candidates = np.where(counts == max_count)[0]
        if candidates.size == 1:
            return float(candidates[0])
        cand_totals = total_counts[candidates]
        min_total = cand_totals.min()
        winners = candidates[cand_totals == min_total]
        if winners.size == 1:
            return float(winners[0])
        return float(winners.min())

    cmap = _terrain_cmap(len(terrain_order))
    norm = BoundaryNorm(np.arange(-0.5, len(terrain_order) + 0.5, 1.0), cmap.N)

    def _plot(y_range: tuple[float, float] | None, suffix: str) -> Path:
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        if args.color_mix:
            extent = _compute_hexbin_extent(pitch_all, cot_all)
            bin_index, good_idxs, _, n_bins = _hexbin_indices(
                pitch_all,
                cot_all,
                args.gridsize,
                extent,
                mincnt=1,
            )
            counts_by_terrain = np.zeros((n_bins, len(terrain_order)), dtype=np.int64)
            valid = bin_index >= 0
            np.add.at(counts_by_terrain, (bin_index[valid], terrain_idx[valid].astype(int)), 1)
            base_colors = np.asarray(cmap.colors, dtype=np.float64)
            mixed_colors = np.zeros((n_bins, 4), dtype=np.float64)
            for idx in range(n_bins):
                weights = _mix_weights(counts_by_terrain[idx], args.color_mix_min)
                if weights.sum() > 0.0:
                    mixed_colors[idx] = weights @ base_colors
            hb = ax.hexbin(
                pitch_all,
                cot_all,
                gridsize=args.gridsize,
                extent=extent,
                mincnt=1,
                linewidths=0.0,
            )
            hb.set_array(None)
            hb.set_facecolors(mixed_colors[good_idxs])
        else:
            hb = ax.hexbin(
                pitch_all,
                cot_all,
                C=terrain_idx,
                reduce_C_function=_dominant_terrain,
                gridsize=args.gridsize,
                cmap=cmap,
                norm=norm,
                mincnt=1,
                linewidths=0.0,
            )
        if y_range is not None:
            ax.set_ylim(y_range)
        ax.set_xlabel("pitch_deg_mean [deg]")
        ax.set_ylabel(cot_col)
        title_suffix = "" if y_range is None else f" (y in [{y_range[0]}, {y_range[1]}])"
        ax.set_title(f"Patch mean robot pitch vs {cot_col}{title_suffix}")
        if x_plot.size and fit_all is not None:
            ax.plot(x_plot, np.polyval(fit_all, x_plot), color="tab:red", lw=1.4, label="quad fit (all)")
        if x_plot.size and fit_no_outliers is not None:
            ax.plot(x_plot, np.polyval(fit_no_outliers, x_plot), color="tab:orange", lw=1.4, linestyle="--", label="quad fit (no outliers)")
        line_handles, line_labels = ax.get_legend_handles_labels()
        if line_handles:
            legend1 = ax.legend(line_handles, line_labels, loc="upper left")
            ax.add_artist(legend1)
        terrain_handles = [
            Patch(facecolor=cmap.colors[i], edgecolor="none", label=terrain)
            for i, terrain in enumerate(terrain_order)
        ]
        ax.legend(
            handles=terrain_handles,
            title="Terrain",
            loc="upper right",
            ncol=_legend_columns(len(terrain_handles)),
            frameon=True,
        )
        ax.grid(alpha=0.25)
        fig.tight_layout()

        out_path = base_out if suffix == "" else base_out.with_name(f"{base_out.stem}{suffix}{base_out.suffix}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"[ok] wrote {out_path}")
        return out_path

    _plot(None, "")
    if args.y_range is not None:
        _plot((args.y_range[0], args.y_range[1]), "_restricted")

    if args.add_terrain_plots:
        per_dir = base_out.parent / "per_terrain"
        per_dir.mkdir(parents=True, exist_ok=True)
        plot_base = "patch_pitch_vs_cot_p95" if args.p95 else "patch_pitch_vs_cot"
        for terrain in terrain_order:
            df_terrain = df_selected[df_selected["terrain"] == terrain]
            pitch_raw, cot_raw, base_mask = _prepare_pitch_cot(df_terrain, cot_col)
            pitch_t = pitch_raw[base_mask]
            cot_t = cot_raw[base_mask]
            if pitch_t.size == 0 or cot_t.size == 0:
                print(f"[warn] No finite pitch/cot data for terrain '{terrain}'.")
                continue

            def _plot_counts(y_range: tuple[float, float] | None, suffix: str) -> None:
                fig, ax = plt.subplots(figsize=(7.5, 5.5))
                extent = (
                    xlim[0],
                    xlim[1],
                    y_range[0],
                    y_range[1],
                ) if y_range is not None else (
                    xlim[0],
                    xlim[1],
                    ylim[0],
                    ylim[1],
                )
                hb = ax.hexbin(
                    pitch_t,
                    cot_t,
                    gridsize=args.gridsize,
                    extent=extent,
                    cmap="viridis",
                    mincnt=1,
                    linewidths=0.0,
                )
                ax.set_xlim(xlim)
                if y_range is not None:
                    ax.set_ylim(y_range)
                else:
                    ax.set_ylim(ylim)
                ax.set_xlabel("pitch_deg_mean [deg]")
                ax.set_ylabel(cot_col)
                title_suffix = "" if y_range is None else f" (y in [{y_range[0]}, {y_range[1]}])"
                ax.set_title(f"{terrain}: patch mean robot pitch vs {cot_col}{title_suffix}")
                cb = fig.colorbar(hb, ax=ax)
                cb.set_label("patch count")
                ax.grid(alpha=0.25)
                fig.tight_layout()

                slug = _slugify(terrain)
                out_path = per_dir / f"{plot_base}_{slug}{suffix}.png"
                fig.savefig(out_path, dpi=200)
                plt.close(fig)
                print(f"[ok] wrote {out_path}")

            _plot_counts(None, "")
            if args.y_range is not None:
                _plot_counts((args.y_range[0], args.y_range[1]), "_restricted")


if __name__ == "__main__":
    main()
