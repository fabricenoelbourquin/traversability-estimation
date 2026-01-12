#!/usr/bin/env python3
"""
Build a patch-level HDF5 dataset for one mission.

Features per patch:
  - Heightmap (DEM): single plane-fit slope in E/N/magnitude, gradient orientation, and quadratic-fit curvatures (k1/k2, mean, abs, directional along/ across heading).
  - DEM (multi-scale): pitch/roll estimates along/ across heading from DEM gradients at multiple smoothing scales.
  - Robot: mean/p5/p95 for selected metrics, actual speed, commanded speed, pitch [deg];
           distance traveled, time span, mean bearing quaternion.

Patches are centered on the trajectory (distance-based stride) and use a square footprint.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml
from pyproj import Transformer

from utils.paths import get_paths
from utils.cli import add_mission_arguments, add_hz_argument, resolve_mission_from_args
from utils.synced import resolve_synced_parquet, infer_hz_from_path
from utils.filtering import load_metrics_config
from utils.geo import find_lat_lon_cols, format_dem_scale_label, discover_dem_path, world_to_rowcol
from utils.quaternion import euler_zyx_from_wxyz, normalize_quat_arrays, yaw_from_wxyz, get_quaternion_block


# -------------------------- helpers --------------------------

def load_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text()) if p.exists() else {}


def format_patch_size_label(size_m: float) -> str:
    # Format patch size label without trailing zeros
    return f"{size_m:.3f}".rstrip("0").rstrip(".")


def resolve_metric_names(cfg_metrics: dict, repo_root: Path) -> list[str]:
    names = cfg_metrics.get("names") or []
    if names:
        return [str(n) for n in names]
    metrics_cfg = load_yaml(repo_root / "config" / "metrics.yaml")
    return list(metrics_cfg.get("metrics", {}).get("names", []))


def fit_plane_to_patch(z_patch: np.ndarray,
                       rows_abs: np.ndarray,
                       cols_abs: np.ndarray,
                       center_row: int,
                       center_col: int,
                       res_x: float,
                       res_y: float) -> tuple[float, float]:
    """
    Fit a single plane to all finite DEM samples in a patch.
    Returns slopes along +E (grad_x) and +N (grad_y).
    """
    mask = np.isfinite(z_patch)
    if not mask.any():
        return (float("nan"), float("nan"))

    dx_m = (cols_abs - center_col) * res_x
    dy_m = (rows_abs - center_row) * res_y

    A = np.stack([
        dx_m[mask],
        dy_m[mask],
        np.ones(mask.sum(), dtype=np.float64)
    ], axis=1)
    b = z_patch[mask]

    if A.shape[0] < 3:
        return (float("nan"), float("nan"))

    coeffs, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    if rank < 3:
        return (float("nan"), float("nan"))

    p, q = coeffs[0], coeffs[1]  # east, north slopes
    return float(p), float(q)


def fit_quadratic_patch(z_patch: np.ndarray,
                        rows_abs: np.ndarray,
                        cols_abs: np.ndarray,
                        center_row: int,
                        center_col: int,
                        res_x: float,
                        res_y: float) -> tuple[float, float, float, float, float, float] | None:
    """
    Fit quadratic surface z = a x^2 + b y^2 + c x y + d x + e y + f over the patch (x=east, y=north).
    Returns coefficients (a, b, c, d, e, f) in meters-based coordinates.
    """
    mask = np.isfinite(z_patch)
    if not mask.any():
        # abort if no valid data
        return None
    # make (0,0) the patch center and convert pixels to meters
    dx_m = (cols_abs - center_col) * res_x
    dy_m = (rows_abs - center_row) * res_y

    A = np.stack([
        (dx_m * dx_m)[mask],
        (dy_m * dy_m)[mask],
        (dx_m * dy_m)[mask],
        dx_m[mask],
        dy_m[mask],
        np.ones(mask.sum(), dtype=np.float64)
    ], axis=1)
    b = z_patch[mask]

    if A.shape[0] < 6:
        # need at least 6 points to fit quadratic
        return None

    coeffs, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    if rank < 6:
        return None

    return tuple(float(x) for x in coeffs)  # type: ignore[return-value]


def select_patch_centers(distances: np.ndarray, valid_idx: np.ndarray, stride_m: float) -> list[int]:
    if distances.size == 0 or stride_m <= 0:
        return []
    d = np.maximum.accumulate(distances)
    centers: list[int] = []
    target = d[0]
    while True:
        pos = int(np.searchsorted(d, target, side="left"))
        if pos >= len(d):
            break
        centers.append(int(valid_idx[pos]))
        target = d[pos] + stride_m
    return centers


def nan_stats(arr: np.ndarray) -> tuple[float, float, float]:
    """Return mean/p5/p95 for finite values; NaN if empty."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return (np.nan, np.nan, np.nan)
    p5 = float(np.nanpercentile(finite, 5.0))
    p95 = float(np.nanpercentile(finite, 95.0))
    return (float(np.nanmean(finite)), p5, p95)


def circular_mean(angles: np.ndarray) -> float:
    finite = angles[np.isfinite(angles)]
    if finite.size == 0:
        return float("nan")
    s = np.nanmean(np.sin(finite))
    c = np.nanmean(np.cos(finite))
    if np.isnan(s) or np.isnan(c) or (s == 0 and c == 0):
        return float("nan")
    return float(math.atan2(s, c))


def get_time_ranges(selection_cfg: dict, mission_keys: list[str]) -> list[tuple[float, float]]:
    ranges_cfg = (selection_cfg or {}).get("time_ranges_s") or {}
    if not isinstance(ranges_cfg, dict):
        return []
    for key in mission_keys:
        if key is None:
            continue
        if key in ranges_cfg:
            raw = ranges_cfg[key] or []
            out = []
            for pair in raw:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    continue
                try:
                    start = float(pair[0]); end = float(pair[1])
                except Exception:
                    continue
                out.append((start, end))
            return out
    return []


def make_segments(df: pd.DataFrame, ranges: list[tuple[float, float]], use_col: str = "t_rel") -> list[pd.DataFrame]:
    if not ranges:
        return [df]
    segs = []
    for start, end in ranges:
        seg = df[(df[use_col] >= start) & (df[use_col] <= end)].copy()
        if len(seg):
            segs.append(seg)
    return segs


def average_quaternion(df: pd.DataFrame) -> tuple[float, float, float, float]:
    for cols in (("qw_WB", "qx_WB", "qy_WB", "qz_WB"), ("qw", "qx", "qy", "qz")):
        if all(c in df.columns for c in cols):
            sub = df[list(cols)].dropna().to_numpy(dtype=float)
            if len(sub) == 0:
                continue
            qw, qx, qy, qz = normalize_quat_arrays(sub[:, 0], sub[:, 1], sub[:, 2], sub[:, 3])
            q = np.column_stack([qw, qx, qy, qz])
            base = q[0]
            aligned = []
            for qi in q:
                if np.dot(base, qi) < 0:
                    qi = -qi
                aligned.append(qi)
            mean_q = np.mean(aligned, axis=0)
            n = np.linalg.norm(mean_q)
            if n == 0:
                continue
            mean_q = mean_q / n
            if mean_q[0] < 0:
                mean_q = -mean_q
            return tuple(float(x) for x in mean_q)
    return (np.nan, np.nan, np.nan, np.nan)


def compute_pitch_deg(df: pd.DataFrame) -> np.ndarray | None:
    """
    Compute pitch [deg] from body->world quaternions.
    Matches visualization (nose-up positive) by flipping the sign.
    """
    block = get_quaternion_block(df)
    if block is None:
        return None
    _, pitch_deg, _ = euler_zyx_from_wxyz(*block, degrees=True)
    return -pitch_deg  # flip axis so nose-up is positive


def yaw_from_quaternion(qw: float, qx: float, qy: float, qz: float) -> float:
    """Extract yaw [rad] from a single quaternion (body→world), ENU frame."""
    if not all(np.isfinite([qw, qx, qy, qz])):
        return float("nan")
    return float(yaw_from_wxyz(qw, qx, qy, qz))


def aggregate_dem_pitch_roll_from_samples(df_patch: pd.DataFrame, scales_m: list[float]) -> dict[str, float]:
    """
    Aggregate precomputed per-sample DEM pitch/roll values inside a patch.
    Expects columns dem_pitch_<scale>_deg and dem_roll_<scale>_deg.
    """
    out: dict[str, float] = {}
    for scale_m in scales_m:
        label = format_dem_scale_label(scale_m)
        pitch_col = f"dem_pitch_{label}_deg"
        roll_col = f"dem_roll_{label}_deg"
        if pitch_col in df_patch.columns:
            out[f"pitch_dem_{label}_mean"], out[f"pitch_dem_{label}_p5"], out[f"pitch_dem_{label}_p95"] = nan_stats(
                df_patch[pitch_col].to_numpy(dtype=float)
            )
        else:
            out[f"pitch_dem_{label}_mean"] = np.nan
            out[f"pitch_dem_{label}_p5"] = np.nan
            out[f"pitch_dem_{label}_p95"] = np.nan
        if roll_col in df_patch.columns:
            out[f"roll_dem_{label}_mean"], out[f"roll_dem_{label}_p5"], out[f"roll_dem_{label}_p95"] = nan_stats(
                df_patch[roll_col].to_numpy(dtype=float)
            )
        else:
            out[f"roll_dem_{label}_mean"] = np.nan
            out[f"roll_dem_{label}_p5"] = np.nan
            out[f"roll_dem_{label}_p95"] = np.nan
    return out


def summarize_metric(df: pd.DataFrame, col: str) -> tuple[float, float, float]:
    if col not in df.columns:
        return (np.nan, np.nan, np.nan)
    arr = df[col].to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (np.nan, np.nan, np.nan)
    return nan_stats(arr)


def compute_patch_cot(df_patch: pd.DataFrame,
                      mass: float,
                      gravity: float,
                      min_cmd_speed: float,
                      power_col: str = "power",
                      min_cmd_pad_s: float = 0.0,
                      turn_min_wz: float = 0.0,
                      turn_lin_thresh: float | None = None,
                      turn_pad_s: float = 0.0) -> tuple[float, float]:
    """
    Distance-normalized energy over the patch using the specified power column,
    skipping samples with commanded speed below min_cmd_speed (optionally padded in time)
    and excluding near-pure turning (high w_cmd_z with low linear command).
    Returns (COT, COT_trimmed_p95).
    """
    # Inputs & Safety Checks
    if not np.isfinite(mass) or not np.isfinite(gravity) or mass <= 0.0 or gravity <= 0.0:
        return (np.nan, np.nan)
    if power_col not in df_patch.columns or "t" not in df_patch or "dist_m" not in df_patch:
        return (np.nan, np.nan)

    # extract commanded speed
    if "v_cmd" in df_patch:
        v_cmd = df_patch["v_cmd"].to_numpy(dtype=float)
    elif {"v_cmd_x", "v_cmd_y"}.issubset(df_patch.columns):
        v_cmd = np.hypot(df_patch["v_cmd_x"], df_patch["v_cmd_y"])
    else:
        return (np.nan, np.nan)

    # extract power, time, distance
    power = df_patch[power_col].to_numpy(dtype=float)
    t = df_patch["t"].to_numpy(dtype=float)
    dist = df_patch["dist_m"].to_numpy(dtype=float)

    # Base validity mask (finite data)
    valid = np.isfinite(power) & np.isfinite(t) & np.isfinite(dist) & np.isfinite(v_cmd)
    if not np.any(valid):
        return (np.nan, np.nan)

    def _expand_mask_by_time(base_mask: np.ndarray, times: np.ndarray, padding_s: float) -> np.ndarray:
        # Expand a boolean mask by padding True regions by padding_s seconds on each side.
        if not np.any(base_mask): return base_mask
        
        # Find indices where mask switches from False->True or True->False
        # This identifies "blocks" of True values
        padded_mask = np.concatenate(([False], base_mask, [False]))
        diffs = np.diff(padded_mask.astype(int))
        starts_idx = np.where(diffs == 1)[0]
        ends_idx = np.where(diffs == -1)[0] - 1
        
        out = np.zeros_like(base_mask, dtype=bool)
        
        # Expand only the block boundaries
        for start, end in zip(starts_idx, ends_idx):
            t_start = times[start] - padding_s
            t_end = times[end] + padding_s
            
            # Binary search for new boundaries
            new_start = np.searchsorted(times, t_start, side='left')
            new_end = np.searchsorted(times, t_end, side='right')
            
            out[new_start:new_end] = True
            
        return out
    
    # Mask A: Low Command Speed
    low_cmd = v_cmd < max(min_cmd_speed, 0.0)
    if min_cmd_pad_s > 0.0:
        # Expand low_cmd mask by padding in time
        low_cmd = _expand_mask_by_time(low_cmd, t, min_cmd_pad_s)

    # Mask B: Turn-in-Place Commands
    turn_only = np.zeros_like(valid)
    if turn_min_wz > 0.0 and "w_cmd_z" in df_patch:
        # Identify turn-in-place commands (turn and slow speed), exclude from COT
        w_cmd = np.abs(df_patch["w_cmd_z"].to_numpy(dtype=float))
        lin_thresh = turn_lin_thresh if (turn_lin_thresh is not None and np.isfinite(turn_lin_thresh)) else min_cmd_speed
        turn_only = valid & (w_cmd >= turn_min_wz) & (v_cmd < lin_thresh)
        if turn_pad_s > 0.0:
            turn_only = _expand_mask_by_time(turn_only, t, turn_pad_s)

    # Combine masks
    valid &= ~(low_cmd | turn_only)
    if not np.any(valid):
        return (np.nan, np.nan)

    # Integration & CoT Calculation
    # Pre-calculate Raw Deltas
    # Note: dist is cumulative (odometer), so diff gives incremental distance
    raw_dt = np.diff(t, prepend=t[0]) # raw_dt[i] = t[i] - t[i-1]
    raw_dist = np.diff(dist, prepend=dist[0])

    # Clean raw deltas (handle gaps/reverse motion globally first)
    raw_dt = np.clip(raw_dt, 0.0, None)
    raw_dt[~np.isfinite(raw_dt)] = 0.0
    
    raw_dist = np.clip(raw_dist, 0.0, None) # Assume efficiency only for forward motion
    raw_dist[~np.isfinite(raw_dist)] = 0.0

    def _cot_from_mask(mask: np.ndarray) -> float:
        """
        Calculate CoT on a specific subset mask.
        Crucial: Only integrates intervals where BOTH current and previous 
        samples are inside the mask to avoid spanning gaps.
        """
        if not np.any(mask):
            return float("nan")

        # Identify continuous intervals valid under THIS mask
        mask_prev = np.concatenate(([False], mask[:-1]))
        valid_transitions = mask & mask_prev

        # Sum energy and distance only over these valid transitions
        # (We use the pre-calculated raw deltas, but filter them by the mask transitions)
        energy = np.nansum(power[valid_transitions] * raw_dt[valid_transitions])
        distance = np.nansum(raw_dist[valid_transitions])
        
        if distance <= 0.0:
            return float("nan")
            
        return energy / (mass * gravity * distance)

    # 4. Final Calculations
    cot_all = _cot_from_mask(valid)

    # Trimmed COT (Robustness against outliers)
    cot_trimmed = float("nan")
    if np.any(valid):
        try:
            lo, hi = np.nanpercentile(power[valid], [5.0, 95.0])
            if np.isfinite(lo) and np.isfinite(hi):
                trimmed_mask = valid & (power >= lo) & (power <= hi)
                cot_trimmed = _cot_from_mask(trimmed_mask)
        except Exception:
            pass # Fallback to nan if percentile fails

    return (cot_all, cot_trimmed)


def aggregate_robot_patch(df_patch: pd.DataFrame,
                          metric_names: list[str],
                          include_speed: bool,
                          include_cmd_speed: bool,
                          cot_cfg: dict[str, float] | None) -> dict:
    out: dict[str, float] = {}

    for m in metric_names:
        mu, p5, p95 = summarize_metric(df_patch, m)
        out[f"metric_{m}_mean"] = mu
        out[f"metric_{m}_p5"] = p5
        out[f"metric_{m}_p95"] = p95

    if include_speed:
        if "v_actual" in df_patch:
            speed = df_patch["v_actual"].to_numpy(dtype=float)
        elif {"vx", "vy"}.issubset(df_patch.columns):
            speed = np.hypot(df_patch["vx"], df_patch["vy"])
        elif "speed" in df_patch:
            speed = df_patch["speed"].to_numpy(dtype=float)
        else:
            speed = np.array([], dtype=float)
        if speed.size:
            out["speed_mean"], out["speed_p5"], out["speed_p95"] = nan_stats(speed)
        else:
            out["speed_mean"] = out["speed_p5"] = out["speed_p95"] = np.nan

    if include_cmd_speed:
        if "v_cmd" in df_patch:
            vcmd = df_patch["v_cmd"].to_numpy(dtype=float)
        elif {"v_cmd_x", "v_cmd_y"}.issubset(df_patch.columns):
            vcmd = np.hypot(df_patch["v_cmd_x"], df_patch["v_cmd_y"])
        else:
            vcmd = np.array([], dtype=float)
        if vcmd.size:
            out["v_cmd_mean"], out["v_cmd_p5"], out["v_cmd_p95"] = nan_stats(vcmd)
        else:
            out["v_cmd_mean"] = out["v_cmd_p5"] = out["v_cmd_p95"] = np.nan
        if cot_cfg:
            cot_val, cot_trim = compute_patch_cot(
                df_patch,
                cot_cfg.get("mass", np.nan),
                cot_cfg.get("gravity", np.nan),
                cot_cfg.get("min_cmd_speed", 0.0),
                cot_cfg.get("power_col", "power"),
                cot_cfg.get("min_cmd_pad_s", 0.0),
                cot_cfg.get("turn_min_wz", 0.0),
                cot_cfg.get("turn_lin_thresh", None),
                cot_cfg.get("turn_pad_s", 0.0),
            )
            out["cot_patch"] = cot_val
            out["cot_patch_p95"] = cot_trim
            out["cot_min_cmd_speed"] = cot_cfg.get("min_cmd_speed", np.nan)

    # Time span
    if "t" in df_patch:
        tvals = df_patch["t"].to_numpy(dtype=float)
        finite_t = tvals[np.isfinite(tvals)]
        if finite_t.size:
            out["t_start"] = float(np.nanmin(finite_t))
            out["t_end"] = float(np.nanmax(finite_t))
            out["time_span_s"] = float(out["t_end"] - out["t_start"])
        else:
            out["t_start"] = out["t_end"] = out["time_span_s"] = np.nan

    # Distance traveled
    if "dist_m" in df_patch:
        dvals = df_patch["dist_m"].to_numpy(dtype=float)
        finite_d = dvals[np.isfinite(dvals)]
        if finite_d.size:
            out["distance_traveled_m"] = float(np.nanmax(finite_d) - np.nanmin(finite_d))
        else:
            out["distance_traveled_m"] = np.nan
    else:
        out["distance_traveled_m"] = np.nan

    # Mean bearing quaternion
    qw, qx, qy, qz = average_quaternion(df_patch)
    out["bearing_qw"] = qw
    out["bearing_qx"] = qx
    out["bearing_qy"] = qy
    out["bearing_qz"] = qz

    # Pitch statistics (deg), using percentiles to reduce outlier influence
    pitch = compute_pitch_deg(df_patch)
    if pitch is not None and pitch.size:
        out["pitch_deg_mean"], out["pitch_deg_p5"], out["pitch_deg_p95"] = nan_stats(pitch)
    else:
        out["pitch_deg_mean"] = out["pitch_deg_p5"] = out["pitch_deg_p95"] = np.nan

    return out


def save_hdf(group_name: str,
             df: pd.DataFrame,
             out_path: Path,
             attrs: dict,
             overwrite: bool,
             compression: str | None) -> None:
    try:
        import h5py  # type: ignore
    except ImportError as e:
        raise SystemExit("h5py is required to write the dataset (pip install h5py).") from e

    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = df.to_records(index=False)

    # simple lock to avoid concurrent writes to the same HDF5 file
    lock_path = out_path.with_suffix(out_path.suffix + ".lock")
    lock_fd = None
    for _ in range(60):  # wait up to ~60s
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            break
        except FileExistsError:
            time.sleep(1.0)
    if lock_fd is None:
        raise SystemExit(f"Could not acquire lock for {out_path} (lock file exists: {lock_path})")

    try:
        with h5py.File(out_path, "a") as f:
            grp = f.require_group(group_name)
            if "patches" in grp:
                if not overwrite:
                    raise SystemExit(f"Group '{group_name}' already has patches; use overwrite_mission=true to replace.")
                del grp["patches"]
            ds = grp.create_dataset("patches", data=data, compression=compression)
            for k, v in attrs.items():
                grp.attrs[k] = v
            grp.attrs["num_patches"] = len(df)
            grp.attrs["column_order"] = json.dumps(list(df.columns))
            ds.attrs["column_order"] = json.dumps(list(df.columns))
    finally:
        try:
            if lock_fd is not None:
                os.close(lock_fd)
            lock_path.unlink(missing_ok=True)
        except Exception:
            pass


# -------------------------- main --------------------------

def main():
    ap = argparse.ArgumentParser(description="Build patch-level HDF5 dataset for one mission.")
    add_mission_arguments(ap)
    ap.add_argument("--config", default="config/dataset.yaml", help="Dataset config YAML")
    add_hz_argument(ap, help_text="Pick synced_<Hz>Hz*.parquet (default: config or latest)")
    ap.add_argument("--patch-size-m", type=float, default=None, help="Override patch size (meters)")
    ap.add_argument("--overlap", type=float, default=None, help="Override overlap ratio (0..0.5)")
    ap.add_argument("--stride-m", type=float, default=None, help="Override stride (meters)")
    ap.add_argument("--out", type=str, default=None, help="Optional output HDF5 path")
    ap.add_argument("--dem", type=str, default=None, help="Optional DEM override (GeoTIFF)")
    ap.add_argument("--allow-missing-gps", action="store_true", help="Proceed without lat/lon (geo fields become NaN).")
    ap.add_argument("--allow-missing-dem", action="store_true", help="Proceed without DEM (height/gradient fields become NaN).")
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))
    patch_cfg = cfg.get("patch", {})
    input_cfg = cfg.get("inputs", {})
    metrics_cfg = cfg.get("metrics", {})
    output_cfg = cfg.get("output", {})
    swiss_cfg = cfg.get("swissimage", {})
    selection_cfg = cfg.get("selection", {})
    allow_missing_gps = bool(args.allow_missing_gps)
    allow_missing_dem = bool(args.allow_missing_dem)

    patch_size_m = float(args.patch_size_m or patch_cfg.get("size_m", 5.0))
    overlap = args.overlap if args.overlap is not None else float(patch_cfg.get("overlap_ratio", 0.5))
    overlap = max(0.0, min(0.5, overlap))
    stride_m = args.stride_m if args.stride_m is not None else patch_cfg.get("stride_m")
    if stride_m is None:
        stride_m = patch_size_m * (1.0 - overlap)
    stride_m = float(max(stride_m, 1e-3))

    min_height_frac = float(patch_cfg.get("min_height_valid_frac", 0.6))
    min_robot_samples = int(patch_cfg.get("min_robot_samples", 5))

    include_dino = bool(swiss_cfg.get("include_dino_embeddings", False))

    P = get_paths()

    metrics_cfg_full = load_metrics_config(Path(P["REPO_ROOT"]) / "config" / "metrics.yaml")
    dem_pitch_roll_cfg = metrics_cfg_full.get("dem_pitch_roll")
    if not isinstance(dem_pitch_roll_cfg, dict):
        raise SystemExit("Missing or invalid 'dem_pitch_roll' in config/metrics.yaml.")
    # Get DEM pitch/roll config and check for validity
    dem_smooth_scales_m_raw = dem_pitch_roll_cfg.get("smooth_scales_m")
    if not isinstance(dem_smooth_scales_m_raw, (list, tuple)):
        raise SystemExit("'dem_pitch_roll.smooth_scales_m' must be a list of meters-based scales.")
    dem_smooth_scales_m = []
    for val in dem_smooth_scales_m_raw:
        try:
            scale_m = float(val)
        except Exception:
            continue
        if np.isfinite(scale_m) and scale_m > 0.0:
            dem_smooth_scales_m.append(scale_m)
    if not dem_smooth_scales_m:
        raise SystemExit("'dem_pitch_roll.smooth_scales_m' must contain positive finite values.")

    if "smooth_window" not in dem_pitch_roll_cfg:
        raise SystemExit("Missing 'dem_pitch_roll.smooth_window' in metrics config.")
    dem_smooth_window = int(dem_pitch_roll_cfg.get("smooth_window"))
    if dem_smooth_window < 1:
        raise SystemExit("'dem_pitch_roll.smooth_window' must be >= 1.")

    if "mad_z_thresh" not in dem_pitch_roll_cfg:
        raise SystemExit("Missing 'dem_pitch_roll.mad_z_thresh' in metrics config.")
    dem_mad_z_thresh = float(dem_pitch_roll_cfg.get("mad_z_thresh"))
    if not np.isfinite(dem_mad_z_thresh) or dem_mad_z_thresh <= 0.0:
        raise SystemExit("'dem_pitch_roll.mad_z_thresh' must be > 0.")
    dem_rate_hampel_filter = bool(dem_pitch_roll_cfg.get("rate_hampel_filter", False))
    dem_rate_hampel_max_rate = float(dem_pitch_roll_cfg.get("rate_hampel_max_rate", np.nan))
    dem_rate_hampel_window = int(dem_pitch_roll_cfg.get("rate_hampel_window", 0))
    dem_rate_hampel_z = float(dem_pitch_roll_cfg.get("rate_hampel_z", np.nan))
    dem_value_gauss_sigma = float(dem_pitch_roll_cfg.get("value_gauss_sigma", np.nan))
    robot_cfg_full = (metrics_cfg_full.get("robot") or {})
    params_cfg_full = (metrics_cfg_full.get("params") or {})
    cot_cfg = {
        "mass": float(robot_cfg_full.get("mass_kg", np.nan)),
        "gravity": float(robot_cfg_full.get("gravity", 9.81)),
        "min_cmd_speed": float(params_cfg_full.get(
            "min_cmd_speed_for_power_norm",
            params_cfg_full.get("min_speed_for_power_norm", 0.0),
        )),
        "min_cmd_pad_s": float(params_cfg_full.get("min_cmd_speed_pad_s", 0.0)),
        "turn_min_wz": float(params_cfg_full.get("turn_only_min_w_cmd_z", 0.0)),
        "turn_lin_thresh": float(params_cfg_full.get(
            "turn_only_max_v_cmd_for_turn",
            params_cfg_full.get("min_cmd_speed_for_power_norm", 0.0),
        )),
        "turn_pad_s": float(params_cfg_full.get("turn_only_pad_s", 0.0)),
        "power_col": str(metrics_cfg.get("cot_power_column", "power")),
    }

    mp = resolve_mission_from_args(args, P)

    synced_path = resolve_synced_parquet(mp.synced, args.hz or input_cfg.get("hz"), prefer_metrics=True)
    hz_used = args.hz or input_cfg.get("hz") or infer_hz_from_path(synced_path)
    print(f"[load] synced: {synced_path}")
    df = pd.read_parquet(synced_path).sort_values("t").reset_index(drop=True)

    # metric names
    metric_names = resolve_metric_names(metrics_cfg, Path(P["REPO_ROOT"]))
    include_speed = bool(metrics_cfg.get("include_speed", True))
    include_cmd = bool(metrics_cfg.get("include_command_speed", True))

    # DEM
    dem_path = None
    dem_crs = None
    transform = None
    z = None
    res_m = float("nan")
    dem_smooth_scales_px: list[float] = []
    try:
        dem_path = discover_dem_path(
            mp.maps,
            bool(input_cfg.get("prefer_dem_from_meta", True)),
            args.dem or input_cfg.get("dem_path"),
        )
        print(f"[load] DEM: {dem_path}")
        with rasterio.open(dem_path) as ds:
            z = ds.read(1).astype(np.float64)
            nodata = ds.nodata
            if nodata is not None:
                z[z == nodata] = np.nan
            transform = ds.transform
            dem_crs = ds.crs
        res_m = float(abs(transform.a))
    except Exception as e:
        if not allow_missing_dem:
            raise
        print(f"[warn] DEM unavailable or unreadable ({e}); proceeding without DEM features.")

    # Optional time ranges per mission (interpreted as seconds since start of mission)
    mission_keys = [args.mission, args.mission_id, mp.display, mp.mission_id]
    time_ranges = get_time_ranges(selection_cfg, [str(k) for k in mission_keys if k])
    t0 = float(df["t"].min()) if len(df) else float("nan")
    df["t_rel_raw"] = df["t"] - t0
    base_segments = make_segments(df, time_ranges, use_col="t_rel_raw") if time_ranges else [df]
    if time_ranges and not base_segments:
        raise SystemExit("No data after applying time ranges.")
    df = pd.concat(base_segments, axis=0).sort_values("t").reset_index(drop=True)
    df["t_rel_raw"] = df["t"] - float(df["t"].min()) + (time_ranges[0][0] if time_ranges else 0.0)

    # lat/lon -> E/N (after time filtering)
    lat_col = lon_col = None
    try:
        lat_col, lon_col = find_lat_lon_cols(df)
    except KeyError as e:
        if allow_missing_gps:
            print(f"[warn] No lat/lon columns found ({e}); continuing without GPS coordinates.")
        else:
            raise

    if dem_crs is not None and lat_col and lon_col:
        lat = df[lat_col].to_numpy(dtype=float)
        lon = df[lon_col].to_numpy(dtype=float)
        transformer = Transformer.from_crs("EPSG:4326", dem_crs, always_xy=True)
        east, north = transformer.transform(lon, lat)
        df["easting_m"] = east
        df["northing_m"] = north
    else:
        lat = df[lat_col].to_numpy(dtype=float) if lat_col else np.full(len(df), np.nan)
        lon = df[lon_col].to_numpy(dtype=float) if lon_col else np.full(len(df), np.nan)

    coord_e_col = coord_n_col = None
    if {"easting_m", "northing_m"}.issubset(df.columns):
        coord_e_col, coord_n_col = "easting_m", "northing_m"
    elif {"x", "y"}.issubset(df.columns):
        coord_e_col, coord_n_col = "x", "y"
    # Build segments (stride restarts per range) using the filtered df with coordinates
    if time_ranges:
        segments = []
        for start, end in time_ranges:
            seg = df[(df["t_rel_raw"] >= start) & (df["t_rel_raw"] <= end)].copy()
            if len(seg):
                segments.append(seg)
    else:
        segments = [df]

    dem_available = z is not None and transform is not None
    dem_labels = [format_dem_scale_label(s) for s in dem_smooth_scales_m]
    use_precomputed_dem = all(
        (f"dem_pitch_{lab}_deg" in df.columns and f"dem_roll_{lab}_deg" in df.columns)
        for lab in dem_labels
    )
    if dem_available and np.isfinite(res_m) and res_m > 0.0:
        dem_smooth_scales_px = [
            float(scale_m / res_m)
            for scale_m in dem_smooth_scales_m
            if np.isfinite(scale_m) and scale_m > 0.0
        ]
    if not use_precomputed_dem:
        raise SystemExit("Missing precomputed DEM pitch/roll columns; run add_dem_features.py first.")
    half_px = max(1, int(round((patch_size_m / 2.0) / res_m))) if dem_available else None
    H, W = z.shape if dem_available else (0, 0)

    rows = []
    skipped_edges = 0
    skipped_height = 0
    skipped_robot = 0
    skipped_dupe = 0
    skipped_contained = 0
    skipped_time_subset = 0
    patch_idx = 0

    for seg in segments:
        if seg.empty:
            continue
        east_seg = seg[coord_e_col].to_numpy(dtype=float) if coord_e_col else np.full(len(seg), np.nan)
        north_seg = seg[coord_n_col].to_numpy(dtype=float) if coord_n_col else np.full(len(seg), np.nan)
        lat_seg = seg[lat_col].to_numpy(dtype=float) if lat_col else np.full(len(seg), np.nan)
        lon_seg = seg[lon_col].to_numpy(dtype=float) if lon_col else np.full(len(seg), np.nan)
        t_seg = seg["t"].to_numpy(dtype=float)
        distances = seg["dist_m"].to_numpy(dtype=float) if "dist_m" in seg else np.full(len(seg), np.nan)

        valid_mask = np.isfinite(distances)
        if coord_e_col:
            valid_mask &= np.isfinite(east_seg)
        if coord_n_col:
            valid_mask &= np.isfinite(north_seg)
        if not valid_mask.any():
            continue

        valid_idx = np.nonzero(valid_mask)[0]
        distances_adj = distances.copy()
        distances_adj[valid_mask] = distances_adj[valid_mask] - distances_adj[valid_idx[0]]
        dist_valid = np.maximum.accumulate(distances_adj[valid_mask])
        centers = select_patch_centers(dist_valid, valid_idx, stride_m)
        if not centers:
            continue

        for ci in centers:
            cx_e = float(east_seg[ci]) if np.isfinite(east_seg[ci]) else float("nan")
            cy_n = float(north_seg[ci]) if np.isfinite(north_seg[ci]) else float("nan")
            cx_lat = float(lat_seg[ci]) if np.isfinite(lat_seg[ci]) else float("nan")
            cx_lon = float(lon_seg[ci]) if np.isfinite(lon_seg[ci]) else float("nan")

            slope_e = slope_n = slope_mag = grad_orient = float("nan")
            k1 = k2 = mean_curv = abs_curv = float("nan")
            curv_heading = curv_cross = float("nan")
            yaw_rad = float("nan")
            height_valid_frac = float("nan")
            side_m = patch_size_m
            quad_coeffs = None
            df_patch = None

            if dem_available and coord_e_col and coord_n_col and half_px is not None:
                row_c, col_c = world_to_rowcol(transform, np.array([cx_e]), np.array([cy_n]))
                r = int(round(row_c[0])); c = int(round(col_c[0]))
                r0 = r - half_px; r1 = r + half_px + 1
                c0 = c - half_px; c1 = c + half_px + 1
                if (r0 < 0 or c0 < 0 or r1 > H or c1 > W):
                    skipped_edges += 1
                    continue

                rows_grid = np.arange(r0, r1)
                cols_grid = np.arange(c0, c1)
                rr_abs, cc_abs = np.meshgrid(rows_grid, cols_grid, indexing="ij")
                z_patch = z[r0:r1, c0:c1]

                valid_cells = np.isfinite(z_patch)
                valid_frac = float(valid_cells.mean()) if valid_cells.size else 0.0
                height_valid_frac = valid_frac
                if valid_frac < min_height_frac:
                    skipped_height += 1
                    continue

                slope_e, slope_n = fit_plane_to_patch(z_patch, rr_abs, cc_abs, r, c, transform.a, transform.e)
                slope_mag = math.hypot(slope_e, slope_n)
                grad_orient = float(math.atan2(slope_n, slope_e)) if np.isfinite(slope_e) and np.isfinite(slope_n) else float("nan")
                in_patch = (
                    (np.abs(east_seg - cx_e) <= (patch_size_m / 2.0)) &
                    (np.abs(north_seg - cy_n) <= (patch_size_m / 2.0)) &
                    np.isfinite(east_seg) & np.isfinite(north_seg)
                )
                df_patch = seg.loc[in_patch].copy()
                side_m = len(rows_grid) * res_m
                quad_coeffs = fit_quadratic_patch(z_patch, rr_abs, cc_abs, r, c, transform.a, transform.e)
            else:
                if coord_e_col and coord_n_col:
                    in_patch = (
                        (np.abs(east_seg - cx_e) <= (patch_size_m / 2.0)) &
                        (np.abs(north_seg - cy_n) <= (patch_size_m / 2.0))
                    )
                    in_patch &= np.isfinite(east_seg) & np.isfinite(north_seg)
                else:
                    center_dist = distances_adj[ci] if np.isfinite(distances_adj[ci]) else distances[ci]
                    in_patch = (
                        np.isfinite(distances_adj) &
                        np.isfinite(center_dist) &
                        (np.abs(distances_adj - center_dist) <= (patch_size_m / 2.0))
                    )
                df_patch = seg.loc[in_patch].copy()

            if df_patch is None or len(df_patch) < min_robot_samples:
                skipped_robot += 1
                continue

            robot_feats = aggregate_robot_patch(df_patch, metric_names, include_speed, include_cmd, cot_cfg if include_cmd else None)
            min_dist_m = float(patch_size_m * 0.9)  # allow small slack; need ~90% of patch size traveled
            if np.isfinite(robot_feats.get("distance_traveled_m", np.nan)) and robot_feats["distance_traveled_m"] < min_dist_m:
                skipped_robot += 1
                continue

            yaw_rad = yaw_from_quaternion(
                robot_feats.get("bearing_qw", np.nan),
                robot_feats.get("bearing_qx", np.nan),
                robot_feats.get("bearing_qy", np.nan),
                robot_feats.get("bearing_qz", np.nan),
            )

            dem_pitch_roll_feats = aggregate_dem_pitch_roll_from_samples(df_patch, dem_smooth_scales_m)

            if quad_coeffs is not None:
                a2, b2, c2, d2, e2, _ = quad_coeffs
                grad_sq = d2 * d2 + e2 * e2
                one_plus_g = 1.0 + grad_sq
                denom_H = (one_plus_g ** 1.5)
                f_xx = 2.0 * a2
                f_xy = c2
                f_yy = 2.0 * b2
                if denom_H > 0.0 and np.isfinite([f_xx, f_xy, f_yy]).all():
                    mean_curv = ((1.0 + e2 * e2) * f_xx - 2.0 * d2 * e2 * f_xy + (1.0 + d2 * d2) * f_yy) / (2.0 * denom_H)
                    K = (f_xx * f_yy - f_xy * f_xy) / (one_plus_g ** 2)
                    disc = mean_curv * mean_curv - K
                    if disc < 0.0:
                        disc = 0.0
                    root = math.sqrt(disc)
                    k1 = mean_curv + root
                    k2 = mean_curv - root
                    abs_curv = abs(k1) + abs(k2)

                    denom_norm = math.sqrt(one_plus_g)
                    L = f_xx / denom_norm
                    M = f_xy / denom_norm
                    N = f_yy / denom_norm
                    E = 1.0 + d2 * d2
                    F = d2 * e2
                    G = 1.0 + e2 * e2

                    if np.isfinite(yaw_rad):
                        u = math.cos(yaw_rad); v = math.sin(yaw_rad)
                        denom_dir = E * u * u + 2.0 * F * u * v + G * v * v
                        if denom_dir != 0.0 and np.isfinite(denom_dir):
                            curv_heading = (L * u * u + 2.0 * M * u * v + N * v * v) / denom_dir
                        u_perp = -v; v_perp = u
                        denom_perp = E * u_perp * u_perp + 2.0 * F * u_perp * v_perp + G * v_perp * v_perp
                        if denom_perp != 0.0 and np.isfinite(denom_perp):
                            curv_cross = (L * u_perp * u_perp + 2.0 * M * u_perp * v_perp + N * v_perp * v_perp) / denom_perp

            row = {
                "patch_index": patch_idx,
                "patch_size_m": patch_size_m,
                "patch_stride_m": stride_m,
                "patch_side_m_actual": side_m,
                "center_t": float(t_seg[ci]) if "t" in seg else np.nan,
                "center_lat": float(cx_lat),
                "center_lon": float(cx_lon),
                "center_e": float(cx_e),
                "center_n": float(cy_n),
                "height_valid_fraction": float(height_valid_frac),
                "slope_e": float(slope_e),
                "slope_n": float(slope_n),
                "slope_mag": float(slope_mag),
                "grad_orientation": grad_orient,
                "k1": float(k1),
                "k2": float(k2),
                "curvature_mean": float(mean_curv),
                "curvature_abs": float(abs_curv),
                "curvature_heading": float(curv_heading),
                "curvature_cross_heading": float(curv_cross),
                "heading_yaw_rad": float(yaw_rad),
                "samples": int(len(df_patch)),
            }
            row.update(robot_feats)
            row.update(dem_pitch_roll_feats)
            rows.append(row)
            patch_idx += 1

    if not rows:
        raise SystemExit("No patches produced (all skipped).")

    # Drop exact-duplicate patches (same center + time span) to avoid full inclusions from stride/rounding quirks.
    deduped: list[dict] = []
    seen: set[tuple] = set()

    def _quant(val: float) -> float | None:
        return float(round(float(val), 3)) if np.isfinite(val) else None

    for r in rows:
        key = (
            _quant(r.get("center_e", float("nan"))),
            _quant(r.get("center_n", float("nan"))),
            _quant(r.get("t_start", float("nan"))),
            _quant(r.get("t_end", float("nan"))),
        )
        if None in key:
            deduped.append(r)
            continue
        if key in seen:
            skipped_dupe += 1
            continue
        seen.add(key)
        deduped.append(r)

    rows = deduped

    # Drop patches that are fully contained (in space + time) inside an earlier patch.
    def _contained(inner: dict, outer: dict) -> bool:
        # Spatial containment (square footprint, same patch size)
        size = float(inner.get("patch_size_m", patch_size_m))
        if not np.isfinite(size):
            return False
        dx = abs(float(inner.get("center_e", np.nan)) - float(outer.get("center_e", np.nan)))
        dy = abs(float(inner.get("center_n", np.nan)) - float(outer.get("center_n", np.nan)))
        if not (np.isfinite(dx) and np.isfinite(dy)):
            return False
        spatial_inside = (dx <= size / 2.0) and (dy <= size / 2.0)

        # Temporal containment
        t0_in = float(inner.get("t_start", np.nan))
        t1_in = float(inner.get("t_end", np.nan))
        t0_out = float(outer.get("t_start", np.nan))
        t1_out = float(outer.get("t_end", np.nan))
        temporal_inside = (
            np.isfinite([t0_in, t1_in, t0_out, t1_out]).all() and
            t0_in >= t0_out and t1_in <= t1_out
        )

        return spatial_inside and temporal_inside

    filtered: list[dict] = []
    for r in rows:
        if any(_contained(r, kept) for kept in filtered):
            skipped_contained += 1
            continue
        filtered.append(r)


    # Drop patches whose time span is fully contained in an earlier patch (per mission),
    # regardless of spatial offset, to avoid near-duplicate temporal coverage.
    time_filtered: list[dict] = []
    def _time_subset(inner: dict, outer: dict, tol: float = 1e-3) -> bool:
        t0_in = float(inner.get("t_start", np.nan))
        t1_in = float(inner.get("t_end", np.nan))
        t0_out = float(outer.get("t_start", np.nan))
        t1_out = float(outer.get("t_end", np.nan))
        if not np.isfinite([t0_in, t1_in, t0_out, t1_out]).all():
            return False
        return (t0_in >= t0_out - tol) and (t1_in <= t1_out + tol)

    for r in filtered:
        if any(_time_subset(r, kept) for kept in time_filtered):
            skipped_time_subset += 1
            continue
        time_filtered.append(r)

    rows = time_filtered

    df_out = pd.DataFrame(rows).sort_values("patch_index").reset_index(drop=True)

    # Output path
    if args.out:
        out_path = Path(args.out)
    else:
        out_dir = Path(output_cfg.get("path") or P["DATASETS"])
        fname_tpl = output_cfg.get("filename", "patches_{patch_size_m}m.h5")
        label = format_patch_size_label(patch_size_m)
        fname = fname_tpl.format(patch_size_m=label)
        out_path = out_dir / fname

    attrs = {
        "mission_id": mp.mission_id,
        "mission_folder": mp.folder,
        "mission_display": mp.display,
        "patch_size_m": patch_size_m,
        "stride_m": stride_m,
        "min_height_valid_frac": min_height_frac,
        "min_robot_samples": min_robot_samples,
        "hz": float(hz_used) if hz_used is not None else float("nan"),
        "dem_path": str(dem_path) if dem_path else "",
        "include_dino_embeddings": include_dino,
        "dem_pitch_roll_scales_m": json.dumps(dem_smooth_scales_m),
        "dem_pitch_roll_scales_px": json.dumps(dem_smooth_scales_px) if dem_smooth_scales_px else "",
        "dem_pitch_roll_smooth_window": dem_smooth_window,
        "dem_pitch_roll_mad_z_thresh": dem_mad_z_thresh,
        "dem_pitch_roll_rate_hampel_filter": bool(dem_rate_hampel_filter),
        "dem_pitch_roll_rate_hampel_max_rate": dem_rate_hampel_max_rate,
        "dem_pitch_roll_rate_hampel_window": dem_rate_hampel_window,
        "dem_pitch_roll_rate_hampel_z": dem_rate_hampel_z,
        "dem_pitch_roll_value_gauss_sigma": dem_value_gauss_sigma,
        "time_ranges_s": json.dumps(time_ranges) if time_ranges else "",
        "config_path": str(Path(args.config).resolve()),
    }

    overwrite = bool(output_cfg.get("overwrite_mission", True))
    compression = output_cfg.get("compression", "gzip") or None
    save_hdf(mp.mission_id, df_out, out_path, attrs, overwrite, compression)

    print(f"[ok] wrote {len(df_out)} patches -> {out_path}")
    if (skipped_edges or skipped_height or skipped_robot or skipped_dupe
            or skipped_contained or skipped_time_subset):
        print("[info] skipped "
              f"(edges={skipped_edges}, height={skipped_height}, robot={skipped_robot}, "
              f"dupes={skipped_dupe}, contained={skipped_contained}, "
              f"time_subset={skipped_time_subset})")


if __name__ == "__main__":
    main()
