#!/usr/bin/env python3
"""
Build a patch-level HDF5 dataset for one mission.

Features per patch:
  - Heightmap (DEM): single plane-fit slope in E/N/magnitude, gradient orientation, and quadratic-fit curvatures (k1/k2, mean, abs, directional along/ across heading).
  - Robot: mean/percentile for selected metrics, actual speed, commanded speed, pitch [deg];
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
from typing import Iterable

import numpy as np
import pandas as pd
import rasterio
import yaml
from pyproj import Transformer

from utils.paths import get_paths
from utils.cli import add_mission_arguments, add_hz_argument, resolve_mission_from_args
from utils.synced import resolve_synced_parquet, infer_hz_from_path
from utils.filtering import load_metrics_config
from utils.filtering import load_metrics_config


# -------------------------- helpers --------------------------

def load_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text()) if p.exists() else {}


def load_dem_params(metrics_path: Path) -> dict:
    """Load DEM slope params (plane-fitting) from metrics.yaml."""
    cfg = load_yaml(metrics_path)
    params = (cfg.get("dem_slope_params") or {})
    if not params:
        raise SystemExit(f"'dem_slope_params' missing in {metrics_path}")
    size = int(params.get("patch_size_pixels", 9))
    if size % 2 == 0:
        raise SystemExit(f"patch_size_pixels ({size}) must be odd.")
    params["patch_size_pixels"] = size
    params["weighting_method"] = params.get("weighting_method", "gaussian")
    params["weighting_sigma_pixels"] = float(params.get("weighting_sigma_pixels", size / 4.0))
    return params


def find_lat_lon_cols(df: pd.DataFrame) -> tuple[str, str]:
    cand_lat = [c for c in df.columns if "lat" in c.lower()]
    cand_lon = [c for c in df.columns if "lon" in c.lower()]
    if not cand_lat or not cand_lon:
        raise KeyError("Could not find lat/lon columns in synced parquet.")
    return cand_lat[0], cand_lon[0]


def format_patch_size_label(size_m: float) -> str:
    # Format patch size label without trailing zeros
    return f"{size_m:.3f}".rstrip("0").rstrip(".")


def resolve_metric_names(cfg_metrics: dict, repo_root: Path) -> list[str]:
    names = cfg_metrics.get("names") or []
    if names:
        return [str(n) for n in names]
    metrics_cfg = load_yaml(repo_root / "config" / "metrics.yaml")
    return list(metrics_cfg.get("metrics", {}).get("names", []))


def discover_dem_path(map_dir: Path, prefer_meta: bool, explicit: str | None = None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"DEM override not found: {p}")
        return p

    swisstopo = map_dir / "swisstopo"
    search_dirs = [d for d in [swisstopo, map_dir] if d.exists()]

    if prefer_meta:
        for base in search_dirs:
            for meta in sorted(base.glob("**/*.json")):
                try:
                    data = json.loads(meta.read_text())
                except Exception:
                    continue
                dem = data.get("dem") or {}
                cand = dem.get("dem_tif")
                if cand:
                    p = Path(cand)
                    if p.exists():
                        return p

    patterns = ["**/*alti3d*.tif", "**/*dem*.tif"]
    for base in search_dirs:
        for pat in patterns:
            for p in sorted(base.glob(pat)):
                if p.is_file():
                    return p

    raise FileNotFoundError(f"DEM not found under {map_dir}/swisstopo (or parent).")


def world_to_rowcol(transform: rasterio.Affine, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, _, c, _, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
    col_f = (x - c) / a
    row_f = (y - f) / e
    return row_f, col_f


def create_weight_kernel(method: str, size: int, sigma: float) -> np.ndarray:
    if method == "uniform":
        return np.ones((size, size), dtype=np.float64)
    if method == "gaussian":
        if sigma <= 0.0 or not np.isfinite(sigma):
            raise ValueError(f"weighting_sigma_pixels must be > 0 (got {sigma})")
        ax = np.arange(-size // 2 + 1., size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        denom = np.nanmax(kernel)
        if not np.isfinite(denom) or denom <= 0.0:
            raise ValueError("Gaussian kernel normalization failed (denom <= 0).")
        kernel = kernel / denom
        return kernel.astype(np.float64)
    raise ValueError(f"Unknown weighting_method: {method}")


def sample_gradients_planefit(z_grid: np.ndarray,
                              transform: rasterio.Affine,
                              row_f: np.ndarray,
                              col_f: np.ndarray,
                              config: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate DEM gradients (p=∂z/∂E, q=∂z/∂N) at arbitrary fractional row/col
    positions by fitting a plane to a local patch.
    Mirrors the logic in add_dem_longlat_slope.py
    Arguments:
        z_grid: (H,W) DEM grid
        transform: rasterio Affine transform for the DEM, maps pixel coords ↔ world coords.
        row_f: (N,) fractional row positions
        col_f: (N,) fractional column positions
        config: dict with plane-fitting parameters
    Returns:
        p_s: (N,) eastward slopes
        q_s: (N,) northward slopes
    """
    H, W_img = z_grid.shape
    N = len(row_f)
    p_s = np.full(N, np.nan, dtype=np.float64)
    q_s = np.full(N, np.nan, dtype=np.float64)

    size = int(config["patch_size_pixels"])
    half_size = size // 2

    kernel = create_weight_kernel(
        config["weighting_method"],
        size,
        float(config.get("weighting_sigma_pixels", 1.25))
    )
    weights = kernel.ravel().astype(np.float64)
    if not np.isfinite(weights).all():
        raise ValueError("Weight kernel contains non-finite values.")

    # transform.a ~ pixel width in meters (east), transform.e ~ pixel height in meters (north, negative)
    a_res, e_res = transform.a, transform.e
    patch_rows_rel = np.arange(-half_size, half_size + 1)
    patch_cols_rel = np.arange(-half_size, half_size + 1)
    xx_rel_pix, yy_rel_pix = np.meshgrid(patch_cols_rel, patch_rows_rel)
    dx_meters = xx_rel_pix.ravel() * a_res
    dy_meters = yy_rel_pix.ravel() * e_res
    # Design matrix for plane fitting: z = p*dx + q*dy + c
    A = np.stack([dx_meters, dy_meters, np.ones(size * size, dtype=np.float64)], axis=1)
    if not np.isfinite(A).all():
        raise ValueError("Design matrix A has non-finite entries.")
    A_w = weights[:, None] * A
    A_w_pinv = np.linalg.pinv(A_w)

    row_i = np.round(row_f).astype(np.int64).ravel()
    col_i = np.round(col_f).astype(np.int64).ravel()

    for i in range(N):
        r_center = int(row_i[i])
        c_center = int(col_i[i])

        if (r_center < half_size or r_center >= H - half_size or
                c_center < half_size or c_center >= W_img - half_size):
            continue

        z_patch = z_grid[
            r_center - half_size: r_center + half_size + 1,
            c_center - half_size: c_center + half_size + 1
        ]
        b = z_patch.ravel()
        if np.isnan(b).any():
            continue

        b_w = weights * b
        C = A_w_pinv @ b_w
        p_s[i] = C[0]
        q_s[i] = C[1]

    return p_s, q_s


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


def compute_gradients(z: np.ndarray, transform: rasterio.Affine) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    res_x = abs(transform.a)
    res_y = abs(transform.e)
    dz_drow, dz_dcol = np.gradient(z, res_y, res_x)  # row -> south, col -> east
    grad_e = dz_dcol
    grad_n = -dz_drow  # flip sign so positive is northward
    grad_mag = np.hypot(grad_e, grad_n)
    grad_theta = np.arctan2(grad_n, grad_e)
    return grad_e, grad_n, grad_mag, grad_theta


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
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return (np.nan, np.nan, np.nan)
    lower = float(np.nanpercentile(finite, 5.0))
    upper = float(np.nanpercentile(finite, 95.0))
    return (float(np.nanmean(finite)), lower, upper)


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
            norms = np.linalg.norm(sub, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            q = sub / norms
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


def get_quaternion_block(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    for cols in (("qw_WB", "qx_WB", "qy_WB", "qz_WB"), ("qw", "qx", "qy", "qz")):
        if all(c in df.columns for c in cols):
            return (
                df[cols[0]].to_numpy(dtype=np.float64),
                df[cols[1]].to_numpy(dtype=np.float64),
                df[cols[2]].to_numpy(dtype=np.float64),
                df[cols[3]].to_numpy(dtype=np.float64),
            )
    return None


def normalize_quat_arrays(qw: np.ndarray, qx: np.ndarray, qy: np.ndarray, qz: np.ndarray
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    n[n == 0.0] = 1.0
    return qw / n, qx / n, qy / n, qz / n


def compute_pitch_deg(df: pd.DataFrame) -> np.ndarray | None:
    """
    Compute pitch [deg] from body->world quaternions.
    Matches visualization (nose-up positive) by flipping the sign.
    """
    block = get_quaternion_block(df)
    if block is None:
        return None
    qw, qx, qy, qz = normalize_quat_arrays(*block)

    xx = qx * qx; yy = qy * qy; zz = qz * qz
    xy = qx * qy; xz = qx * qz; yz = qy * qz
    wx = qw * qx; wy = qw * qy; wz = qw * qz

    r00 = 1.0 - 2.0 * (yy + zz)
    r10 = 2.0 * (xy + wz)
    r20 = 2.0 * (xz - wy)

    pitch = np.arctan2(-r20, np.clip(np.sqrt(r00 * r00 + r10 * r10), 1e-12, None))
    pitch_deg = np.rad2deg(pitch)
    return -pitch_deg  # flip axis so nose-up is positive


def yaw_from_quaternion(qw: float, qx: float, qy: float, qz: float) -> float:
    """Extract yaw [rad] from a single quaternion (body→world), ENU frame."""
    if not all(np.isfinite([qw, qx, qy, qz])):
        return float("nan")
    qw, qx, qy, qz = normalize_quat_arrays(
        np.array([qw], dtype=np.float64),
        np.array([qx], dtype=np.float64),
        np.array([qy], dtype=np.float64),
        np.array([qz], dtype=np.float64),
    )
    qw = float(qw[0]); qx = float(qx[0]); qy = float(qy[0]); qz = float(qz[0])
    xx = qx * qx; yy = qy * qy; zz = qz * qz
    xy = qx * qy; wz = qw * qz
    r00 = 1.0 - 2.0 * (yy + zz)
    r10 = 2.0 * (xy + wz)
    return float(math.atan2(r10, r00))


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
    # check for valid inputs
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
    # valid samples
    valid = np.isfinite(power) & np.isfinite(t) & np.isfinite(dist) & np.isfinite(v_cmd)
    if not np.any(valid):
        return (np.nan, np.nan)

    low_cmd = v_cmd < max(min_cmd_speed, 0.0)

    def _expand_mask_by_time(base_mask: np.ndarray, times: np.ndarray, padding_s: float) -> np.ndarray:
        if padding_s <= 0.0 or not np.any(base_mask):
            return base_mask
        out = base_mask.copy()
        idx = np.nonzero(base_mask)[0]
        for i in idx:
            t0 = times[i]
            j = i
            while j >= 0 and (t0 - times[j]) <= padding_s:
                out[j] = True
                j -= 1
            j = i + 1
            n = len(times)
            while j < n and (times[j] - t0) <= padding_s:
                out[j] = True
                j += 1
        return out

    if min_cmd_pad_s > 0.0:
        low_cmd = _expand_mask_by_time(low_cmd, t, min_cmd_pad_s)

    turn_only = np.zeros_like(valid)
    if turn_min_wz > 0.0 and "w_cmd_z" in df_patch:
        w_cmd = np.abs(df_patch["w_cmd_z"].to_numpy(dtype=float))
        lin_thresh = turn_lin_thresh if (turn_lin_thresh is not None and np.isfinite(turn_lin_thresh)) else min_cmd_speed
        turn_only = valid & (w_cmd >= turn_min_wz) & (v_cmd < lin_thresh)
        if turn_pad_s > 0.0:
            turn_only = _expand_mask_by_time(turn_only, t, turn_pad_s)

    valid &= ~(low_cmd | turn_only)
    if not np.any(valid):
        return (np.nan, np.nan)

    # Integrate energy with forward differences; ignore negative dt
    dt = np.diff(t, prepend=t[0])
    dt = np.clip(dt, 0.0, None)
    dt[~np.isfinite(dt)] = 0.0

    def _cot_from_mask(mask: np.ndarray) -> float:
        if not np.any(mask):
            return float("nan")
        energy = np.nansum(power[mask] * dt[mask])
        dist_masked = dist[mask]
        finite_d = dist_masked[np.isfinite(dist_masked)]
        if finite_d.size == 0:
            return float("nan")
        distance = float(np.nanmax(finite_d) - np.nanmin(finite_d))
        if distance <= 0.0:
            return float("nan")
        return energy / (mass * gravity * distance)

    cot_all = _cot_from_mask(valid)

    cot_trimmed = float("nan")
    if np.any(valid):
        try:
            lo, hi = np.nanpercentile(power[valid], [5.0, 95.0])
        except Exception:
            lo = hi = np.nan
        trimmed = valid
        if np.isfinite(lo) and np.isfinite(hi):
            trimmed = valid & (power >= lo) & (power <= hi)
        cot_trimmed = _cot_from_mask(trimmed)

    return (cot_all, cot_trimmed)


def aggregate_robot_patch(df_patch: pd.DataFrame,
                          metric_names: Iterable[str],
                          include_speed: bool,
                          include_cmd_speed: bool,
                          cot_cfg: dict[str, float] | None) -> dict:
    out: dict[str, float] = {}

    for m in metric_names:
        mu, mn, mx = summarize_metric(df_patch, m)
        out[f"metric_{m}_mean"] = mu
        out[f"metric_{m}_min"] = mn
        out[f"metric_{m}_max"] = mx

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
            out["speed_mean"], out["speed_min"], out["speed_max"] = nan_stats(speed)
        else:
            out["speed_mean"] = out["speed_min"] = out["speed_max"] = np.nan

    if include_cmd_speed:
        if "v_cmd" in df_patch:
            vcmd = df_patch["v_cmd"].to_numpy(dtype=float)
        elif {"v_cmd_x", "v_cmd_y"}.issubset(df_patch.columns):
            vcmd = np.hypot(df_patch["v_cmd_x"], df_patch["v_cmd_y"])
        else:
            vcmd = np.array([], dtype=float)
        if vcmd.size:
            out["v_cmd_mean"], out["v_cmd_min"], out["v_cmd_max"] = nan_stats(vcmd)
        else:
            out["v_cmd_mean"] = out["v_cmd_min"] = out["v_cmd_max"] = np.nan
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
        out["pitch_deg_mean"], out["pitch_deg_min"], out["pitch_deg_max"] = nan_stats(pitch)
    else:
        out["pitch_deg_mean"] = out["pitch_deg_min"] = out["pitch_deg_max"] = np.nan

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
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))
    patch_cfg = cfg.get("patch", {})
    input_cfg = cfg.get("inputs", {})
    metrics_cfg = cfg.get("metrics", {})
    output_cfg = cfg.get("output", {})
    swiss_cfg = cfg.get("swissimage", {})
    selection_cfg = cfg.get("selection", {})

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
    dem_path = discover_dem_path(mp.maps, bool(input_cfg.get("prefer_dem_from_meta", True)), args.dem or input_cfg.get("dem_path"))
    print(f"[load] DEM: {dem_path}")
    with rasterio.open(dem_path) as ds:
        z = ds.read(1).astype(np.float64)
        nodata = ds.nodata
        if nodata is not None:
            z[z == nodata] = np.nan
        transform = ds.transform
        dem_crs = ds.crs

    res_m = float(abs(transform.a))

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
    lat_col, lon_col = find_lat_lon_cols(df)
    lat = df[lat_col].to_numpy(dtype=float)
    lon = df[lon_col].to_numpy(dtype=float)
    transformer = Transformer.from_crs("EPSG:4326", dem_crs, always_xy=True)
    east, north = transformer.transform(lon, lat)
    df["easting_m"] = east
    df["northing_m"] = north
    # Build segments (stride restarts per range) using the filtered df with coordinates
    if time_ranges:
        segments = []
        for start, end in time_ranges:
            seg = df[(df["t_rel_raw"] >= start) & (df["t_rel_raw"] <= end)].copy()
            if len(seg):
                segments.append(seg)
    else:
        segments = [df]

    half_px = max(1, int(round((patch_size_m / 2.0) / res_m)))
    H, W = z.shape

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
        east_seg = seg["easting_m"].to_numpy(dtype=float)
        north_seg = seg["northing_m"].to_numpy(dtype=float)
        lat_seg = seg[lat_col].to_numpy(dtype=float)
        lon_seg = seg[lon_col].to_numpy(dtype=float)
        t_seg = seg["t"].to_numpy(dtype=float)
        distances = seg["dist_m"].to_numpy(dtype=float) if "dist_m" in seg else np.full(len(seg), np.nan)

        valid_mask = np.isfinite(east_seg) & np.isfinite(north_seg) & np.isfinite(distances)
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
            cx_e = east_seg[ci]; cy_n = north_seg[ci]
            cx_lat = lat_seg[ci]; cx_lon = lon_seg[ci]
            row_c, col_c = world_to_rowcol(transform, np.array([cx_e]), np.array([cy_n]))
            r = int(round(row_c[0])); c = int(round(col_c[0]))
            r0 = r - half_px; r1 = r + half_px + 1
            c0 = c - half_px; c1 = c + half_px + 1
            # ensure patch fully inside DEM
            if (r0 < 0 or c0 < 0 or r1 > H or c1 > W):
                skipped_edges += 1
                continue

            rows_grid = np.arange(r0, r1)
            cols_grid = np.arange(c0, c1)
            rr_abs, cc_abs = np.meshgrid(rows_grid, cols_grid, indexing="ij")
            z_patch = z[r0:r1, c0:c1]

            valid_cells = np.isfinite(z_patch)
            valid_frac = float(valid_cells.mean()) if valid_cells.size else 0.0
            if valid_frac < min_height_frac:
                skipped_height += 1
                continue

            # Fit a single plane over the patch to estimate east/north slopes.
            slope_e, slope_n = fit_plane_to_patch(z_patch, rr_abs, cc_abs, r, c, transform.a, transform.e)
            slope_mag = math.hypot(slope_e, slope_n)
            grad_orient = float(math.atan2(slope_n, slope_e)) if np.isfinite(slope_e) and np.isfinite(slope_n) else float("nan")
            in_patch = (
                (np.abs(east_seg - cx_e) <= (patch_size_m / 2.0)) &
                (np.abs(north_seg - cy_n) <= (patch_size_m / 2.0)) &
                np.isfinite(east_seg) & np.isfinite(north_seg)
            )
            df_patch = seg.loc[in_patch].copy()
            if len(df_patch) < min_robot_samples:
                skipped_robot += 1
                continue

            robot_feats = aggregate_robot_patch(df_patch, metric_names, include_speed, include_cmd, cot_cfg if include_cmd else None)
            min_dist_m = float(patch_size_m)  # require travel at least the patch size
            if np.isfinite(robot_feats.get("distance_traveled_m", np.nan)) and robot_feats["distance_traveled_m"] < min_dist_m:
                skipped_robot += 1
                continue
            robot_feats = aggregate_robot_patch(df_patch, metric_names, include_speed, include_cmd, cot_cfg if include_cmd else None)
            min_dist_m = float(patch_size_m)  # require travel at least the patch size
            if np.isfinite(robot_feats.get("distance_traveled_m", np.nan)) and robot_feats["distance_traveled_m"] < min_dist_m:
                skipped_robot += 1
                continue

            quad_coeffs = fit_quadratic_patch(z_patch, rr_abs, cc_abs, r, c, transform.a, transform.e)
            k1 = k2 = mean_curv = abs_curv = float("nan")
            curv_heading = curv_cross = float("nan")
            yaw_rad = float("nan")
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

                    yaw_rad = yaw_from_quaternion(robot_feats["bearing_qw"], robot_feats["bearing_qx"],
                                                  robot_feats["bearing_qy"], robot_feats["bearing_qz"])
                    if np.isfinite(yaw_rad):
                        u = math.cos(yaw_rad); v = math.sin(yaw_rad)
                        denom_dir = E * u * u + 2.0 * F * u * v + G * v * v
                        if denom_dir != 0.0 and np.isfinite(denom_dir):
                            curv_heading = (L * u * u + 2.0 * M * u * v + N * v * v) / denom_dir
                        u_perp = -v; v_perp = u
                        denom_perp = E * u_perp * u_perp + 2.0 * F * u_perp * v_perp + G * v_perp * v_perp
                        if denom_perp != 0.0 and np.isfinite(denom_perp):
                            curv_cross = (L * u_perp * u_perp + 2.0 * M * u_perp * v_perp + N * v_perp * v_perp) / denom_perp

            side_m = len(rows_grid) * res_m
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
                "height_valid_fraction": valid_frac,
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
        "dem_path": str(dem_path),
        "include_dino_embeddings": include_dino,
        "time_ranges_s": json.dumps(time_ranges) if time_ranges else "",
        "config_path": str(Path(args.config).resolve()),
    }

    overwrite = bool(output_cfg.get("overwrite_mission", True))
    compression = output_cfg.get("compression", "gzip") or None
    save_hdf(mp.mission_id, df_out, out_path, attrs, overwrite, compression)

    print(f"[ok] wrote {len(df_out)} patches -> {out_path}")
    if skipped_edges or skipped_height or skipped_robot or skipped_dupe or skipped_contained or skipped_time_subset:
        print(f"[info] skipped (edges={skipped_edges}, height={skipped_height}, robot={skipped_robot}, dupes={skipped_dupe}, contained={skipped_contained}, time_subset={skipped_time_subset})")


if __name__ == "__main__":
    main()
