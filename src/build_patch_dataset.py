#!/usr/bin/env python3
"""
Build a patch-level HDF5 dataset for one mission.

Features per patch:
  - Heightmap (DEM): mean/min/max slope in E, N, and magnitude, plus mean gradient orientation.
  - Robot: mean/min/max for selected metrics, actual speed, commanded speed;
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
    return (float(np.nanmean(finite)), float(np.nanmin(finite)), float(np.nanmax(finite)))


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


def summarize_metric(df: pd.DataFrame, col: str) -> tuple[float, float, float]:
    if col not in df.columns:
        return (np.nan, np.nan, np.nan)
    arr = df[col].to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (np.nan, np.nan, np.nan)
    return (float(np.nanmean(arr)), float(np.nanmin(arr)), float(np.nanmax(arr)))


def aggregate_robot_patch(df_patch: pd.DataFrame,
                          metric_names: Iterable[str],
                          include_speed: bool,
                          include_cmd_speed: bool) -> dict:
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
    dem_params = load_dem_params(Path(P["REPO_ROOT"]) / "config" / "metrics.yaml")
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
    pf_half = dem_params["patch_size_pixels"] // 2
    H, W = z.shape

    rows = []
    skipped_edges = 0
    skipped_height = 0
    skipped_robot = 0
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
            # ensure enough margin for plane-fitting window
            if (r0 - pf_half < 0 or c0 - pf_half < 0 or
                    r1 + pf_half > H or c1 + pf_half > W):
                skipped_edges += 1
                continue

            rows_grid = np.arange(r0, r1)
            cols_grid = np.arange(c0, c1)
            rr, cc = np.meshgrid(rows_grid, cols_grid, indexing="ij")
            p_patch, q_patch = sample_gradients_planefit(z, transform, rr.ravel(), cc.ravel(), dem_params)
            p_patch = p_patch.reshape(rr.shape)
            q_patch = q_patch.reshape(rr.shape)
            grad_mag_patch = np.hypot(p_patch, q_patch)
            grad_theta_patch = np.arctan2(q_patch, p_patch)

            valid_cells = np.isfinite(grad_mag_patch)
            valid_frac = float(valid_cells.mean()) if valid_cells.size else 0.0
            if valid_frac < min_height_frac:
                skipped_height += 1
                continue

            slope_e_mean, slope_e_min, slope_e_max = nan_stats(p_patch)
            slope_n_mean, slope_n_min, slope_n_max = nan_stats(q_patch)
            slope_m_mean, slope_m_min, slope_m_max = nan_stats(grad_mag_patch)
            grad_orient_mean = circular_mean(grad_theta_patch)

            in_patch = (
                (np.abs(east_seg - cx_e) <= (patch_size_m / 2.0)) &
                (np.abs(north_seg - cy_n) <= (patch_size_m / 2.0)) &
                np.isfinite(east_seg) & np.isfinite(north_seg)
            )
            df_patch = seg.loc[in_patch].copy()
            if len(df_patch) < min_robot_samples:
                skipped_robot += 1
                continue

            robot_feats = aggregate_robot_patch(df_patch, metric_names, include_speed, include_cmd)

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
                "slope_e_mean": slope_e_mean,
                "slope_e_min": slope_e_min,
                "slope_e_max": slope_e_max,
                "slope_n_mean": slope_n_mean,
                "slope_n_min": slope_n_min,
                "slope_n_max": slope_n_max,
                "slope_mag_mean": slope_m_mean,
                "slope_mag_min": slope_m_min,
                "slope_mag_max": slope_m_max,
                "grad_orientation_mean": grad_orient_mean,
                "samples": int(len(df_patch)),
            }
            row.update(robot_feats)
            rows.append(row)
            patch_idx += 1

    if not rows:
        raise SystemExit("No patches produced (all skipped).")

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
    if skipped_edges or skipped_height or skipped_robot:
        print(f"[info] skipped (edges={skipped_edges}, height={skipped_height}, robot={skipped_robot})")


if __name__ == "__main__":
    main()
