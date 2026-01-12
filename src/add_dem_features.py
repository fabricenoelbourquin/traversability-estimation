#!/usr/bin/env python3
"""
Compute DEM-based pitch/roll features along a mission trajectory and append them to parquet.

This script computes per-sample DEM pitch/roll at multiple smoothing scales using the
same conventions as build_patch_dataset.py, then applies optional filters to suppress
spikes. Results are merged into synced_<Hz>Hz_metrics.parquet or synced_<Hz>Hz.parquet.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml
from pyproj import Transformer

from utils.paths import get_paths
from utils.cli import add_mission_arguments, add_hz_argument, resolve_mission_from_args
from utils.quaternion import yaw_from_wxyz
from utils.synced import resolve_synced_parquet


def load_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text()) if p.exists() else {}


def find_lat_lon_cols(df: pd.DataFrame) -> tuple[str, str]:
    cand_lat = [c for c in df.columns if "lat" in c.lower()]
    cand_lon = [c for c in df.columns if "lon" in c.lower()]
    if not cand_lat or not cand_lon:
        raise KeyError("Could not find lat/lon columns in synced parquet.")
    return cand_lat[0], cand_lon[0]


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


def format_dem_scale_label(scale_m: float) -> str:
    return f"{scale_m:.1f}".rstrip("0").rstrip(".").replace(".", "p") + "m"


def gaussian_kernel_2d(sigma_px: float, radius_px: int) -> np.ndarray:
    if sigma_px <= 0.0 or radius_px <= 0:
        return np.array([[1.0]], dtype=np.float64)
    ax = np.arange(-radius_px, radius_px + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma_px ** 2))
    ksum = float(np.sum(kernel))
    if not np.isfinite(ksum) or ksum <= 0.0:
        return np.array([[1.0]], dtype=np.float64)
    return (kernel / ksum).astype(np.float64)


def gaussian_smooth_nan(z: np.ndarray, sigma_px: float) -> np.ndarray:
    """Smooth a DEM with a Gaussian kernel while preserving NaNs."""
    if sigma_px <= 0.0:
        return z.astype(np.float64, copy=True)
    radius_px = int(round(2.0 * sigma_px))
    if radius_px < 1:
        return z.astype(np.float64, copy=True)
    kernel = gaussian_kernel_2d(sigma_px, radius_px).astype(np.float32)
    mask = np.isfinite(z).astype(np.float32)
    z_filled = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise SystemExit("OpenCV (cv2) is required for DEM smoothing.") from e
    num = cv2.filter2D(z_filled, -1, kernel, borderType=cv2.BORDER_REFLECT)
    den = cv2.filter2D(mask, -1, kernel, borderType=cv2.BORDER_REFLECT)
    out = num / np.clip(den, 1e-12, None)
    out[den == 0] = np.nan
    return out.astype(np.float64)


def compute_gradients(z: np.ndarray, transform: rasterio.Affine) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    res_x = abs(transform.a)
    res_y = abs(transform.e)
    dz_drow, dz_dcol = np.gradient(z, res_y, res_x)
    grad_e = dz_dcol
    grad_n = -dz_drow
    grad_mag = np.hypot(grad_e, grad_n)
    grad_theta = np.arctan2(grad_n, grad_e)
    return grad_e, grad_n, grad_mag, grad_theta


def bilinear_sample(grid: np.ndarray, row_f: np.ndarray, col_f: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    out = np.full_like(row_f, np.nan, dtype=np.float64)
    if row_f.size == 0:
        return out
    row0 = np.floor(row_f).astype(np.int64)
    col0 = np.floor(col_f).astype(np.int64)
    row1 = row0 + 1
    col1 = col0 + 1
    valid = (
        np.isfinite(row_f) & np.isfinite(col_f) &
        (row0 >= 0) & (col0 >= 0) &
        (row1 < H) & (col1 < W)
    )
    if not np.any(valid):
        return out
    idx = np.nonzero(valid)[0]
    r0 = row0[idx]; c0 = col0[idx]
    r1 = row1[idx]; c1 = col1[idx]
    dr = row_f[idx] - r0
    dc = col_f[idx] - c0

    w00 = (1.0 - dr) * (1.0 - dc)
    w10 = dr * (1.0 - dc)
    w01 = (1.0 - dr) * dc
    w11 = dr * dc

    g00 = grid[r0, c0]
    g10 = grid[r1, c0]
    g01 = grid[r0, c1]
    g11 = grid[r1, c1]

    vals = np.stack([g00, g10, g01, g11], axis=1)
    weights = np.stack([w00, w10, w01, w11], axis=1)
    mask = np.isfinite(vals)
    weights = weights * mask
    denom = weights.sum(axis=1)
    num = np.nansum(vals * weights, axis=1)
    out_idx = np.full(len(idx), np.nan, dtype=np.float64)
    good = denom > 0.0
    out_idx[good] = num[good] / denom[good]
    out[idx] = out_idx
    return out


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


def compute_yaw_rad(df: pd.DataFrame) -> np.ndarray | None:
    """Extract per-sample yaw [rad] from body->world quaternions (ENU frame)."""
    block = get_quaternion_block(df)
    if block is None:
        return None
    return yaw_from_wxyz(*block)


def rolling_median(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or arr.size == 0:
        return arr.astype(np.float64, copy=True)
    return (
        pd.Series(arr)
        .rolling(window=window, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=np.float64)
    )


def mad_outlier_reject(arr: np.ndarray, z_thresh: float = 3.5) -> np.ndarray:
    out = arr.astype(np.float64, copy=True)
    finite = out[np.isfinite(out)]
    if finite.size < 3:
        return out
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    if not np.isfinite(mad) or mad <= 0.0:
        return out
    z = 0.6745 * (out - med) / mad
    out[np.abs(z) > z_thresh] = np.nan
    return out

def gaussian_kernel_1d(sigma: float, radius: int) -> np.ndarray:
    if sigma <= 0.0 or radius <= 0:
        return np.array([1.0], dtype=np.float64)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    ksum = float(np.sum(kernel))
    if not np.isfinite(ksum) or ksum <= 0.0:
        return np.array([1.0], dtype=np.float64)
    return (kernel / ksum).astype(np.float64)


def gaussian_smooth_1d_nan(arr: np.ndarray, sigma: float, window: int | None = None) -> np.ndarray:
    """Gaussian smooth with NaN-aware normalization."""
    if arr.size == 0 or not np.isfinite(sigma) or sigma <= 0.0:
        return arr.astype(np.float64, copy=True)
    if window is None:
        radius = int(round(3.0 * sigma))
    else:
        radius = int(max(1, window // 2))
    kernel = gaussian_kernel_1d(sigma, radius)
    pad = radius
    values = arr.astype(np.float64, copy=True)
    mask = np.isfinite(values).astype(np.float64)
    values[~np.isfinite(values)] = 0.0

    values_pad = np.pad(values, pad, mode="reflect")
    mask_pad = np.pad(mask, pad, mode="reflect")
    num = np.convolve(values_pad, kernel, mode="valid")
    den = np.convolve(mask_pad, kernel, mode="valid")
    out = num / np.clip(den, 1e-12, None)
    out[den == 0.0] = np.nan
    return out.astype(np.float64)


def rate_limit_angles(angle_deg: np.ndarray, t_s: np.ndarray, max_rate_deg_s: float) -> np.ndarray:
    """Clamp per-sample changes to a maximum rate (deg/s)."""
    out = angle_deg.astype(np.float64, copy=True)
    if out.size == 0 or not np.isfinite(max_rate_deg_s) or max_rate_deg_s <= 0.0:
        return out
    dt = np.diff(t_s, prepend=t_s[0])
    for i in range(1, len(out)):
        if not np.isfinite(out[i]) or not np.isfinite(out[i - 1]) or not np.isfinite(dt[i]):
            continue
        if dt[i] <= 0.0:
            continue
        max_delta = max_rate_deg_s * dt[i]
        delta = out[i] - out[i - 1]
        if delta > max_delta:
            out[i] = out[i - 1] + max_delta
        elif delta < -max_delta:
            out[i] = out[i - 1] - max_delta
    return out


def despike_hampel(angle_deg: np.ndarray, window: int, z_thresh: float) -> np.ndarray:
    """Replace spikes using a Hampel filter (rolling median + MAD)."""
    if window <= 1 or angle_deg.size == 0:
        return angle_deg.astype(np.float64, copy=True)
    series = pd.Series(angle_deg)
    med = series.rolling(window=window, center=True, min_periods=1).median()
    mad = (series - med).abs().rolling(window=window, center=True, min_periods=1).median()
    mad = mad.replace(0.0, np.nan)
    z = 0.6745 * (series - med) / mad
    out = series.to_numpy(dtype=np.float64)
    z_vals = z.to_numpy(dtype=np.float64)
    med_vals = med.to_numpy(dtype=np.float64)
    out[np.abs(z_vals) > z_thresh] = med_vals[np.abs(z_vals) > z_thresh]
    return out


def parse_dem_pitch_roll_cfg(cfg: dict) -> dict:
    dem_cfg = cfg.get("dem_pitch_roll")
    if not isinstance(dem_cfg, dict):
        raise SystemExit("Missing or invalid 'dem_pitch_roll' in metrics config.")

    scales_raw = dem_cfg.get("smooth_scales_m")
    if not isinstance(scales_raw, (list, tuple)):
        raise SystemExit("'dem_pitch_roll.smooth_scales_m' must be a list of meters-based scales.")
    scales_m: list[float] = []
    for val in scales_raw:
        try:
            scale_m = float(val)
        except Exception:
            continue
        if np.isfinite(scale_m) and scale_m > 0.0:
            scales_m.append(scale_m)
    if not scales_m:
        raise SystemExit("'dem_pitch_roll.smooth_scales_m' must contain positive finite values.")

    if "smooth_window" not in dem_cfg:
        raise SystemExit("Missing 'dem_pitch_roll.smooth_window' in metrics config.")
    smooth_window = int(dem_cfg.get("smooth_window"))
    if smooth_window < 1:
        raise SystemExit("'dem_pitch_roll.smooth_window' must be >= 1.")

    if "mad_z_thresh" not in dem_cfg:
        raise SystemExit("Missing 'dem_pitch_roll.mad_z_thresh' in metrics config.")
    mad_z_thresh = float(dem_cfg.get("mad_z_thresh"))
    if not np.isfinite(mad_z_thresh) or mad_z_thresh <= 0.0:
        raise SystemExit("'dem_pitch_roll.mad_z_thresh' must be > 0.")

    return {
        "smooth_scales_m": scales_m,
        "smooth_window": smooth_window,
        "mad_z_thresh": mad_z_thresh,
        "rate_hampel_filter": bool(dem_cfg.get("rate_hampel_filter", False)),
        "rate_hampel_max_rate": float(dem_cfg.get("rate_hampel_max_rate", 0.0)),
        "rate_hampel_window": int(dem_cfg.get("rate_hampel_window", 0)),
        "rate_hampel_z": float(dem_cfg.get("rate_hampel_z", 0.0)),
        "value_gauss_sigma": float(dem_cfg.get("value_gauss_sigma", 0.0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Add DEM pitch/roll features to synced parquet.")
    add_mission_arguments(ap)
    add_hz_argument(ap, help_text="Which synced_<Hz>Hz.parquet to use (default: latest).")
    ap.add_argument("--metrics-config", default="config/metrics.yaml", help="Metrics config YAML (dem_pitch_roll).")
    ap.add_argument("--dataset-config", default="config/dataset.yaml", help="Dataset config YAML (inputs).")
    ap.add_argument("--dem", type=str, default=None, help="Optional DEM override (GeoTIFF).")
    ap.add_argument("--write-to", choices=["metrics", "synced"], default="metrics",
                    help="Append to synced_<Hz>Hz_metrics.parquet or overwrite synced_<Hz>Hz.parquet.")
    args = ap.parse_args()

    P = get_paths()
    mp = resolve_mission_from_args(args, P)
    metrics_cfg = load_yaml(Path(args.metrics_config))
    dem_cfg = parse_dem_pitch_roll_cfg(metrics_cfg)
    scales_m = dem_cfg["smooth_scales_m"]

    dataset_cfg = load_yaml(Path(args.dataset_config))
    input_cfg = dataset_cfg.get("inputs", {})
    dem_path = discover_dem_path(
        mp.maps,
        bool(input_cfg.get("prefer_dem_from_meta", True)),
        args.dem or input_cfg.get("dem_path"),
    )

    synced_path = resolve_synced_parquet(mp.synced, args.hz, prefer_metrics=False)
    if args.hz is None:
        try:
            hz = int(synced_path.stem.split("_")[1].replace("Hz", ""))
        except Exception:
            hz = 10
    else:
        hz = int(args.hz)

    print(f"[load] synced: {synced_path}")
    df = pd.read_parquet(synced_path).sort_values("t").reset_index(drop=True)
    if df.empty:
        raise SystemExit("Synced parquet is empty.")

    lat_col, lon_col = find_lat_lon_cols(df)
    if not lat_col or not lon_col:
        raise SystemExit("Missing lat/lon columns; cannot compute DEM features.")

    yaw = compute_yaw_rad(df)
    if yaw is None:
        raise SystemExit("Quaternion columns not found; cannot compute yaw.")

    print(f"[load] DEM: {dem_path}")
    with rasterio.open(dem_path) as ds:
        z = ds.read(1).astype(np.float64)
        nodata = ds.nodata
        if nodata is not None:
            z[z == nodata] = np.nan
        transform = ds.transform
        dem_crs = ds.crs
    res_m = float(abs(transform.a))

    lat = df[lat_col].to_numpy(dtype=float)
    lon = df[lon_col].to_numpy(dtype=float)
    transformer = Transformer.from_crs("EPSG:4326", dem_crs, always_xy=True)
    east, north = transformer.transform(lon, lat)
    row_f, col_f = world_to_rowcol(transform, east, north)

    t = df["t"].to_numpy(dtype=float)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    out = pd.DataFrame({"t": t})
    for scale_m in scales_m:
        sigma_px = float(scale_m / res_m)
        if not np.isfinite(sigma_px) or sigma_px <= 0.0:
            continue
        z_smooth = gaussian_smooth_nan(z, sigma_px)
        grad_e, grad_n, _, _ = compute_gradients(z_smooth, transform)
        g_e = bilinear_sample(grad_e, row_f, col_f)
        g_n = bilinear_sample(grad_n, row_f, col_f)

        s_parallel = g_e * cos_yaw + g_n * sin_yaw
        s_perp = -g_e * sin_yaw + g_n * cos_yaw

        pitch_deg = np.rad2deg(np.arctan(s_parallel))
        roll_deg = np.rad2deg(np.arctan(s_perp))

        pitch_deg = rolling_median(pitch_deg, dem_cfg["smooth_window"])
        roll_deg = rolling_median(roll_deg, dem_cfg["smooth_window"])
        pitch_deg = mad_outlier_reject(pitch_deg, dem_cfg["mad_z_thresh"])
        roll_deg = mad_outlier_reject(roll_deg, dem_cfg["mad_z_thresh"])

        if dem_cfg["rate_hampel_filter"]:
            pitch_deg = despike_hampel(pitch_deg, dem_cfg["rate_hampel_window"], dem_cfg["rate_hampel_z"])
            roll_deg = despike_hampel(roll_deg, dem_cfg["rate_hampel_window"], dem_cfg["rate_hampel_z"])
            pitch_deg = rate_limit_angles(pitch_deg, t, dem_cfg["rate_hampel_max_rate"])
            roll_deg = rate_limit_angles(roll_deg, t, dem_cfg["rate_hampel_max_rate"])

        if dem_cfg["value_gauss_sigma"] and dem_cfg["value_gauss_sigma"] > 0.0:
            pitch_deg = gaussian_smooth_1d_nan(pitch_deg, dem_cfg["value_gauss_sigma"], None)
            roll_deg = gaussian_smooth_1d_nan(roll_deg, dem_cfg["value_gauss_sigma"], None)

        label = format_dem_scale_label(scale_m)
        out[f"dem_pitch_{label}_deg"] = pitch_deg
        out[f"dem_roll_{label}_deg"] = roll_deg

    # Merge to metrics or synced
    dem_cols = [c for c in out.columns if c != "t"]
    if args.write_to == "metrics":
        metrics_path = mp.synced / f"synced_{hz}Hz_metrics.parquet"
        if metrics_path.exists():
            m = pd.read_parquet(metrics_path)
            drop_cols = [c for c in m.columns if c in dem_cols or c.endswith("_dup")]
            if drop_cols:
                m = m.drop(columns=drop_cols)
            m = m.merge(out, on="t", how="outer", suffixes=("", "_dup"))
            drop_dups = [c for c in m.columns if c.endswith("_dup")]
            m = m.drop(columns=drop_dups)
        else:
            m = out
        m = m.sort_values("t").reset_index(drop=True)
        m.to_parquet(metrics_path, index=False)
        print(f"[save] {metrics_path}  (rows: {len(m)})")
    else:
        df_full = pd.read_parquet(synced_path)
        drop_cols = [c for c in df_full.columns if c in dem_cols or c.endswith("_dup")]
        if drop_cols:
            df_full = df_full.drop(columns=drop_cols)
        merged = df_full.merge(out, on="t", how="left")
        merged.to_parquet(synced_path, index=False)
        print(f"[save] {synced_path}  (rows: {len(merged)})")

    valid_pitch = np.isfinite(out.filter(regex=r"^dem_pitch_").to_numpy()).sum()
    print(f"[stats] valid DEM pitch samples: {valid_pitch}")
    print("[done]")


if __name__ == "__main__":
    main()
