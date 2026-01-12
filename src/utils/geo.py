from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


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


def bilinear_sample(
    grid: np.ndarray,
    row_f: np.ndarray,
    col_f: np.ndarray,
    *,
    nan_policy: str = "weighted",
) -> np.ndarray:
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

    if nan_policy == "strict":
        vals = g00 * w00 + g10 * w10 + g01 * w01 + g11 * w11
        nanmask = np.isnan(g00) | np.isnan(g10) | np.isnan(g01) | np.isnan(g11)
        out_idx = np.full(len(idx), np.nan, dtype=np.float64)
        out_idx[~nanmask] = vals[~nanmask]
        out[idx] = out_idx
        return out

    if nan_policy != "weighted":
        raise ValueError(f"Unknown nan_policy '{nan_policy}'. Use 'weighted' or 'strict'.")

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

    filter_params = dem_cfg.get("filter_params")
    if not isinstance(filter_params, dict):
        filter_params = {}

    if "smooth_window" not in filter_params:
        raise SystemExit("Missing 'dem_pitch_roll.filter_params.smooth_window' in metrics config.")
    smooth_window = int(filter_params.get("smooth_window"))
    if smooth_window < 1:
        raise SystemExit("'dem_pitch_roll.filter_params.smooth_window' must be >= 1.")

    if "mad_z_thresh" not in filter_params:
        raise SystemExit("Missing 'dem_pitch_roll.filter_params.mad_z_thresh' in metrics config.")
    mad_z_thresh = float(filter_params.get("mad_z_thresh"))
    if not np.isfinite(mad_z_thresh) or mad_z_thresh <= 0.0:
        raise SystemExit("'dem_pitch_roll.filter_params.mad_z_thresh' must be > 0.")

    filter_params = {
        "smooth_window": smooth_window,
        "mad_z_thresh": mad_z_thresh,
        "rate_hampel_filter": bool(filter_params.get("rate_hampel_filter", False)),
        "rate_hampel_max_rate": float(filter_params.get("rate_hampel_max_rate", 0.0)),
        "rate_hampel_window": int(filter_params.get("rate_hampel_window", 0)),
        "rate_hampel_z": float(filter_params.get("rate_hampel_z", 0.0)),
        "value_gauss_sigma": float(filter_params.get("value_gauss_sigma", 0.0)),
    }

    return {
        "smooth_scales_m": scales_m,
        "filter_params": filter_params,
    }
