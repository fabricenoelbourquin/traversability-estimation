#!/usr/bin/env python3
"""
Compare DEM-derived pitch/roll (multi-scale, smoothed) against quaternion pitch/roll for one mission.

This is a lightweight alignment check to validate sign and axis conventions:
  - DEM pitch/roll are computed with the same logic as build_patch_dataset.py
  - Robot pitch/roll are computed from q_WB (body->world, ENU, body FLU)
  - Optional yaw-from-velocity is plotted for a quick sanity check
  - Optional spike suppression reduces implausible DEM swings for analysis
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio
from pyproj import Transformer

from utils.paths import get_paths
from utils.cli import add_mission_arguments, add_hz_argument, resolve_mission_from_args
from utils.synced import resolve_synced_parquet
from utils.quaternion import euler_zyx_from_wxyz

from add_dem_features import (
    load_yaml,
    parse_dem_pitch_roll_cfg,
    find_lat_lon_cols,
    discover_dem_path,
    world_to_rowcol,
    gaussian_smooth_nan,
    compute_gradients,
    bilinear_sample,
    compute_yaw_rad,
    rolling_median,
    mad_outlier_reject,
    format_dem_scale_label,
)
from build_patch_dataset import (
    compute_pitch_deg,
    get_quaternion_block,
)


def compute_roll_deg(df: pd.DataFrame) -> np.ndarray | None:
    """
    Compute roll [deg] from body->world quaternions (ENU, body FLU).
    Positive roll means left side up.
    """
    block = get_quaternion_block(df)
    if block is None:
        return None
    _, _, roll_deg = euler_zyx_from_wxyz(*block, degrees=True)
    return roll_deg


def wrap_angle_rad(a: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi] for stable difference plots."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def gaussian_kernel_1d(sigma: float, radius: int) -> np.ndarray:
    """Build a normalized 1D Gaussian kernel."""
    if sigma <= 0.0 or radius <= 0:
        return np.array([1.0], dtype=np.float64)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    ksum = float(np.sum(kernel))
    if not np.isfinite(ksum) or ksum <= 0.0:
        return np.array([1.0], dtype=np.float64)
    return (kernel / ksum).astype(np.float64)


def gaussian_smooth_1d_nan(arr: np.ndarray, sigma: float, window: int | None = None) -> np.ndarray:
    """
    Gaussian smooth with NaN-aware normalization.
    This is applied to values (not just plotting) when enabled.
    """
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
    """
    Clamp per-sample changes to a maximum rate (deg/s) to suppress DEM spikes.
    This preserves larger changes if they occur over longer time spans.
    """
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
    """
    Remove isolated spikes via Hampel filter (rolling median + MAD).
    This keeps trends but suppresses short, unrealistic swings.
    """
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


def compute_yaw_from_velocity(df: pd.DataFrame, coord_e: str | None, coord_n: str | None) -> np.ndarray | None:
    """
    Estimate heading from velocity or position derivatives.
    Returns yaw [rad] in ENU (east=0, CCW positive).
    """
    if {"vx", "vy"}.issubset(df.columns):
        vx = df["vx"].to_numpy(dtype=float)
        vy = df["vy"].to_numpy(dtype=float)
        return np.arctan2(vy, vx)
    if coord_e and coord_n and "t" in df.columns:
        e = df[coord_e].to_numpy(dtype=float)
        n = df[coord_n].to_numpy(dtype=float)
        t = df["t"].to_numpy(dtype=float)
        dt = np.gradient(t)
        dt[dt == 0.0] = np.nan
        ve = np.gradient(e) / dt
        vn = np.gradient(n) / dt
        return np.arctan2(vn, ve)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare DEM pitch/roll vs quaternion pitch/roll with multi-scale smoothing.")
    add_mission_arguments(ap, required=False)
    add_hz_argument(ap, help_text="Which synced_<Hz>Hz.parquet to use (default: latest).")
    ap.add_argument("--metrics-config", default="config/metrics.yaml", help="Metrics config YAML (dem_pitch_roll).")
    ap.add_argument("--dataset-config", default="config/dataset.yaml", help="Dataset config YAML (inputs).")
    ap.add_argument("--dem", type=str, default=None, help="Optional DEM override (GeoTIFF).")
    ap.add_argument("--t-start", type=float, default=None, help="Start time [s] relative to mission start.")
    ap.add_argument("--t-end", type=float, default=None, help="End time [s] relative to mission start.")
    ap.add_argument("--rate-hampel-filter", action="store_true", default=None,
                    help="Apply Hampel + rate-limit spike suppression to DEM pitch/roll before plotting.")
    ap.add_argument("--rate-hampel-max-rate", type=float, default=None,
                    help="Max pitch/roll rate [deg/s] when realistic filter is on.")
    ap.add_argument("--rate-hampel-window", type=int, default=None,
                    help="Hampel filter window (samples) for realistic filter.")
    ap.add_argument("--rate-hampel-z", type=float, default=None,
                    help="Hampel MAD z-threshold for realistic filter.")
    ap.add_argument("--value-gauss-sigma", type=float, default=None,
                    help="Gaussian smoothing sigma (samples) applied to values before stats/plot.")
    ap.add_argument("--out", type=str, default=None, help="Optional output PNG path.")
    args = ap.parse_args()

    if not args.mission and not args.mission_id:
        args.mission = "ETH-1"

    P = get_paths()
    mp = resolve_mission_from_args(args, P)
    metrics_cfg = load_yaml(Path(args.metrics_config))
    dem_cfg = parse_dem_pitch_roll_cfg(metrics_cfg)
    dataset_cfg = load_yaml(Path(args.dataset_config))
    scales_m = dem_cfg["smooth_scales_m"]
    smooth_window = dem_cfg["smooth_window"]
    mad_z_thresh = dem_cfg["mad_z_thresh"]
    if args.rate_hampel_filter is None:
        args.rate_hampel_filter = dem_cfg["rate_hampel_filter"]
    if args.rate_hampel_max_rate is None:
        args.rate_hampel_max_rate = dem_cfg["rate_hampel_max_rate"]
    if args.rate_hampel_window is None:
        args.rate_hampel_window = dem_cfg["rate_hampel_window"]
    if args.rate_hampel_z is None:
        args.rate_hampel_z = dem_cfg["rate_hampel_z"]
    if args.value_gauss_sigma is None:
        args.value_gauss_sigma = dem_cfg["value_gauss_sigma"]

    synced_path = resolve_synced_parquet(mp.synced, args.hz, prefer_metrics=False)
    df = pd.read_parquet(synced_path).sort_values("t").reset_index(drop=True)

    if df.empty:
        raise SystemExit("Synced parquet is empty.")

    t0 = float(df["t"].iloc[0])
    t_rel_series = df["t"] - t0
    if args.t_start is not None:
        df = df[t_rel_series >= float(args.t_start)].copy()
    if args.t_end is not None:
        df = df[t_rel_series <= float(args.t_end)].copy()
    df = df.reset_index(drop=True)
    if df.empty:
        raise SystemExit("No data after applying time filters.")

    lat_col, lon_col = find_lat_lon_cols(df)

    input_cfg = dataset_cfg.get("inputs", {})
    dem_path = discover_dem_path(
        mp.maps,
        bool(input_cfg.get("prefer_dem_from_meta", True)),
        args.dem or input_cfg.get("dem_path"),
    )
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
    df["easting_m"] = east
    df["northing_m"] = north

    yaw_quat = compute_yaw_rad(df)
    if yaw_quat is None:
        raise SystemExit("Quaternion columns not found; cannot compute yaw.")
    pitch_quat = compute_pitch_deg(df)
    roll_quat = compute_roll_deg(df)
    if pitch_quat is None or roll_quat is None:
        raise SystemExit("Quaternion columns not found; cannot compute pitch/roll.")

    t_rel = df["t"].to_numpy(dtype=float) - float(df["t"].iloc[0])
    cos_yaw = np.cos(yaw_quat)
    sin_yaw = np.sin(yaw_quat)
    row_f, col_f = world_to_rowcol(transform, east, north)

    dem_pitch: dict[str, np.ndarray] = {}
    dem_roll: dict[str, np.ndarray] = {}
    for scale_m in scales_m:
        sigma_px = scale_m / res_m
        z_smooth = gaussian_smooth_nan(z, sigma_px)
        grad_e, grad_n, _, _ = compute_gradients(z_smooth, transform)
        g_e = bilinear_sample(grad_e, row_f, col_f)
        g_n = bilinear_sample(grad_n, row_f, col_f)

        s_parallel = g_e * cos_yaw + g_n * sin_yaw
        s_perp = -g_e * sin_yaw + g_n * cos_yaw

        pitch_deg = np.rad2deg(np.arctan(s_parallel))
        roll_deg = np.rad2deg(np.arctan(s_perp))

        pitch_deg = rolling_median(pitch_deg, smooth_window)
        roll_deg = rolling_median(roll_deg, smooth_window)
        pitch_deg = mad_outlier_reject(pitch_deg, mad_z_thresh)
        roll_deg = mad_outlier_reject(roll_deg, mad_z_thresh)
        if args.rate_hampel_filter:
            pitch_deg = despike_hampel(pitch_deg, args.rate_hampel_window, args.rate_hampel_z)
            roll_deg = despike_hampel(roll_deg, args.rate_hampel_window, args.rate_hampel_z)
            pitch_deg = rate_limit_angles(pitch_deg, t_rel, args.rate_hampel_max_rate)
            roll_deg = rate_limit_angles(roll_deg, t_rel, args.rate_hampel_max_rate)

        label = format_dem_scale_label(scale_m)
        dem_pitch[label] = pitch_deg
        dem_roll[label] = roll_deg

    if args.value_gauss_sigma and args.value_gauss_sigma > 0.0:
        pitch_quat = gaussian_smooth_1d_nan(pitch_quat, args.value_gauss_sigma, None)
        roll_quat = gaussian_smooth_1d_nan(roll_quat, args.value_gauss_sigma, None)
        for label in dem_pitch:
            dem_pitch[label] = gaussian_smooth_1d_nan(dem_pitch[label], args.value_gauss_sigma, None)
            dem_roll[label] = gaussian_smooth_1d_nan(dem_roll[label], args.value_gauss_sigma, None)

    # Stats summary
    def stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
        mask = np.isfinite(a) & np.isfinite(b)
        if not np.any(mask):
            return (np.nan, np.nan, np.nan)
        d = a[mask] - b[mask]
        rmse = float(np.sqrt(np.nanmean(d * d)))
        bias = float(np.nanmean(d))
        corr = float(np.corrcoef(a[mask], b[mask])[0, 1]) if np.sum(mask) > 2 else np.nan
        return rmse, bias, corr

    print("[alignment] DEM vs quat pitch/roll")
    for label in dem_pitch:
        rmse_p, bias_p, corr_p = stats(dem_pitch[label], pitch_quat)
        rmse_r, bias_r, corr_r = stats(dem_roll[label], roll_quat)
        print(f"  scale {label}: pitch rmse={rmse_p:.3f}°, bias={bias_p:+.3f}°, corr={corr_p:.3f}")
        print(f"               : roll  rmse={rmse_r:.3f}°, bias={bias_r:+.3f}°, corr={corr_r:.3f}")

    # Plot pitch/roll and differences
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex="col")
    ax_pitch = axes[0, 0]
    ax_pitch_diff = axes[0, 1]
    ax_roll = axes[1, 0]
    ax_roll_diff = axes[1, 1]

    ax_pitch.plot(t_rel, pitch_quat, color="k", linewidth=1.2, label="quat pitch")
    ax_roll.plot(t_rel, roll_quat, color="k", linewidth=1.2, label="quat roll")

    for label in dem_pitch:
        dem_p = dem_pitch[label]
        dem_r = dem_roll[label]
        ax_pitch.plot(t_rel, dem_p, linewidth=1.0, label=f"dem pitch {label}")
        ax_roll.plot(t_rel, dem_r, linewidth=1.0, label=f"dem roll {label}")

        diff_p = dem_p - pitch_quat
        diff_r = dem_r - roll_quat
        ax_pitch_diff.plot(t_rel, diff_p, linewidth=1.0, label=f"{label}")
        ax_roll_diff.plot(t_rel, diff_r, linewidth=1.0, label=f"{label}")

    ax_pitch.set_ylabel("Pitch [deg]")
    ax_pitch.set_title("Pitch: quat vs DEM")
    ax_pitch.grid(True, alpha=0.3)
    ax_pitch.legend(ncol=2, fontsize=8)

    ax_pitch_diff.set_ylabel("Pitch diff [deg]")
    ax_pitch_diff.set_title("Pitch: DEM - quat")
    ax_pitch_diff.grid(True, alpha=0.3)
    ax_pitch_diff.legend(ncol=2, fontsize=8)

    ax_roll.set_ylabel("Roll [deg]")
    ax_roll.set_xlabel("Time since start [s]")
    ax_roll.set_title("Roll: quat vs DEM")
    ax_roll.grid(True, alpha=0.3)
    ax_roll.legend(ncol=2, fontsize=8)

    ax_roll_diff.set_ylabel("Roll diff [deg]")
    ax_roll_diff.set_xlabel("Time since start [s]")
    ax_roll_diff.set_title("Roll: DEM - quat")
    ax_roll_diff.grid(True, alpha=0.3)
    ax_roll_diff.legend(ncol=2, fontsize=8)

    fig.tight_layout()

    if args.out:
        out_path = Path(args.out)
    else:
        out_dir = Path(P["REPO_ROOT"]) / "reports" / mp.display
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{mp.display}_dem_vs_quat_pitch_roll_multiscale.png"
    fig.savefig(out_path, dpi=150)
    print(f"[saved] {out_path}")

    # Optional yaw sanity check
    yaw_vel = compute_yaw_from_velocity(df, "easting_m", "northing_m")
    if yaw_vel is not None:
        yaw_diff = wrap_angle_rad(yaw_vel - yaw_quat)
        yaw_rmse = float(np.sqrt(np.nanmean(yaw_diff * yaw_diff)))
        yaw_bias = float(np.nanmean(yaw_diff))
        print(f"[yaw] vel vs quat rmse={np.degrees(yaw_rmse):.2f}°, bias={np.degrees(yaw_bias):+.2f}°")

        fig_yaw, ax = plt.subplots(1, 1, figsize=(12, 3))
        ax.plot(t_rel, np.rad2deg(yaw_quat), label="yaw quat", linewidth=1.0)
        ax.plot(t_rel, np.rad2deg(yaw_vel), label="yaw velocity", linewidth=1.0, alpha=0.8)
        ax.set_ylabel("Yaw [deg]")
        ax.set_xlabel("Time since start [s]")
        ax.set_title("Yaw: quaternion vs velocity heading")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig_yaw.tight_layout()
        out_yaw = out_path.with_name(out_path.stem + "_yaw.png")
        fig_yaw.savefig(out_yaw, dpi=150)
        print(f"[saved] {out_yaw}")


if __name__ == "__main__":
    main()
