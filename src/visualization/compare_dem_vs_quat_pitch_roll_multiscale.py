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

from add_dem_features import load_yaml, compute_yaw_rad
from utils.geo import (
    parse_dem_pitch_roll_cfg,
    find_lat_lon_cols,
    discover_dem_path,
    world_to_rowcol,
    gaussian_smooth_nan,
    compute_gradients,
    bilinear_sample,
    format_dem_scale_label,
)
from build_patch_dataset import compute_pitch_deg
from utils.quaternion import get_quaternion_block
from utils.filtering import build_dem_pitch_roll_chain, filter_signal, gaussian_smooth_1d_nan


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
    filter_params = dict(dem_cfg["filter_params"])
    if args.rate_hampel_filter is not None:
        filter_params["rate_hampel_filter"] = args.rate_hampel_filter
    if args.rate_hampel_max_rate is not None:
        filter_params["rate_hampel_max_rate"] = args.rate_hampel_max_rate
    if args.rate_hampel_window is not None:
        filter_params["rate_hampel_window"] = args.rate_hampel_window
    if args.rate_hampel_z is not None:
        filter_params["rate_hampel_z"] = args.rate_hampel_z
    if args.value_gauss_sigma is not None:
        filter_params["value_gauss_sigma"] = args.value_gauss_sigma
    dem_filter_chain = build_dem_pitch_roll_chain(filter_params)
    dem_filters_cfg = {"dem_pitch_roll": {"chain": dem_filter_chain}}
    value_gauss_sigma = float(filter_params.get("value_gauss_sigma", 0.0))

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
    filter_context = {"t_s": t_rel}
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

        pitch_deg = filter_signal(
            pitch_deg,
            "dem_pitch_roll",
            filters_cfg=dem_filters_cfg,
            context=filter_context,
        )
        roll_deg = filter_signal(
            roll_deg,
            "dem_pitch_roll",
            filters_cfg=dem_filters_cfg,
            context=filter_context,
        )

        label = format_dem_scale_label(scale_m)
        dem_pitch[label] = pitch_deg
        dem_roll[label] = roll_deg

    if value_gauss_sigma and value_gauss_sigma > 0.0:
        pitch_quat = gaussian_smooth_1d_nan(pitch_quat, value_gauss_sigma, None)
        roll_quat = gaussian_smooth_1d_nan(roll_quat, value_gauss_sigma, None)

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
