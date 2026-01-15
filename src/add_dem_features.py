#!/usr/bin/env python3
"""
Compute DEM-based pitch/roll features along a mission trajectory and append them to parquet.

This script computes per-sample DEM pitch/roll at multiple smoothing scales using the
same conventions as build_patch_dataset.py, then applies optional filters to suppress
spikes. Results are merged into synced_<Hz>Hz_metrics.parquet or synced_<Hz>Hz.parquet.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml
from pyproj import Transformer

from utils.paths import get_paths
from utils.cli import add_mission_arguments, add_hz_argument, resolve_mission_from_args
from utils.filtering import build_dem_pitch_roll_chain, filter_signal, kalman_smooth_xy
from utils.geo import (
    find_lat_lon_cols,
    discover_dem_path,
    world_to_rowcol,
    format_dem_scale_label,
    gaussian_smooth_nan,
    compute_gradients,
    bilinear_sample,
    parse_dem_pitch_roll_cfg,
)
from utils.quaternion import get_quaternion_block, yaw_from_wxyz
from utils.synced import resolve_synced_parquet


def load_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text()) if p.exists() else {}


def compute_yaw_rad(df: pd.DataFrame) -> np.ndarray | None:
    """Extract per-sample yaw [rad] from body->world quaternions (ENU frame)."""
    block = get_quaternion_block(df)
    if block is None:
        return None
    return yaw_from_wxyz(*block)


def wrap_pi(angle_rad: np.ndarray) -> np.ndarray:
    return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi


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
    filter_chain = build_dem_pitch_roll_chain(dem_cfg["filter_params"])
    dem_filters_cfg = {"dem_pitch_roll": {"chain": filter_chain}}
    filters_cfg = metrics_cfg.get("filters", {})
    gps_yaw_cfg = (metrics_cfg.get("dem_pitch_roll") or {}).get("gps_yaw", {})

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
    filter_context = {"t_s": t}

    out = pd.DataFrame({"t": t})
    gps_yaw_rad = None
    gps_speed_mps = None
    if gps_yaw_cfg is not None:
        process_var = float(gps_yaw_cfg.get("process_var", 1.0))
        meas_var = float(gps_yaw_cfg.get("meas_var", 1.0))
        init_pos_var = float(gps_yaw_cfg.get("init_pos_var", 10.0))
        init_vel_var = float(gps_yaw_cfg.get("init_vel_var", 1.0))
        min_speed_mps = float(gps_yaw_cfg.get("min_speed_mps", 0.05))

        _east_s, _north_s, vel_e, vel_n = kalman_smooth_xy(
            t,
            east,
            north,
            process_var=process_var,
            meas_var=meas_var,
            init_pos_var=init_pos_var,
            init_vel_var=init_vel_var,
        )
        gps_speed = np.hypot(vel_e, vel_n)
        gps_speed_filt = filter_signal(
            gps_speed,
            "dem_pitch_roll_gps_speed",
            filters_cfg=filters_cfg,
            context=filter_context,
        )
        gps_speed_mps = gps_speed_filt if gps_speed_filt is not None else gps_speed
        gps_yaw = np.arctan2(vel_n, vel_e)
        moving = gps_speed >= min_speed_mps
        gps_yaw[~moving] = np.nan
        if np.isfinite(gps_yaw).any():
            yaw_filled = pd.Series(gps_yaw).ffill().bfill().to_numpy(dtype=np.float64)
            yaw_unwrapped = np.unwrap(yaw_filled)
            yaw_deg = np.rad2deg(yaw_unwrapped)
            yaw_deg_filt = filter_signal(
                yaw_deg,
                "dem_pitch_roll_gps_yaw",
                filters_cfg=filters_cfg,
                context=filter_context,
            )
            yaw_deg_filt = yaw_deg_filt if yaw_deg_filt is not None else yaw_deg
            gps_yaw_rad = wrap_pi(np.deg2rad(yaw_deg_filt))
        else:
            gps_yaw_rad = np.full_like(gps_yaw, np.nan, dtype=np.float64)

        out["gps_speed_mps"] = gps_speed_mps
        out["gps_yaw_rad"] = gps_yaw_rad
        out["gps_yaw_deg"] = np.rad2deg(gps_yaw_rad)
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
        out[f"dem_pitch_{label}_deg"] = pitch_deg
        out[f"dem_roll_{label}_deg"] = roll_deg
        if gps_yaw_rad is not None:
            cos_yaw_gps = np.cos(gps_yaw_rad)
            sin_yaw_gps = np.sin(gps_yaw_rad)
            s_parallel_gps = g_e * cos_yaw_gps + g_n * sin_yaw_gps
            s_perp_gps = -g_e * sin_yaw_gps + g_n * cos_yaw_gps
            pitch_deg_gps = np.rad2deg(np.arctan(s_parallel_gps))
            roll_deg_gps = np.rad2deg(np.arctan(s_perp_gps))
            pitch_deg_gps = filter_signal(
                pitch_deg_gps,
                "dem_pitch_roll",
                filters_cfg=dem_filters_cfg,
                context=filter_context,
            )
            roll_deg_gps = filter_signal(
                roll_deg_gps,
                "dem_pitch_roll",
                filters_cfg=dem_filters_cfg,
                context=filter_context,
            )
            out[f"dem_pitch_{label}_deg_gps"] = pitch_deg_gps
            out[f"dem_roll_{label}_deg_gps"] = roll_deg_gps

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
