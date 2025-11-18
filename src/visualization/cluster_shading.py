#!/usr/bin/env python3
"""
Shared helpers for shading timeseries plots according to terrain cluster labels.

- Samples per-timestep cluster IDs from GeoTIFF rasters (cluster_kmeansK_*).
- Converts label sequences into colored spans and applies them to Matplotlib axes.
"""
from __future__ import annotations

import colorsys
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import rasterio
    from pyproj import Transformer
    from rasterio.transform import rowcol
except ImportError:
    rasterio = None
    Transformer = None
    rowcol = None


@dataclass
class ClusterShading:
    segments: list[tuple[int, int, int]]
    colors: dict[int, tuple[float, float, float]]
    alpha: float
    description: str


def _cluster_palette(n_clusters: int) -> list[tuple[float, float, float]]:
    n = max(1, int(n_clusters))
    palette = []
    for i in range(n):
        h = (i / float(n)) % 1.0
        s, v = 0.65, 0.95
        palette.append(colorsys.hsv_to_rgb(h, s, v))
    return palette


def _compute_step_edges(x_values: np.ndarray) -> np.ndarray:
    x = np.asarray(x_values, dtype=np.float64)
    if x.size == 0:
        return np.asarray([], dtype=np.float64)
    if x.size == 1:
        step = 0.5
        return np.array([x[0] - step, x[0] + step], dtype=np.float64)
    diffs = np.diff(x)
    mids = x[:-1] + 0.5 * diffs
    edges = np.empty(x.size + 1, dtype=np.float64)
    edges[1:-1] = mids
    edges[0] = x[0] - 0.5 * diffs[0]
    edges[-1] = x[-1] + 0.5 * diffs[-1]
    return edges


def _segment_cluster_ids(cluster_ids: np.ndarray) -> list[tuple[int, int, int]]:
    segments: list[tuple[int, int, int]] = []
    start = None
    current = None
    for idx, val in enumerate(cluster_ids):
        if not np.isfinite(val):
            if current is not None and start is not None:
                segments.append((start, idx, current))
                start = None
                current = None
            continue
        cid = int(val)
        if current is None:
            start = idx
            current = cid
            continue
        if cid != current:
            segments.append((start, idx, current))
            start = idx
            current = cid
    if current is not None and start is not None:
        segments.append((start, cluster_ids.shape[0], current))
    return segments


def apply_cluster_shading(ax, x_values: np.ndarray, shading: ClusterShading | None) -> None:
    if shading is None or not shading.segments:
        return
    edges = _compute_step_edges(x_values)
    if edges.size == 0:
        return
    for start, end, cid in shading.segments:
        if end <= start:
            continue
        if end >= edges.size:
            end = edges.size - 1
        color = shading.colors.get(cid)
        if color is None:
            continue
        x0 = edges[start]
        x1 = edges[end]
        if not np.isfinite(x0) or not np.isfinite(x1):
            continue
        ax.axvspan(x0, x1, facecolor=color, alpha=shading.alpha, zorder=0.02, edgecolor="none", linewidth=0)


def _locate_cluster_raster(mission_maps: Path, kmeans: int | None, emb: str) -> tuple[Path, int | None]:
    clusters_dir = mission_maps / "clusters"
    if not clusters_dir.exists():
        raise FileNotFoundError(f"Cluster directory not found: {clusters_dir}")
    if kmeans is not None:
        cand = clusters_dir / f"cluster_kmeans{kmeans}_{emb}.tif"
        if not cand.exists():
            raise FileNotFoundError(f"{cand} not found (needed for cluster shading).")
        return cand, kmeans
    cands = sorted(
        clusters_dir.glob(f"cluster_kmeans*_{emb}.tif"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not cands:
        raise FileNotFoundError(f"No cluster_kmeans*_{emb}.tif files in {clusters_dir}")
    picked = cands[0]
    match = re.search(r"cluster_kmeans(\d+)", picked.name)
    detected = int(match.group(1)) if match else None
    return picked, detected


def _sample_cluster_labels(lat: np.ndarray, lon: np.ndarray, raster_path: Path) -> tuple[np.ndarray, int]:
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    labels = np.full(lat.shape, np.nan, dtype=np.float64)
    with rasterio.open(raster_path) as ds:
        band = ds.read(1)
        max_label = int(np.nanmax(band)) if band.size else -1
        ok = np.isfinite(lat) & np.isfinite(lon)
        if not ok.any():
            return labels, max_label
        tf = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
        x, y = tf.transform(lon[ok], lat[ok])
        rows, cols = rowcol(ds.transform, x, y, op=np.round)
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        inside = (rows >= 0) & (rows < ds.height) & (cols >= 0) & (cols < ds.width)
        if inside.any():
            idx_ok = np.nonzero(ok)[0]
            valid_idx = idx_ok[inside]
            labels_vals = band[rows[inside], cols[inside]].astype(np.float64)
            labels[valid_idx] = labels_vals
    return labels, max_label


def prepare_cluster_shading(
    df: pd.DataFrame,
    mission_maps: Path,
    emb: str,
    kmeans: int | None,
    alpha: float,
    mask: np.ndarray | None = None,
) -> ClusterShading | None:
    if rasterio is None or Transformer is None or rowcol is None:
        print("[warn] Cluster shading requires rasterio + pyproj; install them to enable this feature.")
        return None
    if "lat" not in df.columns or "lon" not in df.columns:
        print("[warn] Cannot shade clusters — 'lat'/'lon' columns missing.")
        return None
    try:
        raster_path, inferred_k = _locate_cluster_raster(mission_maps, kmeans, emb)
    except FileNotFoundError as exc:
        print(f"[warn] {exc}")
        return None

    lat = df["lat"].to_numpy()
    lon = df["lon"].to_numpy()
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != lat.shape[0]:
            raise ValueError("Mask length must match dataframe rows for cluster shading.")
        lat = lat[mask]
        lon = lon[mask]

    try:
        labels, max_label = _sample_cluster_labels(lat, lon, raster_path)
    except Exception as exc:
        print(f"[warn] Unable to sample cluster raster ({raster_path}): {exc}")
        return None
    segments = _segment_cluster_ids(labels)
    if not segments:
        print("[warn] Cluster shading requested but no valid cluster samples were found.")
        return None
    palette_len = inferred_k if inferred_k is not None else (max_label + 1 if max_label >= 0 else 0)
    if palette_len <= 0:
        palette_len = int(max(seg[2] for seg in segments) + 1)
    palette = _cluster_palette(palette_len)
    colors = {cid: palette[cid % len(palette)] for cid in {seg[2] for seg in segments}}
    desc = f"k={palette_len}, {emb} ({raster_path.name})"
    print(f"[info] Cluster shading enabled using {desc}")
    return ClusterShading(segments=segments, colors=colors, alpha=max(0.0, float(alpha)), description=desc)


__all__ = [
    "ClusterShading",
    "apply_cluster_shading",
    "prepare_cluster_shading",
]
