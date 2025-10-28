#!/usr/bin/env python3
"""
Overlay a synced metric (default: speed_error_abs) onto the SwissImage chip PNG,
the selected cluster RGB PNG, or both side-by-side.

- Mission-aware: finds latest synced_<Hz>Hz.parquet, swissimg *_rgb8.tif/.png,
  and (optionally) the cluster PNG named: cluster_kmeans{K}_{emb}_rgb.png
  where K is --kmeans (default 50) and emb ∈ {stego,dino} (default stego).
- GPS (lat/lon) -> LV95 (EPSG:2056) -> pixel (row/col) via GeoTIFF transform.
- Aggregates multiple samples falling into the same pixel by mean().
- Plots colored points over the PNG(s).

Usage:
  python src/visualization/plot_on_map.py --mission ETH-1
  python src/visualization/plot_on_map.py --mission ETH-1 --background both --metric speed_tracking_score
  python src/visualization/plot_on_map.py --mission ETH-1 --hz 10 --clip-percentile 97 --point-size 8 --min-zero
  python src/visualization/plot_on_map.py --mission ETH-1 --kmeans 75 --emb dino --background cluster
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer

# --- paths helper ---
import sys
from pathlib import Path
THIS_FILE = Path(__file__).resolve()
SRC_ROOT = THIS_FILE.parents[1]    # .../src
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from utils.paths import get_paths


P = get_paths()

# ----------------- helpers -----------------
def resolve_mission(mission: str) -> Tuple[Path, Path, str, str]:
    """Return (tables_dir, synced_dir, mission_id, short_folder)."""
    mj = P["REPO_ROOT"] / "config" / "missions.json"
    meta = json.loads(mj.read_text())
    mission_id = meta.get("_aliases", {}).get(mission, mission)
    entry = meta.get(mission_id)
    if not entry:
        raise KeyError(f"Mission '{mission}' not found in missions.json")
    short = entry["folder"]
    return (P["TABLES"] / short, P["SYNCED"] / short, mission_id, short)

def pick_synced(sync_dir: Path, hz: Optional[int]) -> Path:
    """
    Prefer metrics files:
        synced_{Hz}Hz_metrics.parquet  (if Hz given)
        newest synced_*Hz_metrics.parquet (if Hz not given)
    Fallback to legacy:
        synced_{Hz}Hz.parquet or newest synced_*Hz.parquet
    """
    # 1) Prefer metrics flavor
    if hz is not None:
        p = sync_dir / f"synced_{hz}Hz_metrics.parquet"
        if p.exists():
            return p
    else:
        metrics = sorted(sync_dir.glob("synced_*Hz_metrics.parquet"),
                         key=lambda p: p.stat().st_mtime, reverse=True)
        if metrics:
            return metrics[0]
    raise FileNotFoundError(
        f"No synced *_metrics parquet or legacy synced_*Hz.parquet found in {sync_dir}"
    )

def latest_swissimg_paths(short: str) -> Tuple[Path, Path]:
    """Return (tif_path, png_path) for the newest swissimg chip."""
    d = P["MAPS"] / short / "swissimg"
    tifs = sorted(d.glob("*_rgb8.tif"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not tifs:
        raise FileNotFoundError(f"No swissimg *_rgb8.tif in {d}")
    tif = tifs[0]
    # prefer the PNG with matching stem, otherwise pick newest PNG
    png_candidate = tif.with_suffix(".png")
    if png_candidate.exists():
        png = png_candidate
    else:
        pngs = sorted(d.glob("*_rgb8.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not pngs:
            raise FileNotFoundError(f"No swissimg *_rgb8.png in {d}")
        png = pngs[0]
    return tif, png

def cluster_png_path(short: str, kmeans_k: int, emb: str) -> Path:
    """
    Find cluster PNG with standardized name:
        cluster_kmeans{K}_{emb}_rgb.png
    under: MAPS/<short>/clusters/
    """
    d = P["MAPS"] / short / "clusters"
    p = d / f"cluster_kmeans{kmeans_k}_{emb}_rgb.png"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found (expected standardized cluster PNG).")
    return p

def latlon_to_pixels(lat: np.ndarray, lon: np.ndarray, tif_path: Path) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Project WGS84 lat/lon -> LV95 (EPSG:2056) -> pixel (col,row) for the given GeoTIFF."""
    with rasterio.open(tif_path) as ds:
        crs_dst = ds.crs
        if crs_dst is None:
            raise RuntimeError(f"{tif_path} has no CRS")
        H, W = ds.height, ds.width
        tf = Transformer.from_crs("EPSG:4326", crs_dst, always_xy=True)
        x, y = tf.transform(lon.astype(float), lat.astype(float))
        r, c = rowcol(ds.transform, x, y, op=round)
    return np.asarray(c), np.asarray(r), W, H

def aggregate_per_pixel(cols: np.ndarray, rows: np.ndarray, values: np.ndarray, W: int, H: int) -> pd.DataFrame:
    """Keep in-bounds samples and average values per (row,col)."""
    m = ~np.isnan(values)
    m &= (cols >= 0) & (cols < W) & (rows >= 0) & (rows < H)
    if m.sum() == 0:
        raise RuntimeError("No in-bounds samples to plot.")
    df = pd.DataFrame({"col": cols[m].astype(int), "row": rows[m].astype(int), "val": values[m].astype(float)})
    g = df.groupby(["row", "col"], as_index=False)["val"].mean()
    return g  # columns: row, col, val

def pick_vmin_vmax(vals: np.ndarray, clip_percentile: float, min_zero: bool) -> Tuple[float, float]:
    vmax = np.nanpercentile(vals, clip_percentile) if clip_percentile else np.nanmax(vals)
    vmax = float(max(vmax, 1e-9))
    vmin = 0.0 if min_zero else float(np.nanmin(vals))
    if not np.isfinite(vmin): vmin = 0.0
    if vmax <= vmin: vmax = vmin + 1e-6
    return vmin, vmax

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser(description="Plot a synced metric over SwissImage / cluster PNG(s).")
    ap.add_argument("--mission", required=True, help="Mission alias or UUID")
    ap.add_argument("--hz", type=int, default=None, help="Pick synced_<Hz>Hz.parquet (default: latest)")
    ap.add_argument("--metric", default="speed_error_abs", help="Metric column in the synced parquet")
    ap.add_argument("--background", choices=["raw", "cluster", "both"], default="raw",
                    help="Which background(s) to plot under the trajectory")
    ap.add_argument("--kmeans", type=int, default=50, help="Cluster count K for background cluster map (default: 50)")
    ap.add_argument("--emb", choices=["stego","dino"], default="stego",
                    help="Embedding used for cluster map filename (default: stego)")
    ap.add_argument("--clip-percentile", type=float, default=95.0, help="Clip colors at this percentile (0..100).")
    ap.add_argument("--min-zero", action="store_true", help="Force color scale to start at 0.")
    ap.add_argument("--cmap", default="coolwarm", help="Matplotlib colormap (blue→red default: coolwarm).")
    ap.add_argument("--point-size", type=float, default=6.0, help="Marker size in pixels.")
    ap.add_argument("--alpha", type=float, default=0.9, help="Marker alpha.")
    ap.add_argument("--out", default=None,
                    help="Output path (default: reports/<mission>/<metric>_<bg>_k{K}_{emb}.png)")
    args = ap.parse_args()

    # Paths
    _, sync_dir, mission_id, short = resolve_mission(args.mission)
    synced = pick_synced(sync_dir, args.hz)
    tif_path, png_raw = latest_swissimg_paths(short)

    png_cluster = None
    if args.background in ("cluster", "both"):
        try:
            png_cluster = cluster_png_path(short, args.kmeans, args.emb)
        except FileNotFoundError as e:
            if args.background == "cluster":
                raise
            else:
                print(f"[warn] {e} — will draw only raw background.")

    # Load metric + GPS
    df = pd.read_parquet(synced)
    if not {"lat", "lon", args.metric}.issubset(df.columns):
        missing = {"lat", "lon", args.metric} - set(df.columns)
        raise KeyError(f"Synced parquet missing columns: {missing}")

    lat = df["lat"].to_numpy()
    lon = df["lon"].to_numpy()
    val = df[args.metric].to_numpy()

    # Map GPS -> pixel using GeoTIFF georeference
    cols, rows, W, H = latlon_to_pixels(lat, lon, tif_path)
    perpix = aggregate_per_pixel(cols, rows, val, W, H)
    vmin, vmax = pick_vmin_vmax(perpix["val"].to_numpy(), args.clip_percentile, args.min_zero)

    # Prepare figure
    ncols = 2 if (args.background == "both" and png_cluster is not None) else 1
    fig_w = 12 if ncols == 2 else 8
    fig, axes = plt.subplots(1, ncols, figsize=(fig_w, 8), squeeze=False)
    axes = axes[0]

    def draw_on(ax, bg_png: Path, title: str):
        bg = np.asarray(Image.open(bg_png).convert("RGB"))
        ax.imshow(bg)
        sc = ax.scatter(perpix["col"], perpix["row"],
                        c=perpix["val"], s=args.point_size, cmap=args.cmap,
                        vmin=vmin, vmax=vmax, alpha=args.alpha, linewidths=0)
        ax.set_xlim([0, bg.shape[1]])
        ax.set_ylim([bg.shape[0], 0])  # invert y to match image coords
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        return sc

    # Draw backgrounds
    scatters = []
    if ncols == 2:
        scatters.append(draw_on(axes[0], png_raw, f"{short} — SwissImage"))
        scatters.append(draw_on(axes[1], png_cluster, f"{short} — Clusters (k={args.kmeans}, {args.emb})"))
    else:
        if args.background == "cluster" and png_cluster is not None:
            scatters.append(draw_on(axes[0], png_cluster, f"{short} — Clusters (k={args.kmeans}, {args.emb})"))
        else:
            scatters.append(draw_on(axes[0], png_raw, f"{short} — SwissImage"))

    # Single shared colorbar
    cbar = fig.colorbar(scatters[0], ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label(args.metric)

    fig.suptitle(f"{mission_id} · metric: {args.metric}", y=0.995, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    # Output -> repo-local reports, not data
    out_dir_default = P["REPO_ROOT"] / "reports" / short
    out_dir_default.mkdir(parents=True, exist_ok=True)
    bg_tag = ("both" if (args.background == "both" and png_cluster is not None)
              else ("cluster" if (args.background == "cluster" and png_cluster is not None) else "raw"))
    default_name = f"{short}_{args.metric}_{bg_tag}_k{args.kmeans}_{args.emb}.png"
    out_path = Path(args.out) if args.out else (out_dir_default / default_name)
    fig.savefig(out_path, dpi=200)
    print(f"[ok] saved {out_path}")

if __name__ == "__main__":
    main()
