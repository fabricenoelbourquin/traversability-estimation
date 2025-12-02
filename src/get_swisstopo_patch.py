#!/usr/bin/env python3
"""
Fetch a SwissImage RGB basemap chip centered on a mission trajectory,
reading GPS (lat/lon) from the synced parquet.

Optionally, fetch the swissALTI3D DEM for the same LV95 frame by downloading
the corresponding 1 km swissALTI3D tiles from data.geo.admin.ch and mosaicking
them to the chip extent.

Saves:
  <DATA_ROOT>/maps/<mission_folder>/swisstopo/
    swissimg_chip{chip_px}_gsd{gsd}_rgb8.tif
    swissimg_chip{chip_px}_gsd{gsd}_rgb8.png
    swissimg_chip{chip_px}_gsd{gsd}.json

  and if enabled in imagery.yaml (fetch.alti.mode: dem|both):
    swissimg_chip{chip_px}_gsd{gsd}_dem{dem_res}m.tif
    swissimg_chip{chip_px}_gsd{gsd}_dem{dem_res}m.png
Usage:
    python src/get_swisstopo_patch.py --mission <mission>
    e.g. python src/get_swisstopo_patch.py --mission ETH-1
"""

from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import requests
import yaml
from shapely.geometry import LineString
from shapely.ops import transform as shp_transform
from pyproj import Transformer
import rasterio
from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.transform import from_bounds
from affine import Affine

# ---- paths helper (your existing one) ----
from utils.paths import get_paths
from utils.missions import resolve_mission
from utils.synced import resolve_synced_parquet

P = get_paths()

# ----------------- helpers -----------------
def load_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text()) if p.exists() else {}

def get_ll_points_from_synced(df: pd.DataFrame) -> list[tuple[float, float]]:
    """Return list of (lon, lat) from synced parquet, dropping NaNs."""
    if not {"lat", "lon"}.issubset(df.columns):
        raise KeyError("Synced parquet has no 'lat'/'lon'. Did GPS make it into sync?")
    sub = df[["lon", "lat"]].dropna()
    if len(sub) < 2:
        raise RuntimeError("Not enough GPS points to build chip center.")
    if len(sub) > 20000:
        sub = sub.iloc[::10, :]
    return list(map(tuple, sub.to_numpy(dtype=float)))

# ---- geometry: fixed chip in LV95 ----
def _to_2056_geom(geom_ll):
    """Project shapely geometry from EPSG:4326 to EPSG:2056 (LV95)."""
    to_2056 = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True).transform
    return shp_transform(to_2056, geom_ll)

def build_centered_chip_2056(points_ll: Sequence[tuple[float,float]], chip_px: int, gsd_m: float):
    """
    Square chip (EPSG:2056 bounds) centered on the track bbox center.
    side_m = chip_px * gsd_m
    """
    line_ll = LineString(points_ll)
    line_2056 = _to_2056_geom(line_ll)
    minx, miny, maxx, maxy = line_2056.bounds
    cx = 0.5*(minx + maxx); cy = 0.5*(miny + maxy)
    side_m = chip_px * float(gsd_m)
    half = side_m / 2.0
    return (cx - half, cy - half, cx + half, cy + half)

# ---- WMS fetch (SwissImage) ----
def _wms_getmap(bbox_2056, width, height, url, layer, version, fmt):
    minx, miny, maxx, maxy = bbox_2056
    params = {
        "SERVICE": "WMS",
        "VERSION": version,
        "REQUEST": "GetMap",
        "LAYERS":  layer,
        "STYLES":  "",
        "CRS":     "EPSG:2056",   # WMS 1.3.0
        "BBOX":    f"{minx},{miny},{maxx},{maxy}",
        "WIDTH":   str(int(width)),
        "HEIGHT":  str(int(height)),
        "FORMAT":  fmt,
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    ctype = r.headers.get("Content-Type", "")
    if not ctype.startswith("image/"):
        raise RuntimeError(f"WMS error (not an image):\n{r.text[:600]}")
    with MemoryFile(r.content) as mem:
        with mem.open() as ds:
            arr = ds.read()  # (bands, H, W)
            if arr.shape[0] == 4:
                arr = arr[:3]
            return arr

def _fetch_wms_tiled(bbox_2056, width_px, height_px, url, layer, version, fmt, max_px):
    W, H = int(width_px), int(height_px)
    out = np.zeros((3, H, W), dtype=np.uint8)
    minx, miny, maxx, maxy = map(float, bbox_2056)
    sx = (maxx - minx) / W
    sy = (maxy - miny) / H
    nx = math.ceil(W / max_px)
    ny = math.ceil(H / max_px)
    for iy in range(ny):
        y0 = iy * max_px
        y1 = min(H, (iy + 1) * max_px)
        for ix in range(nx):
            x0 = ix * max_px
            x1 = min(W, (ix + 1) * max_px)
            sub_minx = minx + x0 * sx
            sub_maxx = minx + x1 * sx
            sub_miny = miny + y0 * sy
            sub_maxy = miny + y1 * sy
            tile = _wms_getmap((sub_minx, sub_miny, sub_maxx, sub_maxy),
                               x1 - x0, y1 - y0, url, layer, version, fmt)
            out[:, y0:y1, x0:x1] = tile
    return out

def save_geotiff_rgb8(path: Path, rgb_uint8, bbox_2056):
    minx, miny, maxx, maxy = bbox_2056
    H, W = rgb_uint8.shape[1], rgb_uint8.shape[2]
    transform = from_bounds(minx, miny, maxx, maxy, W, H)
    with rasterio.open(
        path, "w",
        driver="GTiff", width=W, height=H, count=3, dtype="uint8",
        crs="EPSG:2056", transform=transform,
        compress="deflate", tiled=True, photometric="RGB"
    ) as dst:
        dst.write(rgb_uint8[0], 1)
        dst.write(rgb_uint8[1], 2)
        dst.write(rgb_uint8[2], 3)

def save_png_rgb8(path: Path, rgb_uint8):
    H, W = rgb_uint8.shape[1], rgb_uint8.shape[2]
    with rasterio.open(path, "w", driver="PNG", width=W, height=H, count=3, dtype="uint8") as dst:
        dst.write(rgb_uint8[0], 1)
        dst.write(rgb_uint8[1], 2)
        dst.write(rgb_uint8[2], 3)

# ---- swissALTI3D (1km tiles, deterministic) ----
def _alti_tiles_for_bbox_2056(bbox_2056: tuple[float, float, float, float]) -> list[tuple[int, int]]:
    """
    swissALTI3D is published as 1 km tiles in LV95.
    Tile name is basically <E_km>-<N_km>, e.g. 2573-1085.
    """
    minx, miny, maxx, maxy = bbox_2056
    e_min = int(math.floor(minx / 1000))
    e_max = int(math.floor((maxx - 1e-6) / 1000))
    n_min = int(math.floor(miny / 1000))
    n_max = int(math.floor((maxy - 1e-6) / 1000))
    tiles: list[tuple[int, int]] = []
    for e in range(e_min, e_max + 1):
        for n in range(n_min, n_max + 1):
            tiles.append((e, n))
    return tiles


def _download_alti_tile(
    session: requests.Session,
    year: int,
    e_km: int,
    n_km: int,
    res_m_opts: list[float],
) -> tuple[rasterio.io.DatasetReader, MemoryFile, float] | None:
    """
    Try to download ONE tile, trying several resolutions for that tile.
    Returns (dataset, memfile, resolution_m) or None.
    """
    tile_id = f"swissalti3d_{year}_{e_km}-{n_km}"
    base = f"https://data.geo.admin.ch/ch.swisstopo.swissalti3d/{tile_id}/{tile_id}"

    for res_m in res_m_opts:
        # swissalti3d_2019_2573-1085_0.5_2056_5728.tif
        url = f"{base}_{res_m:g}_2056_5728.tif"
        r = session.get(url, timeout=60)
        if r.status_code == 200:
            mem = MemoryFile(r.content)
            ds = mem.open()
            return ds, mem, float(res_m)
    return None

def fetch_swissalti3d_dem(
    bbox_2056: tuple[float, float, float, float],
    cfg_alti: dict,
    out_dir: Path,
    stem_like_img: str,
):
    """
    Deterministic swissALTI3D download:
    - figure out which 1 km tiles we need
    - for EACH tile, try year, then fallbacks, and for EACH year try 0.5 m then 2.0 m
    - whatever we get, we mosaic to EXACT bbox at EXACT target_res (default 0.5 m)
    """
    mode = cfg_alti.get("mode", "none")
    if mode not in {"dem", "both"}:
        return None

    dem_cfg = cfg_alti.get("dem", {})
    pref_year = int(dem_cfg.get("release_year", 2019))
    fallback_years = dem_cfg.get("fallback_years", [2024, 2023, 2022, 2021, 2020, 2019])
    # dedupe while keeping order
    years = []
    for y in [pref_year, *fallback_years]:
        if y not in years:
            years.append(y)

    # you said: "I absolutely want 0.5m"
    target_res = float(dem_cfg.get("resolution_m", 0.5))
    # when downloading we can try 0.5 first, then 2.0 (resample later)
    res_opts_download = dem_cfg.get("resolutions_m", [0.5, 2.0])

    tiles = _alti_tiles_for_bbox_2056(bbox_2056)
    if not tiles:
        print("[warn] swissALTI3D: no tiles computed for this bbox (too small?)")
        return None

    session = requests.Session()
    src_datasets: list[rasterio.io.DatasetReader] = []
    src_memfiles: list[MemoryFile] = []
    used_info: list[dict] = []

    for (e_km, n_km) in tiles:
        ds = None
        mem = None
        used_year = None
        used_res = None
        for year in years:
            got = _download_alti_tile(session, year, e_km, n_km, res_opts_download)
            if got is not None:
                ds, mem, used_res = got
                used_year = year
                break
        if ds is not None:
            src_datasets.append(ds)
            src_memfiles.append(mem)  # keep alive
            used_info.append(
                {
                    "tile": f"{e_km}-{n_km}",
                    "year": used_year,
                    "resolution_m": used_res,
                }
            )
        else:
            print(f"[warn] swissALTI3D: tile {e_km}-{n_km} not found in years {years}")

    if not src_datasets:
        print("[warn] swissALTI3D: no tiles could be downloaded for this bbox (after per-tile fallback)")
        return None

    overscan_px = 1  # how many pixels extra on each side to avoid edge artifacts due to grid misalignment
    overscan_m  = overscan_px * target_res
    minx, miny, maxx, maxy = bbox_2056

    # 1) request a slightly bigger area
    bigger_bounds = (
        minx - overscan_m,
        miny - overscan_m,
        maxx + overscan_m,
        maxy + overscan_m,
    )

    mosaic, bigger_transform = merge(
        src_datasets,
        bounds=bigger_bounds,
        res=target_res,
        nodata=-9999.0,
    )
    big_arr = mosaic[0] # shape: (512 + 2) x (512 + 2) = 514 x 514, (1, H, W) -> (H, W)
    # 2) crop back to the original 512x512 (remove 1 px from each side)
    dem_arr = big_arr[overscan_px:-overscan_px, overscan_px:-overscan_px]
    out_transform = bigger_transform * Affine.translation(overscan_px, overscan_px)

    height, width = dem_arr.shape
    side_px = width
    res_str = f"{int(round(target_res * 100)):03d}"
    dem_stem = f"alti3d_chip{side_px}_gsd{res_str}"
    dem_path = out_dir / f"{dem_stem}.tif"
    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="float32",
        crs="EPSG:2056",
        transform=out_transform,
        nodata=-9999.0,
        compress="deflate",
        tiled=True,
    ) as dst:
        dst.write(dem_arr, 1)

    # make a viewable PNG (no more “all black”)
    valid = dem_arr != -9999.0
    if valid.any():
        vmin = float(dem_arr[valid].min())
        vmax = float(dem_arr[valid].max())
        if vmax > vmin:
            scaled = np.zeros_like(dem_arr, dtype=np.uint8)
            scaled[valid] = ((dem_arr[valid] - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        else:
            scaled = np.zeros_like(dem_arr, dtype=np.uint8)
    else:
        scaled = np.zeros_like(dem_arr, dtype=np.uint8)

    dem_png  = out_dir / f"{dem_stem}.png"
    with rasterio.open(
        dem_png,
        "w",
        driver="PNG",
        width=width,
        height=height,
        count=1,
        dtype="uint8",
    ) as dst:
        dst.write(scaled, 1)

    coverage = float((dem_arr != -9999.0).mean()) * 100.0
    print(
        f"[ok] swissALTI3D DEM saved: {dem_path} "
        f"(res={target_res} m, size=({width}x{height}), coverage={coverage:.1f}%)"
    )

    return {
        "dem_tif": str(dem_path),
        "dem_png": str(dem_png),
        "dem_resolution_m": target_res,
        "dem_coverage_percent": coverage,
        "tiles": used_info,
    }


# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser(description="Fetch fixed-chip SwissImage patch for a mission using synced GPS.")
    ap.add_argument("--mission", required=True, help="Mission alias or UUID")
    ap.add_argument("--hz", type=int, default=None, help="Pick synced_<Hz>Hz.parquet (default: latest)")
    ap.add_argument("--config", default=None, help="Path to config/imagery.yaml (default: repo config)")
    # optional overrides
    ap.add_argument("--chip-px", type=int, help="Override chip_px")
    ap.add_argument("--gsd-m", type=float, help="Override gsd_m")
    args = ap.parse_args()

    # Load config
    cfg_path = Path(args.config) if args.config else (P["REPO_ROOT"] / "config" / "imagery.yaml")
    cfg = load_yaml(cfg_path)

    # your yaml has "fetch: {...}"
    fetch_cfg = cfg.get("fetch", cfg)

    chip_px = int(args.chip_px or fetch_cfg.get("chip_px", 1024))
    gsd_m   = float(args.gsd_m   or fetch_cfg.get("gsd_m", 0.25))

    wms = fetch_cfg.get("wms", {})
    WMS_URL   = wms.get("url", "https://wms.geo.admin.ch/")
    WMS_LAYER = wms.get("layer", "ch.swisstopo.swissimage")
    WMS_VER   = wms.get("version", "1.3.0")
    WMS_FMT   = wms.get("format", "image/jpeg")
    MAX_WMS_PX = int(wms.get("max_px", 10000))

    out_cfg = fetch_cfg.get("out", {})
    subdir = out_cfg.get("subdir", "swisstopo")
    prefix = out_cfg.get("prefix", "")

    # Resolve mission + synced parquet
    mp = resolve_mission(args.mission, P)
    sync_dir, mission_id, short = mp.synced, mp.mission_id, mp.folder
    synced_path = resolve_synced_parquet(sync_dir, args.hz)
    df = pd.read_parquet(synced_path)
    pts_ll = get_ll_points_from_synced(df)

    # Build fixed chip in LV95
    bbox2056 = build_centered_chip_2056(pts_ll, chip_px, gsd_m)
    width_px = height_px = int(chip_px)

    # Fetch SwissImage
    if width_px <= MAX_WMS_PX and height_px <= MAX_WMS_PX:
        rgb = _wms_getmap(bbox2056, width_px, height_px, WMS_URL, WMS_LAYER, WMS_VER, WMS_FMT)
    else:
        rgb = _fetch_wms_tiled(bbox2056, width_px, height_px, WMS_URL, WMS_LAYER, WMS_VER, WMS_FMT, MAX_WMS_PX)

    # Output paths
    out_dir = (P["MAPS"] / short / subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = (prefix or "swissimg") + f"_chip{chip_px}_gsd{gsd_m}"
    tif_path = out_dir / f"{stem}_rgb8.tif"
    png_path = out_dir / f"{stem}_rgb8.png"

    save_geotiff_rgb8(tif_path, rgb, bbox2056)
    save_png_rgb8(png_path, rgb)

    # swissALTI3D (DEM) – best effort, no crash if empty
    alti_cfg = fetch_cfg.get("alti") or cfg.get("alti") or {}
    dem_meta = None
    if alti_cfg:
        dem_meta = fetch_swissalti3d_dem(bbox2056, alti_cfg, out_dir, stem)

    # sidecar meta (image)
    meta = {
        "mission_id": mission_id,
        "folder": short,
        "chip_px": chip_px,
        "gsd_m": gsd_m,
        "bbox_lv95": bbox2056,
        "crs": "EPSG:2056",
        "synced": str(synced_path),
        "tif": str(tif_path),
        "png": str(png_path),
        "wms": {"url": WMS_URL, "layer": WMS_LAYER, "version": WMS_VER, "format": WMS_FMT},
        "dem": dem_meta,
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2))
    print(f"[ok] Saved:\n  {tif_path}\n  {png_path}")

if __name__ == "__main__":
    main()
