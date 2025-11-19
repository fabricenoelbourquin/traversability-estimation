"""
Full pipeline runner for a single mission (sequential).
Usage:
    python src/run_pipeline.py --mission-id <UUID> [--mission-name <alias>]
    e.g. python src/run_pipeline.py --mission-id e97e35ad-dd7b-49c4-a158-95aba246520e --mission-name ETH-1
"""

#!/usr/bin/env python3
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
import yaml

from pathlib import Path as _P
_THIS = _P(__file__).resolve()
_SRC_ROOT = _THIS.parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
from utils.paths import get_paths
from utils.missions import resolve_mission

def sh(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)

def yes(x) -> bool:
    return bool(x) and str(x).lower() not in {"0", "false", "no", "off"}

# ---------- helpers for corruption handling ----------

def _header_ok(p: Path) -> bool:
    try:
        with p.open("rb") as f:
            head = f.read(16)
        return head.startswith(b"#ROSBAG")
    except Exception:
        return False

def _parse_bag_extras(bags_cfg) -> list[str]:
    """
    Accepts either:
      bags:
        enabled: true|false   # optional master switch, defaults to true
        camera: true|false    # toggles by name
        # ...future toggles...
        extras: [camera, ...] # optional explicit list
    or simply:
      bags: [camera, ...]
    Returns a sorted list of enabled extras.
    """
    if not bags_cfg:
        return []

    # master enable (defaults to True)
    if isinstance(bags_cfg, dict) and not yes(bags_cfg.get("enabled", True)):
        return []

    extras = set()

    if isinstance(bags_cfg, dict):
        # explicit list
        ex_list = bags_cfg.get("extras")
        if isinstance(ex_list, (list, tuple, set)):
            for x in ex_list:
                if x:
                    extras.add(str(x))
        # per-toggle booleans (e.g., camera: true)
        for k, v in bags_cfg.items():
            if k == "extras" or k == "enabled":
                continue
            if yes(v):
                extras.add(str(k))

    elif isinstance(bags_cfg, (list, tuple, set)):
        for x in bags_cfg:
            if x:
                extras.add(str(x))

    return sorted(extras)

def _find_bad_bags(raw_dir: Path) -> list[Path]:
    """
    Detect unreadable ROS1 bags. Ignore macOS AppleDouble files (._*).
    Returns a list of paths that should be deleted & re-downloaded.
    """
    bad: list[Path] = []
    try:
        from rosbags.highlevel import AnyReader
    except Exception:
        # If rosbags is not importable here, just fallback to header-only screening.
        AnyReader = None  # type: ignore

    for p in sorted(raw_dir.glob("*.bag")):
        if p.name.startswith("._"):  # macOS sidecar, not a real bag
            bad.append(p)
            continue
        if not _header_ok(p):
            bad.append(p)
            continue
        if AnyReader is not None:
            try:
                with AnyReader([p]) as r:
                    pass
            except Exception:
                bad.append(p)
    return bad

def _delete_files(files: list[Path]) -> None:
    for p in files:
        try:
            print(f"[warn] Removing unreadable file: {p}")
            p.unlink(missing_ok=True)
        except Exception as e:
            print(f"[warn] Could not remove {p}: {e}")

def main():
    p = argparse.ArgumentParser(description="Run pipeline for a single mission (sequential).")
    p.add_argument("--mission-id", type=str, required=False, help="Mission UUID (REQUIRED).")
    p.add_argument("--mission-name", type=str, default=None, help="Optional alias (e.g., ETH-1).")
    p.add_argument("--config", type=str, default="config/pipeline.yaml", help="YAML config path")
    p.add_argument("--already-downloaded", action="store_true",
                   help="Skip download + extract + sync; recompute metrics and make plots.")
    p.add_argument("--metrics-only", action="store_true",
                   help="Only run compute_metrics.py (skip download/extract/sync and all later stages).")
    # optional fine-grained skips
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--skip-extract", action="store_true")
    p.add_argument("--skip-sync", action="store_true")
    p.add_argument("--skip-swissimg", action="store_true")
    p.add_argument("--skip-cluster", action="store_true")
    p.add_argument("--skip-visuals", action="store_true")
    p.add_argument("--skip-dem-slope", action="store_true")
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    bags_cfg = (cfg.get("bags") or {})
    bag_extras = _parse_bag_extras(bags_cfg)
    def _build_download_cmd() -> list[str]:
        cmd = [py, "src/download_bag.py", "--mission-id", mission_id]
        if mission_name:
            cmd += ["--mission-name", mission_name]
        for ex in bag_extras:
            cmd += ["--extra", ex]
        return cmd
    
    # ID required (from CLI or YAML)
    mission_id = args.mission_id or cfg.get("mission_id")
    mission_name = args.mission_name or cfg.get("mission_name")
    if not mission_id:
        raise ValueError("Missing required --mission-id (or set mission_id in the YAML).")

    P = get_paths()

    # Find the mission folder in RAW to scan/delete bad bags if needed
    mp = resolve_mission(args.mission_id, P)
    raw_dir, tables_dir, display_name = mp.raw, mp.tables, mp.display


    # Helper: for stages that accept either --mission OR --mission-id (but not both)
    def ident_either() -> list[str]:
        if mission_name:
            return ["--mission", mission_name]
        return ["--mission-id", mission_id]

    # Stage toggles
    metrics_only = args.metrics_only
    if metrics_only:
        print(">> Metrics-only mode enabled: skipping download/extract/sync and all post-metric stages.")
    skip_download = args.skip_download or args.already_downloaded or metrics_only
    skip_extract  = args.skip_extract or metrics_only
    skip_sync     = args.skip_sync or metrics_only
    skip_swissimg = args.skip_swissimg or metrics_only
    skip_cluster  = args.skip_cluster or metrics_only
    skip_visuals  = args.skip_visuals or metrics_only
    skip_dem_slope = args.skip_dem_slope or metrics_only
    py = sys.executable

    # 1) Download (this script accepts id + name together)
    if not skip_download:
        cmd = _build_download_cmd()
        try:
            sh(cmd)
        except subprocess.CalledProcessError as e:
            if e.returncode == 20:
                print(f"[skip] {display_name}: missing GPS bag; skipping mission.")
                sys.exit(20)
            raise
    else:
        print(">> Skipping download (already-downloaded / skip-download).")

    # 2) Extract with self-heal
    def _try_extract_with_self_heal() -> None:
        """Attempt extract; if it fails due to corrupted bags, delete bad ones, re-download, and retry once."""
        # First attempt
        try:
            sh([py, "src/extract_bag.py", *ident_either()])
            return
        except subprocess.CalledProcessError as first_err:
            print("[warn] extract_bag.py failed - scanning for unreadable bags…")
            # Scan RAW mission dir
            if not raw_dir.exists():
                print(f"[error] Raw dir not found: {raw_dir}")
                raise

            bad = _find_bad_bags(raw_dir)
            if not bad:
                print("[warn] No unreadable bags detected. Re-raising original error.")
                raise first_err

            # Delete unreadable files (including '._*')
            _delete_files(bad)

            # Re-download mission bags (pattern-based) so the removed ones are fetched again
            try:
                dl_cmd = _build_download_cmd()
                sh(dl_cmd)
            except subprocess.CalledProcessError as e:
                print(f"[error] Re-download failed: {e}")
                raise first_err

            # Retry extract once
            try:
                sh([py, "src/extract_bag.py", *ident_either()])
                print("[info] extract_bag.py succeeded after self-heal.")
                return
            except subprocess.CalledProcessError as second_err:
                print("[error] extract_bag.py failed again after re-download. Giving up.")
                # Optional: show which bags are still broken
                still_bad = _find_bad_bags(raw_dir)
                if still_bad:
                    print("[error] Unreadable bags still present:")
                    for pth in still_bad:
                        print(f"  - {pth}")
                raise second_err

    if not skip_extract:
        _try_extract_with_self_heal()
    else:
        print(">> Skipping extract (already-downloaded / skip-extract).")

    # Early GPS check after extraction (skip if gps.parquet missing/empty)    
    gps_parquet = tables_dir / "gps.parquet"
    if not gps_parquet.exists():
        print(f"[skip] {display_name}: GPS table missing after extraction; skipping mission.")
        sys.exit(20)
    try:
        import pandas as _pd
        if _pd.read_parquet(gps_parquet).empty:
            print(f"[skip] {display_name}: GPS table is empty; skipping mission.")
            sys.exit(20)
    except Exception as _e:
        # If we can’t read it, treat as missing
        print(f"[skip] {display_name}: GPS table unreadable ({_e}); skipping mission.")
        sys.exit(20)

    # 3) Sync streams
    if not skip_sync:
        sh([py, "src/sync_streams.py", *ident_either()])
    else:
        print(">> Skipping sync (already-downloaded / skip-sync).")

    # 4) Compute metrics
    sh([py, "src/compute_metrics.py", *ident_either()])

    # 5) SwissImage patch
    if not skip_swissimg:
        sh([py, "src/get_swisstopo_patch.py", *ident_either()])
    else:
        print(">> Skipping SwissImage patch.")

    # 6) Infer cluster map
    if not skip_cluster:
        sh([py, "src/infer_cluster_map.py", *ident_either()])
    else:
        print(">> Skipping cluster inference.")

    # 7) Add DEM/longlat/slope
    if not skip_dem_slope:
        sh([py, "src/add_dem_longlat_slope.py", *ident_either()])
    else:
        print(">> Skipping DEM/longlat/slope.")    

    # 8-11) Visualizations
    if not skip_visuals:
        vis = cfg.get("visuals", {})

        ts_cfg = (vis.get("timeseries") or {})
        if yes(ts_cfg.get("enabled", True)):
            plot_time = yes(ts_cfg.get("plot_time", True))
            plot_distance = yes(ts_cfg.get("plot_distance", False))
            if not plot_time and not plot_distance:
                print(">> Timeseries plot enabled but no axes selected (set plot_time/plot_distance). Skipping.")
            else:
                cmd = [py, "src/visualization/plot_metric_progress.py", *ident_either()]
                if yes(ts_cfg.get("with_speeds", False)):
                    cmd.append("--with-speeds")
                if yes(ts_cfg.get("overlay_pitch", False)):
                    cmd.append("--overlay-pitch")
                if plot_time:
                    cmd.append("--plot-time")
                if plot_distance:
                    cmd.append("--plot-distance")
                sh(cmd)
        else:
            print(">> Timeseries plot disabled in config.")

        map_cfg = (vis.get("map") or {})
        if yes(map_cfg.get("enabled", True)):
            cmd = [py, "src/visualization/plot_on_map.py", *ident_either()]
            bg = map_cfg.get("background", "both")
            if bg:
                cmd += ["--background", bg]
            if yes(map_cfg.get("emb_both", False)):
                cmd.append("--emb-both")
            sh(cmd)
        else:
            print(">> Map plot disabled in config.")

        # --- DEM 3D plot
        dem3d_cfg = (vis.get("dem3d") or {})
        if yes(dem3d_cfg.get("enabled", True)):
            cmd = [py, "src/visualization/plot_dem_3d.py", *ident_either()]
            # Default true; pass the flag when true (harmless if script also defaults to true)
            if yes(dem3d_cfg.get("plot_trajectory", True)):
                cmd.append("--plot-trajectory")
            sh(cmd)
        else:
            print(">> DEM 3D plot disabled in config.")

        # --- Compare DEM vs quat pitch/roll
        cmp_cfg = (vis.get("compare_dem_vs_quat_pitch_roll") or {})
        if yes(cmp_cfg.get("enabled", True)):
            cmd = [py, "src/visualization/compare_dem_vs_quat_pitch_roll.py", *ident_either()]
            smooth_win = int(cmp_cfg.get("smooth_win", 1) or 1)
            if smooth_win != 1:
                cmd += ["--smooth-win", str(smooth_win)]
            sh(cmd)
        else:
            print(">> DEM vs quat pitch/roll comparison disabled in config.")

        # --- Videos
        vids_cfg = (vis.get("videos") or {})

        # video_metric_viewer
        vmv_cfg = (vids_cfg.get("metric_viewer") or {})
        if yes(vmv_cfg.get("enabled", False)):  # default off to avoid heavy runs unless requested
            cmd = [py, "src/visualization/video_metric_viewer.py", *ident_either()]
            # no-plot-orientation: default False -> add flag only if True
            if yes(vmv_cfg.get("no_plot_orientation", False)):
                cmd.append("--no-plot-orientation")
            # overlay-pitch: default True -> add flag if True
            if yes(vmv_cfg.get("overlay_pitch", True)):
                cmd.append("--overlay-pitch")
            if yes(vmv_cfg.get("shade_clusters", False)):
                cmd.append("--shade-clusters")
            sh(cmd)
        else:
            print(">> Video (metric_viewer) disabled in config.")

        # video_metric_dual_viewer
        vmd_cfg = (vids_cfg.get("metric_dual_viewer") or {})
        if yes(vmd_cfg.get("enabled", False)):
            cmd = [py, "src/visualization/video_metric_dual_viewer.py", *ident_either()]
            if yes(vmd_cfg.get("overlay_pitch", True)):
                cmd.append("--overlay-pitch")
            if yes(vmd_cfg.get("shade_clusters", False)):
                cmd.append("--shade-clusters")
            if yes(vmd_cfg.get("add_depth_cam", False)):
                cmd.append("--add-depth-cam")
            sh(cmd)
        else:
            print(">> Video (metric_dual_viewer) disabled in config.")

        # video_dem_vs_quat_pitch_roll
        vdq_cfg = (vids_cfg.get("dem_vs_quat_pitch_roll") or {})
        if yes(vdq_cfg.get("enabled", False)):  # default off
            cmd = [py, "src/visualization/video_dem_vs_quat_pitch_roll.py", *ident_either()]
            sh(cmd)
        else:
            print(">> Video (dem_vs_quat_pitch_roll) disabled in config.")
    else:
        print(">> Skipping visualizations (--skip-visuals).")

    print("\n Pipeline finished.")

if __name__ == "__main__":
    main()
