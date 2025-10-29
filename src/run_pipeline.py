#!/usr/bin/env python3
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
from typing import List
import yaml

def sh(cmd: List[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)

def yes(x) -> bool:
    return bool(x) and str(x).lower() not in {"0", "false", "no", "off"}

def main():
    p = argparse.ArgumentParser(description="Run pipeline for a single mission (sequential).")
    p.add_argument("--mission-id", type=str, required=False, help="Mission UUID (REQUIRED).")
    p.add_argument("--mission-name", type=str, default=None, help="Optional alias (e.g., ETH-1).")
    p.add_argument("--config", type=str, default="config/pipeline.yaml", help="YAML config path")
    p.add_argument("--already-downloaded", action="store_true",
                   help="Skip download + extract + sync; recompute metrics and make plots.")
    # optional fine-grained skips
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--skip-extract", action="store_true")
    p.add_argument("--skip-sync", action="store_true")
    p.add_argument("--skip-swissimg", action="store_true")
    p.add_argument("--skip-cluster", action="store_true")
    p.add_argument("--skip-visuals", action="store_true")
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f) or {}

    # ID required (from CLI or YAML)
    mission_id = args.mission_id or cfg.get("mission_id")
    mission_name = args.mission_name or cfg.get("mission_name")
    if not mission_id:
        raise ValueError("Missing required --mission-id (or set mission_id in the YAML).")

    # Helper: for stages that accept either --mission OR --mission-id (but not both)
    def ident_either() -> List[str]:
        if mission_name:
            return ["--mission", mission_name]
        return ["--mission-id", mission_id]

    # Stage toggles
    skip_download = args.skip_download or args.already_downloaded
    skip_extract  = args.skip_extract  or args.already_downloaded
    skip_sync     = args.skip_sync     or args.already_downloaded
    skip_swissimg = args.skip_swissimg
    skip_cluster  = args.skip_cluster
    skip_visuals  = args.skip_visuals

    py = sys.executable

    # 1) Download (this script accepts id + name together)
    if not skip_download:
        cmd = [py, "src/download_bag.py", "--mission-id", mission_id]
        if mission_name:
            cmd += ["--mission-name", mission_name]
        sh(cmd)
    else:
        print(">> Skipping download (already-downloaded / skip-download).")

    # 2) Extract (must pass ONE of the two)
    if not skip_extract:
        sh([py, "src/extract_bag.py", *ident_either()])
    else:
        print(">> Skipping extract (already-downloaded / skip-extract).")

    # 3) Sync streams
    if not skip_sync:
        sh([py, "src/sync_streams.py", *ident_either()])
    else:
        print(">> Skipping sync (already-downloaded / skip-sync).")

    # 4) Compute metrics
    sh([py, "src/compute_metrics.py", *ident_either()])

    # 5) SwissImage patch
    if not skip_swissimg:
        sh([py, "src/get_swissimg_patch.py", *ident_either()])
    else:
        print(">> Skipping SwissImage patch.")

    # 6) Infer cluster map
    if not skip_cluster:
        sh([py, "src/infer_cluster_map.py", *ident_either()])
    else:
        print(">> Skipping cluster inference.")

    # 7–8) Visualizations
    if not skip_visuals:
        vis = cfg.get("visuals", {})

        ts_cfg = (vis.get("timeseries") or {})
        if yes(ts_cfg.get("enabled", True)):
            cmd = [py, "src/visualization/plot_timeseries.py", *ident_either()]
            if yes(ts_cfg.get("with_speeds", False)):
                cmd.append("--with-speeds")
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
    else:
        print(">> Skipping visualizations (--skip-visuals).")

    print("\n Pipeline finished.")

if __name__ == "__main__":
    main()
