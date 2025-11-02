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

def sh(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)

def yes(x) -> bool:
    return bool(x) and str(x).lower() not in {"0", "false", "no", "off"}

# ---------- helpers for corruption handling ----------
def _resolve_short_folder(mission_id: str, mission_name: str | None, repo_root: Path) -> tuple[str, str]:
    """
    Return (short_folder, display_name) for the mission using config/missions.json.
    display_name prefers the alias (mission_name) if it maps to mission_id.
    """
    mj = repo_root / "config" / "missions.json"
    meta = yaml.safe_load(mj.read_text()) if mj.suffix in {".yaml", ".yml"} else __import__("json").loads(mj.read_text())
    alias_map = meta.get("_aliases", {})
    # If name given and maps to some id, prefer that id; else mission_id
    resolved_id = alias_map.get(mission_name, mission_id) if mission_name else mission_id
    entry = meta.get(resolved_id)
    if not entry:
        raise KeyError(f"Mission '{mission_name or mission_id}' not found in missions.json")
    short = entry["folder"]
    # choose display
    if mission_name and alias_map.get(mission_name) == resolved_id:
        display = mission_name
    else:
        # any alias pointing to this id
        rev = [a for a, mid in alias_map.items() if mid == resolved_id]
        display = rev[0] if rev else short
    return short, display

def _header_ok(p: Path) -> bool:
    try:
        with p.open("rb") as f:
            head = f.read(16)
        return head.startswith(b"#ROSBAG")
    except Exception:
        return False

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

    P = get_paths()
    REPO_ROOT: Path = P["REPO_ROOT"]
    RAW_ROOT: Path = P["RAW"]
    TABLES_ROOT: Path = P["TABLES"]

    # Find the mission folder in RAW to scan/delete bad bags if needed
    short_folder, display_name = _resolve_short_folder(mission_id, mission_name, REPO_ROOT)
    raw_dir = RAW_ROOT / short_folder
    tables_dir = TABLES_ROOT / short_folder


    # Helper: for stages that accept either --mission OR --mission-id (but not both)
    def ident_either() -> list[str]:
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
                dl_cmd = [py, "src/download_bag.py", "--mission-id", mission_id]
                if mission_name:
                    dl_cmd += ["--mission-name", mission_name]
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
