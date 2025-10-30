#!/usr/bin/env python3
"""
Run the full pipeline (run_pipeline.py) for multiple missions in parallel.

Usage:
  python src/run_all_missions.py --missions config/missions_to_run.json --max-workers 3 --already-downloaded

Notes:
  - Missions file: list of {"id": "...", "name": "..."}.
  - Each run calls: `python src/run_pipeline.py --mission-id <id> --mission-name <name>`.
  - Concurrency defaults to half of available cores.
  - You can use `--already-downloaded` to skip bag downloads.

"""

from __future__ import annotations
import argparse
import concurrent.futures
import subprocess
import sys
import json
import os
from pathlib import Path

def run_one(mission_id: str, mission_name: str | None, already_downloaded: bool, config: str | None) -> int:
    """Run one mission through the pipeline, return exit code."""
    py = sys.executable
    cmd = [py, "src/run_pipeline.py", "--mission-id", mission_id]
    if mission_name:
        cmd += ["--mission-name", mission_name]
    if already_downloaded:
        cmd.append("--already-downloaded")
    if config:
        cmd += ["--config", config]

    log_dir = Path("logs"); log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"{mission_name or mission_id[:8]}.log"
    with log_file.open("w") as f:
        f.write(f"$ {' '.join(cmd)}\n\n")
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        return result.returncode


def main():
    ap = argparse.ArgumentParser(description="Run pipeline for multiple missions in parallel.")
    ap.add_argument("--missions", required=True, help="JSON file with [{'id': ..., 'name': ...}, ...]")
    ap.add_argument("--config", default="config/pipeline.yaml", help="Config passed to run_pipeline.py")
    ap.add_argument("--max-workers", type=int, default=None,
                    help="Number of parallel workers (default: half of available cores)")
    ap.add_argument("--already-downloaded", action="store_true",
                    help="Skip download/extract/sync stages for all missions")
    args = ap.parse_args()

    with open(args.missions) as f:
        missions = json.load(f)

    if not isinstance(missions, list):
        raise ValueError("Missions file must contain a list of {id,name} objects")

    # Determine concurrency
    import multiprocessing
    max_workers = args.max_workers or max(1, multiprocessing.cpu_count() // 2)

    print(f"[info] Launching {len(missions)} missions with {max_workers} workers")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = []
        for m in missions:
            mission_id = m["id"]
            mission_name = m.get("name")
            fut = ex.submit(run_one, mission_id, mission_name, args.already_downloaded, args.config)
            futs.append((mission_name or mission_id, fut))

        for name, fut in futs:
            try:
                code = fut.result()
                status = "ok" if code == 0 else f"fail ({code})"
                print(f"[{status}] {name}")
            except Exception as e:
                print(f"[error] {name}: {e}")

if __name__ == "__main__":
    main()
