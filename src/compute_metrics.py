"""
Compute selected metrics on a synced parquet (per mission) and write back.

By default, the script:
  - locates data/synced/<mission>/synced_<Hz>Hz.parquet (or latest if --hz omitted)
  - loads config values from:
      - config/metrics.yaml
  - appends requested metrics as new columns and writes back to the same parquet
    (or to --out if provided)

Example usage:
python src/compute_metrics.py --mission ETH-1 \
    --hz 10 \
    --metrics speed_error_abs speed_tracking_score
"""

#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml, pandas as pd

# paths helper
from utils.paths import get_paths
from utils.missions import resolve_mission
from utils.cli import add_mission_arguments, add_hz_argument, resolve_mission_from_args
from utils.synced import resolve_synced_parquet, infer_hz_from_path

from metrics import compute, REGISTRY

def load_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text()) if p.exists() else {}

def merge_dicts(base: dict, override: dict) -> dict:
    """Deep merge two dicts without mutating inputs; override wins."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_dicts(out[k], v)
        else:
            out[k] = v
    return out

def main():
    ap = argparse.ArgumentParser(description="Compute metrics on a synced parquet.")
    add_mission_arguments(ap)
    add_hz_argument(ap, help_text="Pick synced_<Hz>Hz.parquet (default: latest)")
    ap.add_argument("--metrics", nargs="*", default=["speed_error_abs", "speed_tracking_score"],
                    help=f"Metric names to compute. Available: {list(REGISTRY)}")
    ap.add_argument("--out", default=None, help="Optional output path (default: overwrite input parquet)")
    args = ap.parse_args()

    P = get_paths()
    mp = resolve_mission_from_args(args, P)
    sync_dir = mp.synced
    in_parquet = resolve_synced_parquet(sync_dir, args.hz)

    cfg_path = P["REPO_ROOT"] / "config" / "metrics.yaml"
    cfg_private_path = P["REPO_ROOT"] / "config" / "metrics.private.yaml"
    cfg = merge_dicts(load_yaml(cfg_path), load_yaml(cfg_private_path))  # for max_speed_mps, etc.

    # Determine metric names from CLI or config (config wins if CLI left as default)
    cfg_metrics = cfg.get("metrics", {})
    cfg_metric_names = cfg_metrics.get("names", [])
    metric_names = cfg_metric_names if args.metrics == ["speed_error_abs", "speed_tracking_score"] and cfg_metric_names else args.metrics

    # Decide output policy: default separate file unless config says otherwise
    separate_output = bool(cfg_metrics.get("separate_output", True))

    df = pd.read_parquet(in_parquet)
    df2 = compute(df, metric_names, cfg)

    # Choose output path
    if args.out:
        out_path = Path(args.out)
    else:
        if separate_output:
            hz_val = args.hz if args.hz is not None else (infer_hz_from_path(in_parquet) or 0)
            out_path = in_parquet.with_name(f"synced_{hz_val}Hz_metrics.parquet")
        else:
            out_path = in_parquet  # append to original

    df2.to_parquet(out_path)
    print(f"[ok] wrote {out_path}")

if __name__ == "__main__":
    main()
