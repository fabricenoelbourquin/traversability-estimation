#!/usr/bin/env python3
"""
Download selected rosbags for a mission into data/raw/<mission_folder>/.

Arguments:
--mission-id: required, Mission UUID
--mission-name: optional, Optional alias (e.g., Heap_1)
--extra: optional, Extra bag kinds to include (e.g., adis locomotion dlio)
--dry-run: optional, Print commands, do not download

Examples:
  python src/download_bag.py --mission-id e97e35ad-dd7b-49c4-a158-95aba246520e
  python src/download_bag.py --mission-id e97e35ad-... --mission-name Heap_1
  python src/download_bag.py --mission-id ... --extra adis locomotion dlio --dry-run
  python src/download_bag.py --mission-id e97e35ad-dd7b-49c4-a158-95aba246520e --mission-name ETH-1 --dry-run 
"""

from __future__ import annotations
import argparse, json, os, re, shlex, subprocess, sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# --- paths helper ---
from utils.paths import get_paths

# Try to import YAML (PyYAML). If unavailable we'll use hardcoded fallbacks.
import yaml


P = get_paths()  # expects keys RAW, REPO_ROOT, (optionally) MISSIONS_JSON

BAG_SUFFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_(.+)\.bag$")

def repo_root() -> Path:
    # If helper didn’t provide it, infer from this file
    return P.get("REPO_ROOT", Path(__file__).resolve().parents[1])

def missions_json_path() -> Path:
    return P.get("MISSIONS_JSON", repo_root() / "config" / "missions.json")

# --- rosbags.yaml loader ---
def rosbags_yaml_path() -> Path:
    """
    Determine path to rosbags.yaml.
    - If utils.paths returns ROSBAGS_YAML, use that.
    - Else default to <REPO_ROOT>/config/rosbags.yaml
    """
    return Path(P.get("ROSBAGS_YAML", repo_root() / "config" / "rosbags.yaml"))

def _read_rosbags_yaml() -> Optional[dict]:
    """
    Load rosbags.yaml safely.
    Returns dict with keys:
      - defaults: List[str]
      - extras: Dict[str, str]  (alias -> bare suffix, e.g. "dlio" -> "dlio")
    or None if file missing/unreadable/pyyaml unavailable.
    """
    path = rosbags_yaml_path()
    if not path.exists():
        return None
    if yaml is None:
        print(f"[warn] PyYAML not installed; ignoring {path}. Using built-in defaults.", file=sys.stderr)
        return None
    try:
        cfg = yaml.safe_load(path.read_text()) or {}
        # Normalize shapes
        defaults = cfg.get("defaults", [])
        extras = cfg.get("extras", {})
        if not isinstance(defaults, list):
            print(f"[warn] 'defaults' in {path} must be a list; ignoring file.", file=sys.stderr)
            return None
        if not isinstance(extras, dict):
            extras = {}
        return {"defaults": defaults, "extras": extras}
    except Exception as e:
        print(f"[warn] Could not parse {path}: {e}. Using built-in defaults.", file=sys.stderr)
        return None

def _read_missions_json() -> Dict[str, dict]:
    """Load missions.json safely; return {} on any parse/missing error."""
    mj = missions_json_path()
    if not mj.exists():
        return {}
    try:
        return json.loads(mj.read_text())
    except Exception:
        print(f"[warn] Could not parse {mj}; treating as empty.", file=sys.stderr)
        return {}

def short_folder_from_uuid(mission_id: str, dest_root: Path) -> Tuple[str, Path]:
    """
    Use the first part of mission id for subfolder name, but:
    - If missions.json already has an entry for this mission_id, reuse its folder.
    - Otherwise choose the short prefix (first 8 chars) unless it is already
      claimed by a different mission in missions.json; only then use full id.
    """
    cur = _read_missions_json()

    # 1) If mission already known, reuse recorded folder (stable behavior)
    existing = cur.get(mission_id, {})
    if "folder" in existing and existing["folder"]:
        folder_name = existing["folder"]
        print(f"[info] Mission already exists; updating folders at '{folder_name}'.")
        return folder_name, dest_root / folder_name

    # 2) Propose short prefix
    short = mission_id.split("-")[0]

    # Is this short name claimed by a *different* mission in missions.json?
    claimed_by_other = False
    for mid, entry in cur.items():
        if mid == "_aliases":
            continue
        if isinstance(entry, dict) and entry.get("folder") == short and mid != mission_id:
            claimed_by_other = True
            break

    if claimed_by_other:
        # Real collision: fall back to full UUID
        print(f"[info] Short folder '{short}' is already used by another mission; using full id.")
        return mission_id, dest_root / mission_id

    # Otherwise we can safely use the short folder (even if it already exists on disk)
    return short, dest_root / short

def run(cmd: List[str]) -> subprocess.CompletedProcess:
    print("[cmd]", " ".join(shlex.quote(c) for c in cmd))
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# --- pattern mode (skip listing; server resolves globs) ---
def make_patterns_for_defaults(extras: List[str]) -> List[str]:
    """
    Create glob patterns for the default set and any extras.
    Examples: '*_anymal_command.bag', '*_cpt7_ie_{tc,rt}.bag', '*_tf_minimal.bag'
    """
    pats: List[str] = []

    # If YAML provided, use it; otherwise use the previous built-in defaults.
    cfg = _read_rosbags_yaml()

    if cfg is not None:
        # YAML-driven: defaults are full patterns already.
        defaults = cfg.get("defaults", [])
        pats += defaults

        # Extras mapping (alias -> bare suffix or full pattern). We treat a
        # value as either a bare suffix "dlio" (we’ll expand to '*_dlio.bag')
        # or already a glob pattern (contains '*' or '?').
        yaml_extras: Dict[str, str] = cfg.get("extras", {})
        for ex in extras:
            key = ex.lower()
            val = yaml_extras.get(key)
            if not val:
                # allow raw suffix too if user typed something not in map
                val = key
            if "*" in val or "?" in val:
                pats.append(val)
            else:
                pats.append(f"*_{val}.bag")
    else:
        # Built-in fallback (preserves your current behavior)
        # Core command/state
        pats += ["*_anymal_command.bag", "*_anymal_state.bag"]

        # GPS/TF
        pats += ["*_cpt7_ie_tc.bag", "*_cpt7_ie_rt.bag", "*_tf_minimal.bag"]

        # IMU preference chain – ask for all; server returns those that exist
        pats += ["*_anymal_imu.bag", "*_adis.bag", "*_stim320_imu.bag"]

        # Extras mapping (reuse EXTRA_MAP)
        for ex in extras:
            key = ex.lower()
            suf = EXTRA_MAP.get(key, key)  # allow raw suffix too
            pats.append(f"*_{suf}.bag")

    # Deduplicate while preserving order
    seen = set()
    out: List[str] = []
    for p in pats:
        if p not in seen:
            seen.add(p); out.append(p)
    return out

def ensure_dirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)

def update_missions_json(
    mission_id: str,
    folder: str,
    mission_name: str | None,
    bags: List[str] | None,
) -> None:
    mj = missions_json_path()
    mj.parent.mkdir(parents=True, exist_ok=True)

    cur: Dict[str, dict] = {}
    if mj.exists():
        try:
            cur = json.loads(mj.read_text())
        except Exception:
            print(f"[warn] Could not parse {mj}; rewriting clean.", file=sys.stderr)
            cur = {}

    entry = cur.get(mission_id, {})
    entry.update({"folder": folder, "name": mission_name})

    # Only store concrete bag names (no globs). If none, omit the key.
    bags_to_store = []
    if bags:
        for b in bags:
            if "*" not in b and "?" not in b:
                bags_to_store.append(b)
    if bags_to_store:
        entry["bags"] = sorted(bags_to_store)
    else:
        # ensure we don't keep old bag lists if we only have patterns now
        entry.pop("bags", None)

    cur[mission_id] = entry

    # Alias map (so you can later use --mission-name Heap_1)
    if mission_name:
        aliases = cur.get("_aliases", {})
        aliases[mission_name] = mission_id
        cur["_aliases"] = aliases

    atomic_write_json(mj, cur)
    print(f"[info] Updated {mj}")

def download_selected(mission_id: str, dest: Path, selected_patterns: List[str], dry: bool) -> None:
    if not selected_patterns:
        raise RuntimeError("No patterns provided to download.")
    ensure_dirs(dest)

    # Pass patterns directly to the server; it will resolve them to matching files.
    cmd = ["klein", "download", "-m", mission_id, "--dest", str(dest), *selected_patterns]
    if dry:
        print("[dry-run]", " ".join(shlex.quote(c) for c in cmd))
    else:
        cp = run(cmd)
        if cp.returncode != 0:
            print(cp.stdout)
            print(cp.stderr, file=sys.stderr)
            raise RuntimeError(f"'klein download' failed for patterns: {selected_patterns}")
    print("[info] Downloaded (pattern mode).")

def main():
    ap = argparse.ArgumentParser(description="Download selected rosbags for a mission.")
    ap.add_argument("--mission-id", required=True, help="Mission UUID")
    ap.add_argument("--mission-name", default=None, help="Optional alias (e.g., Heap-1)")
    ap.add_argument("--extra", nargs="*", default=[], help="Extra bag kinds to include (e.g., adis locomotion dlio)")
    ap.add_argument("--dry-run", action="store_true", help="Print commands, do not download")
    args = ap.parse_args()

    raw_root: Path = P["RAW"]
    ensure_dirs(raw_root)

    # Choose folder deterministically based on missions.json (see rules above)
    folder_name, dest_dir = short_folder_from_uuid(args.mission_id, raw_root)
    ensure_dirs(dest_dir)

    # Pattern-only flow — no listing
    patterns = make_patterns_for_defaults(args.extra)
    print("[info] Patterns to request:")
    for p in patterns:
        print("  -", p)

    download_selected(args.mission_id, dest_dir, patterns, args.dry_run)

    # Store mission id, name and chosen folder only (no globs)
    update_missions_json(args.mission_id, folder_name, args.mission_name, bags=None)
    print("[done] Ready at:", dest_dir)

if __name__ == "__main__":
    main()
