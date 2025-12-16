#!/usr/bin/env python3
"""
Download selected rosbags for a mission into data/raw/<mission_folder>/.

Arguments:
--mission-id: required, Mission UUID
--mission-name: optional, Optional alias (e.g., HEAP-1)
--extra: optional, Extra bag kinds to include (e.g., adis locomotion dlio)
--dry-run: optional, Print commands, do not download

Examples:
  python src/download_bag.py --mission-id e97e35ad-dd7b-49c4-a158-95aba246520e --mission-name ETH-1
  python src/download_bag.py --mission-id ... --extra adis locomotion dlio --dry-run
  python src/download_bag.py --mission-id e97e35ad-dd7b-49c4-a158-95aba246520e --mission-name ETH-1 --dry-run
  python src/download_bag.py \
  --mission-id e97e35ad-dd7b-49c4-a158-95aba246520e \
  --mission-name ETH-1 \
  --extra camera depth
"""

from __future__ import annotations
import argparse, json, os, re, shlex, subprocess, sys
from pathlib import Path
from argparse import BooleanOptionalAction
import yaml

# --- paths helper ---
from utils.paths import get_paths
from utils.missions import missions_json_path, load_missions_file, upsert_mission_entry, pick_folder_name

P = get_paths()  # expects keys RAW, REPO_ROOT, (optionally) MISSIONS_JSON

EXIT_NO_GPS = 20

def repo_root() -> Path:
    # If helper didn’t provide it, infer from this file
    return P.get("REPO_ROOT", Path(__file__).resolve().parents[1])

# --- rosbags.yaml loader ---
def rosbags_yaml_path() -> Path:
    """
    Determine path to rosbags.yaml.
    - If utils.paths returns ROSBAGS_YAML, use that.
    - Else default to <REPO_ROOT>/config/rosbags.yaml
    """
    return Path(P.get("ROSBAGS_YAML", repo_root() / "config" / "rosbags.yaml"))

def _read_rosbags_yaml() -> dict:
    """
    Load rosbags.yaml safely.
    Returns dict with keys:
      - defaults: List[str]
      - extras: Dict[str, str]  (alias -> pattern/suffix, e.g. "camera" -> "hdr_front"), additional bags to download
    or None if file missing/unreadable/pyyaml unavailable.
    """
    path = rosbags_yaml_path()
    if not path.exists():
        print(f"[error] Required YAML not found: {path}", file=sys.stderr)
        sys.exit(2)

    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        print(f"[error] Could not parse {path}: {e}", file=sys.stderr)
        sys.exit(2)

    defaults = cfg.get("defaults", [])
    extras = cfg.get("extras", {})

    if not isinstance(defaults, list):
        print(f"[error] 'defaults' in {path} must be a list of glob patterns.", file=sys.stderr)
        sys.exit(2)
    if not isinstance(extras, dict):
        print(f"[error] 'extras' in {path} must be a mapping (alias -> pattern|suffix).", file=sys.stderr)
        sys.exit(2)

    return {"defaults": defaults, "extras": extras}

def _read_missions_json() -> dict[str, dict]:
    """Load missions.json safely; return {} on any parse/missing error."""
    mj = missions_json_path(P)
    return load_missions_file(mj)

def short_folder_from_uuid(mission_id: str, dest_root: Path) -> tuple[str, Path]:
    """
    Use the first part of mission id for subfolder name, but:
    - If missions.json already has an entry for this mission_id, reuse its folder.
    - Otherwise choose the short prefix (first 8 chars) unless it is already
      claimed by a different mission in missions.json; only then use full id.
    """
    cur = _read_missions_json()

    folder_name = pick_folder_name(mission_id, cur)

    existing = cur.get(mission_id, {})
    if isinstance(existing, dict) and existing.get("folder"):
        print(f"[info] Mission already exists; updating folders at '{folder_name}'.")
    elif folder_name != mission_id.split("-")[0]:
        print(f"[info] Short folder '{mission_id.split('-')[0]}' is already used by another mission; using full id.")

    return folder_name, dest_root / folder_name

def run(cmd: list[str]) -> subprocess.CompletedProcess:
    print("[cmd]", " ".join(shlex.quote(c) for c in cmd))
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# --- pattern mode (skip listing; server resolves globs) ---
def make_patterns_for_defaults(extras: list[str]) -> list[str]:
    """
    Create glob patterns from YAML defaults + requested extras.
    - defaults: already glob patterns
    - extras: alias -> either a glob or a bare suffix (we expand to '*_<suffix>.bag' plus numbered variants)
    """
    cfg = _read_rosbags_yaml()
    pats: list[str] = list(cfg["defaults"])

    yaml_extras: dict[str, str] = cfg["extras"]
    for ex in extras:
        key = ex.lower()
        val = yaml_extras.get(key, key)  # allow raw suffix as a convenience
        if "*" in val or "?" in val:
            pats.append(val)
        else:
            base = f"*_{val}"
            pats.append(f"{base}.bag")
            pats.append(f"{base}_*.bag")  # allow numbered splits (e.g., *_hdr_front_0.*)

    # Expand .bag patterns to also request .mcap variants.
    expanded: list[str] = []
    for pat in pats:
        expanded.append(pat)
        # add trailing-wildcard variant when pattern ends with .bag but not already *.bag
        if pat.endswith(".bag") and not pat.endswith("*.bag"):
            expanded.append(pat[:-4] + "*.bag")
        if ".bag" in pat:
            expanded.append(pat.replace(".bag", ".mcap"))
            if pat.endswith(".bag") and not pat.endswith("*.bag"):
                expanded.append(pat[:-4] + "*.mcap")

    # de-duplicate preserve order
    seen, out = set(), []
    for p in expanded:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

def ensure_dirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def download_selected(mission_id: str, dest: Path, selected_patterns: list[str], dry: bool) -> None:
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

def _gps_files_in_dir(d: Path) -> list[Path]:
    gps: list[Path] = []
    for suf in (".bag", ".mcap"):
        gps.extend(d.glob(f"*_cpt7_ie_tc*{suf}"))
        gps.extend(d.glob(f"*_cpt7_ie_rt*{suf}"))
    # ignore macOS AppleDouble
    return [p for p in gps if not p.name.startswith("._")]

def _cleanup_downloaded_bags(dest_dir: Path, remove_folder: bool) -> int:
    """Remove downloaded bag/mcap files (skip AppleDouble). Optionally remove folder if it ends up empty."""
    removed = 0
    for p in list(dest_dir.glob("*.bag")) + list(dest_dir.glob("*.mcap")):
        if p.name.startswith("._"):
            continue
        try:
            p.unlink()
            removed += 1
        except Exception as e:
            print(f"[warn] Could not remove {p}: {e}", file=sys.stderr)
    if remove_folder:
        try:
            # remove folder iff empty
            next(dest_dir.iterdir())
        except StopIteration:
            try:
                dest_dir.rmdir()
                print(f"[skip] Removed empty folder {dest_dir}", file=sys.stderr)
            except Exception as e:
                print(f"[warn] Could not remove folder {dest_dir}: {e}", file=sys.stderr)
        except FileNotFoundError:
            pass
    return removed

def main():
    ap = argparse.ArgumentParser(description="Download selected rosbags for a mission.")
    ap.add_argument("--mission-id", required=True, help="Mission UUID")
    ap.add_argument("--mission-name", default=None, help="Optional alias (e.g., Heap-1)")
    ap.add_argument("--extra", nargs="*", default=[], help="Extra bag kinds to include (e.g., adis locomotion dlio)")
    ap.add_argument("--dry-run", action="store_true", help="Print commands, do not download")

    ap.add_argument("--require-gps", action=BooleanOptionalAction, default=True,
                    help="If set (default), exit with code 20 when no GPS bag is present.")
    ap.add_argument("--cleanup-on-missing-gps", action=BooleanOptionalAction, default=True,
                    help="If require-gps and GPS missing, remove downloaded files before exiting.")
    ap.add_argument("--remove-folder-on-missing-gps", action=BooleanOptionalAction, default=True,
                    help="If GPS missing and folder becomes empty, remove the mission folder.")
    args = ap.parse_args()

    try:
        raw_root: Path = P["RAW"]
    except KeyError as e:
        raise SystemExit("[error] paths config missing 'RAW' directory") from e
    ensure_dirs(raw_root)

    # Choose folder deterministically based on missions.json (see rules above)
    folder_name, dest_dir = short_folder_from_uuid(args.mission_id, raw_root)
    ensure_dirs(dest_dir)

    # Pattern-only flow — no listing
    patterns = make_patterns_for_defaults(args.extra)
    print("[info] Patterns to request:")
    for p in patterns:
        print("  -", p)

    # --- Phase 1: GPS-first probe ---
    gps_patterns = [p for p in patterns if ("cpt7_ie_tc" in p or "cpt7_ie_rt" in p)]
    non_gps_patterns = [p for p in patterns if p not in gps_patterns]

    if args.require_gps and not args.dry_run:
        if gps_patterns:
            print("[info] Probing for GPS bags first…")
            download_selected(args.mission_id, dest_dir, gps_patterns, dry=False)
        else:
            print("[warn] No GPS patterns in configuration; cannot probe before full download.", file=sys.stderr)

        if len(_gps_files_in_dir(dest_dir)) == 0:
            print("[skip] No GPS bag found (expected *_cpt7_ie_tc.bag or *_cpt7_ie_rt.bag).", file=sys.stderr)
            print(f"[skip] Mission {args.mission_name or args.mission_id} will be skipped.", file=sys.stderr)
            if args.cleanup_on_missing_gps:
                removed = _cleanup_downloaded_bags(dest_dir, args.remove_folder_on_missing_gps)
                print(f"[skip] Cleaned up {removed} bag(s) in {dest_dir}", file=sys.stderr)
            # Do NOT update missions.json in this case (we didn’t really import usable data)
            sys.exit(EXIT_NO_GPS)

    # --- Phase 2: Download the rest (including GPS if require-gps=False or dry-run) ---
    if args.dry_run:
        # In dry-run we can’t know presence; just show both phases as “would run”
        if gps_patterns:
            print("[dry-run] Would run GPS probe download:")
            print("          ", " ".join(["klein", "download", "-m", args.mission_id, "--dest", str(dest_dir), *gps_patterns]))
        print("[dry-run] Would run full/non-GPS download:")
        pats = non_gps_patterns if args.require_gps else patterns
        print("          ", " ".join(["klein", "download", "-m", args.mission_id, "--dest", str(dest_dir), *pats]))
    else:
        pats = non_gps_patterns if args.require_gps else patterns
        if pats:
            download_selected(args.mission_id, dest_dir, pats, dry=False)

    # Record mission metadata only when proceeding (i.e., not skipped)
    mj_path = missions_json_path(P)
    upsert_mission_entry(mj_path, args.mission_id, folder_name, args.mission_name)
    print(f"[info] Updated {mj_path}")
    print("[done] Ready at:", dest_dir)

if __name__ == "__main__":
    main()
