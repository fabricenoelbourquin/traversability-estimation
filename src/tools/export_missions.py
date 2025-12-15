#!/usr/bin/env python3
"""
Export (id, name) pairs for missions to JSON.

Strategy:
1) Fetch a mission listing once via `klein list missions [-p PROJECT]` (plain text).
2) From the listing, collect "handles" (UUIDs, short tokens like ETH-1, slugs, ids).
3) For each handle, run: `klein mission info -m <HANDLE> [-p PROJECT]`
   and extract the canonical UUID and short name from the info table.
4) Filter/sort/limit and write/merge JSON.

Usage:
  python src/tools/export_missions.py --out config/missions_to_run.json \
      --project GrandTourDataset --max-workers 8
  python src/tools/export_missions.py --source-file missions.txt --out config/missions_to_run.json
"""

from __future__ import annotations
import argparse, json, re, subprocess, sys, shutil, os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

# ---------- Regexes ----------
UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE)

# "Short name" token like ETH-1, PIL-2, KÄB-2 (allow Unicode letters):
SHORT_TOKEN_RE = re.compile(r"\b([^\s\|:│]+-\d+)\b", re.UNICODE)

# Info table row e.g. "Short Name | ETH-1"
SHORT_LABEL_RE = re.compile(r"(?i)\bshort\s*name\b\s*(?:[│|:])\s*([^\s\|:│]+)", re.UNICODE)

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

def _norm_uuid(u: str) -> str:
    return u.lower()

def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)

# ---------- Naming helpers ----------
def _split_prefix_and_number(token: str) -> Tuple[str, Optional[int]]:
    """
    Given a token like 'srb1' or 'seilbahn', return (prefix, number|None).
    """
    # Allow any Unicode letter sequence followed by optional digits
    m = re.match(r"([^\W\d_]+)(\d+)?", token, re.UNICODE)
    if not m:
        return token[:4].upper() if token else "", None
    prefix = m.group(1).upper()
    num = m.group(2)
    return prefix[:4], int(num) if num else None

def derive_short_name(raw_short: Optional[str], mission_name: Optional[str],
                      used_names: Dict[str, str], for_id: str) -> str:
    """
    - Prefer provided short name (e.g., from mission info) if its letter prefix is <=5 chars.
    - Else derive from mission_name:
        * Drop leading digits/underscores (e.g., date).
        * Take first token after that; split letters+digits.
        * Prefix is first up-to-4 letters; number is trailing digits if present.
    - If no usable number, pick the smallest available integer not already used for this prefix.
    Names are uppercased and de-duplicated across IDs.
    """
    def claim(prefix: str, preferred_num: Optional[int]) -> str:
        prefix = prefix[:5] if prefix else "MISS"
        if preferred_num is not None:
            cand = f"{prefix}-{preferred_num}"
            owner = used_names.get(cand)
            if owner is None or owner == for_id:
                used_names[cand] = for_id
                return cand
        n = preferred_num if preferred_num is not None else 1
        while True:
            cand = f"{prefix}-{n}"
            owner = used_names.get(cand)
            if owner is None or owner == for_id:
                used_names[cand] = for_id
                return cand
            n += 1

    if raw_short:
        cand = raw_short.strip().upper().replace("_", "-")
        m = re.match(r"([^\W\d_]+)(?:-?)(\d+)?$", cand, re.UNICODE)
        if m and len(m.group(1)) <= 5:
            prefix = m.group(1)
            num = int(m.group(2)) if m.group(2) else None
            return claim(prefix, num)

    token = ""
    if mission_name:
        trimmed = re.sub(r"^[0-9_]+", "", mission_name)
        for part in trimmed.split("_"):
            if part:
                token = part
                break

    prefix, num = _split_prefix_and_number(token) if token else ("MISS", None)
    if not prefix:
        prefix = "MISS"
    return claim(prefix, num)

# ---------- Shell helpers ----------
def run_cmd(cmd: List[str]) -> Optional[str]:
    try:
        env = os.environ.copy()
        # Help avoid truncated table output from klein; harmless if ignored.
        env.setdefault("COLUMNS", "2000")
        p = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        return p.stdout
    except Exception:
        return None

def fetch_listing(project: Optional[str]) -> Optional[str]:
    """
    Fetch mission listing as plain text.
    """
    base = ["klein", "list", "missions"]
    cmds: List[List[str]] = []
    project_flags: List[List[str]] = []
    if project:
        project_flags = [["--project", project], ["-p", project]]
    else:
        project_flags = [[]]

    for pf in project_flags:
        cmds.append(base + pf)

    for cmd in cmds:
        out = run_cmd(cmd)
        if out:
            return out
    return None

# ---------- Parsing the listing into generic "handles" ----------
def handles_from_text(s: str) -> Tuple[List[str], Dict[str, str]]:
    """
    From plain text listing, extract any UUIDs and short tokens like ETH-1.
    If a line contains a UUID, skip short tokens from that line to avoid truncated fragments.
    Also try to capture the mission name column from table-like output for UUID rows.
    """
    s = strip_ansi(s)
    handles: Set[str] = set()
    names: Dict[str, str] = {}
    for line in s.splitlines():
        has_uuid = False
        uuids = list(UUID_RE.findall(line))
        if uuids:
            has_uuid = True
            for u in uuids:
                handles.add(_norm_uuid(u))
            # Attempt to parse table columns: project | name | id | ...
            parts = [p.strip() for p in re.split(r"[│|]", line) if p.strip()]
            if len(parts) >= 3:
                mission_name = parts[1]
                for u in uuids:
                    names.setdefault(_norm_uuid(u), mission_name)
        if has_uuid:
            continue
        if "…" in line:
            # Likely truncated table output; skip short tokens from this line.
            continue
        for m in SHORT_TOKEN_RE.findall(line):
            handles.add(m)
    return list(handles), names

def listing_to_handles_and_names(s: str) -> Tuple[List[str], Dict[str, str], str]:
    """
    Parse plain text listing. Return handles, mapping UUID -> mission name, and format string.
    """
    # Save a debug snapshot to help diagnose future changes
    try:
        Path(".cache").mkdir(exist_ok=True)
        Path(".cache/export_missions_last_listing.txt").write_text(s)
    except Exception:
        pass

    hs, names = handles_from_text(s)
    return sorted(set(hs)), names, "text"

# ---------- Query details and extract canonical (uuid, short) ----------
def run_info(handle: str, project: Optional[str]) -> Optional[str]:
    if not shutil.which("klein"):
        sys.exit("klein not found on PATH. Please install kleinkram.")
    base = ["klein", "mission", "info", "-m", handle]
    variants: List[List[str]] = []
    if project:
        variants.append(base + ["--project", project])
        variants.append(base + ["-p", project])
    else:
        variants.append(base)
    last_err: Optional[str] = None
    for cmd in variants:
        try:
            p = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            return p.stdout
        except subprocess.CalledProcessError as e:
            last_err = e.stdout.strip()
        except Exception as e:
            last_err = str(e)
    if last_err:
        print(f"[warn] mission info failed for {handle}: {last_err}", file=sys.stderr)
    return None

def extract_uuid_and_short(info_text: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not info_text:
        return None, None
    t = strip_ansi(info_text)
    # UUID anywhere in the info text
    u = None
    m = UUID_RE.search(t)
    if m:
        u = m.group(0).lower()
    # Short name: labeled row preferred
    sname = None
    m2 = SHORT_LABEL_RE.search(t)
    if m2:
        sname = m2.group(1).strip().upper()
    else:
        # Fallback: first plausible token in first lines
        for line in t.splitlines()[:80]:
            m3 = SHORT_TOKEN_RE.search(line)
            if m3:
                sname = m3.group(0).upper()
                break
    return u, sname

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="config/missions_to_run.json")
    ap.add_argument("--contains", default=None, help="substring to filter short names (after resolution)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--source-file", default=None, help="parse from saved listing (text or JSON)")
    ap.add_argument("--project", "-p", default=None, help="Project for 'klein mission info'")
    ap.add_argument("--max-workers", type=int, default=8, help="Parallel info lookups")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output file instead of appending/merging")
    args = ap.parse_args()

    # Acquire listing text
    if args.source_file:
        listing = Path(args.source_file).read_text()
    else:
        if not shutil.which("klein"):
            sys.exit("klein not found on PATH. Install kleinkram or use --source-file.")
        listing = fetch_listing(args.project)
        if not listing:
            sys.exit("Could not obtain mission listing from kleinkram. Try:\n"
                     "  COLUMNS=2000 klein list missions [-p PROJECT] > missions.txt\n"
                     "and rerun with --source-file missions.txt")

    # Turn listing into handles (UUIDs, names, slugs, etc.) and capture mission names if present
    handles, listing_names, listing_format = listing_to_handles_and_names(listing)
    if not handles:
        sys.exit("No missions found in the provided listing.")
    print(f"[info] Parsed listing using {listing_format.upper()} format; {len(handles)} handle(s) found.")

    # Query each handle to resolve canonical (uuid, short)
    # Deduplicate work: if listing produced duplicates, remove
    handles = sorted(set(handles))

    # Resolve in parallel
    results: List[Tuple[Optional[str], Optional[str]]] = [None] * len(handles)  # type: ignore
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        futs = {ex.submit(run_info, h, args.project): i for i, h in enumerate(handles)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                info = fut.result()
                u, sname = extract_uuid_and_short(info)
                results[i] = (u, sname)
            except Exception as e:
                print(f"[warn] failed resolving {handles[i]}: {e}", file=sys.stderr)
                results[i] = (None, None)

    # Build items map by UUID (use handle UUID when info fails)
    by_uuid: Dict[str, Dict[str, str]] = {}
    for idx, (u, sname) in enumerate(results):
        handle = handles[idx]
        if not u and isinstance(handle, str) and UUID_RE.fullmatch(handle):
            u = _norm_uuid(handle)
        if not u:
            continue
        if u not in by_uuid:
            by_uuid[u] = {"id": u}
        if sname:
            by_uuid[u]["name"] = sname
        # Attach mission name from listing if available (used for fallback naming)
        if u in listing_names:
            by_uuid[u]["_mission_name"] = listing_names[u]

    if not by_uuid:
        # Provide a helpful debug hint
        dbg = Path(".cache/export_missions_last_listing.txt")
        hint = f"\n[debug] Saved last listing to {dbg}" if dbg.exists() else ""
        sys.exit("Resolved zero UUIDs from mission info. The listing may require login or the CLI format changed."
                 + hint)

    # Convert to list, filter/sort/limit
    items = list(by_uuid.values())

    # Merge with existing file unless overwrite is requested (also collect used names)
    existing: Dict[str, Dict[str, str]] = {}
    used_names: Dict[str, str] = {}
    out_path = Path(args.out)
    if not args.overwrite and out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            if isinstance(prev, list):
                for entry in prev:
                    if isinstance(entry, dict) and isinstance(entry.get("id"), str):
                        existing[entry["id"]] = entry
                        n = entry.get("name")
                        if isinstance(n, str):
                            used_names[n] = entry["id"]
        except Exception:
            print(f"[warn] Could not read existing {out_path}; starting fresh", file=sys.stderr)

    def release_names_for_id(mission_id: str) -> None:
        for k, v in list(used_names.items()):
            if v == mission_id:
                used_names.pop(k, None)

    # Resolve final short names (<=5 letters + dash + number) with fallback from mission name
    for item in items:
        release_names_for_id(item["id"])
        raw_short = item.get("name")
        mission_label = item.get("_mission_name")
        final_short = derive_short_name(raw_short, mission_label, used_names, item["id"])
        item["name"] = final_short
        item.pop("_mission_name", None)

    if args.contains:
        items = [d for d in items if d.get("name") and args.contains in d["name"]]

    items.sort(key=lambda d: (0 if d.get("name") else 1, d.get("name", ""), d["id"]))

    if args.limit:
        items = items[: args.limit]

    merged: Dict[str, Dict[str, str]] = dict(existing)
    for item in items:
        merged[item["id"]] = item  # new run overrides existing entry with same id

    final_items = list(merged.values())
    final_items.sort(key=lambda d: (0 if d.get("name") else 1, d.get("name", ""), d["id"]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(final_items, indent=2, ensure_ascii=False))
    print(f"[ok] wrote {out_path} with {len(final_items)} missions")

if __name__ == "__main__":
    main()
