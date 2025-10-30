#!/usr/bin/env python3
"""
Export (id, name) pairs for GrandTour missions to JSON, robust to changing CLI output.

Strategy:
1) Fetch a mission listing (JSON if available, else text).
2) From the listing, collect "handles" (UUIDs, short tokens like ETH-1, slugs, ids).
3) For each handle, run: `klein mission info -m <HANDLE> [-p PROJECT]`
   and extract the canonical UUID and short name from the info table.
4) Filter/sort/limit and write JSON.

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

def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)

# ---------- Shell helpers ----------
def run_cmd(cmd: List[str]) -> Optional[str]:
    try:
        p = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return p.stdout
    except Exception:
        return None

def try_fetch_listing(project: Optional[str]) -> Optional[str]:
    """
    Try a few kleinkram variants. Prefer JSON forms first; return raw text either way.
    """
    candidates: List[List[str]] = []
    for base in (["klein", "mission", "list"], ["klein", "list", "missions"], ["klein", "mission"]):
        candidates.append(base + ["--format", "json"])
        candidates.append(base)
    # Try with project attached (harmless if ignored)
    tried: List[List[str]] = []
    for c in candidates:
        tried.append(c)
        if project:
            tried.append(c + ["-p", project])

    for cmd in tried:
        out = run_cmd(cmd)
        if out:
            return out
    return None

# ---------- Parsing the listing into generic "handles" ----------
def handles_from_json(data) -> List[str]:
    """
    Given parsed JSON (list/dict), extract any plausible handle to feed into `mission info`.
    Accept keys: id, uuid, mission_id, slug, name, short_name, shortName.
    """
    handles: List[str] = []

    def add(v: Optional[str]):
        if not isinstance(v, str):
            return
        v = v.strip()
        if not v:
            return
        handles.append(v)

    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("missions") if isinstance(data.get("missions"), list) else []
    else:
        items = []

    for it in items:
        if not isinstance(it, dict):
            continue
        # Prefer UUID if present, but collect all as fallbacks:
        add(it.get("uuid"))
        add(it.get("id"))
        add(it.get("mission_id"))
        add(it.get("slug"))
        add(it.get("short_name"))
        add(it.get("shortName"))
        add(it.get("name"))

    return handles

def handles_from_text(s: str) -> List[str]:
    """
    From plain text listing, extract any UUIDs and short tokens like ETH-1.
    """
    s = strip_ansi(s)
    handles: Set[str] = set()
    for line in s.splitlines():
        for u in UUID_RE.findall(line):
            handles.add(u)
        for m in SHORT_TOKEN_RE.findall(line):
            handles.add(m)
    return list(handles)

def listing_to_handles(s: str) -> List[str]:
    """
    Try JSON first; otherwise parse as text.
    """
    # Save a debug snapshot to help diagnose future changes
    try:
        Path(".cache").mkdir(exist_ok=True)
        Path(".cache/export_missions_last_listing.txt").write_text(s)
    except Exception:
        pass

    try:
        data = json.loads(s)
        hs = handles_from_json(data)
        if hs:
            return sorted(set(hs))
    except json.JSONDecodeError:
        pass
    return sorted(set(handles_from_text(s)))

# ---------- Query details and extract canonical (uuid, short) ----------
def run_info(handle: str, project: Optional[str]) -> Optional[str]:
    if not shutil.which("klein"):
        sys.exit("klein not found on PATH. Please install kleinkram.")
    cmd = ["klein", "mission", "info", "-m", handle]
    if project:
        cmd += ["-p", project]
    try:
        p = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return p.stdout
    except subprocess.CalledProcessError as e:
        print(f"[warn] mission info failed for {handle}: {e.stdout.strip()}", file=sys.stderr)
    except Exception as e:
        print(f"[warn] mission info failed for {handle}: {e}", file=sys.stderr)
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
    args = ap.parse_args()

    # Acquire listing text
    if args.source_file:
        listing = Path(args.source_file).read_text()
    else:
        if not shutil.which("klein"):
            sys.exit("klein not found on PATH. Install kleinkram or use --source-file.")
        listing = try_fetch_listing(args.project)
        if not listing:
            sys.exit("Could not obtain mission listing from kleinkram. Try:\n"
                     "  klein list missions > missions.txt\n"
                     "and rerun with --source-file missions.txt")

    # Turn listing into handles (UUIDs, names, slugs, etc.)
    handles = listing_to_handles(listing)
    if not handles:
        sys.exit("No missions found in the provided listing.")

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

    # Build items map by UUID (only keep entries where we found a UUID)
    by_uuid: Dict[str, Dict[str, str]] = {}
    for (u, sname) in results:
        if not u:
            continue
        if u not in by_uuid:
            by_uuid[u] = {"id": u}
        if sname:
            by_uuid[u]["name"] = sname

    if not by_uuid:
        # Provide a helpful debug hint
        dbg = Path(".cache/export_missions_last_listing.txt")
        hint = f"\n[debug] Saved last listing to {dbg}" if dbg.exists() else ""
        sys.exit("Resolved zero UUIDs from mission info. The listing may require login or the CLI format changed."
                 + hint)

    # Convert to list, filter/sort/limit
    items = list(by_uuid.values())

    if args.contains:
        items = [d for d in items if d.get("name") and args.contains in d["name"]]

    items.sort(key=lambda d: (0 if d.get("name") else 1, d.get("name", ""), d["id"]))

    if args.limit:
        items = items[: args.limit]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(items, indent=2, ensure_ascii=False))
    print(f"[ok] wrote {args.out} with {len(items)} missions")

if __name__ == "__main__":
    main()
