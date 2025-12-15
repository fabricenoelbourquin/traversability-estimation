# utils/missions.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
from functools import lru_cache
from typing import Mapping

@dataclass(frozen=True)
class MissionPaths:
    tables: Path
    synced: Path
    raw: Path
    mission_id: str
    folder: str
    display: str
    maps: str

class MissionNotFound(KeyError):
    pass

@lru_cache(maxsize=1)
def _load_missions(missions_json: Path) -> tuple[Mapping[str, dict], Mapping[str, str]]:
    if not missions_json.exists():
        raise FileNotFoundError(f"{missions_json} not found.")
    meta = json.loads(missions_json.read_text())
    aliases = meta.get("_aliases", {})
    return meta, aliases

def pick_folder_name(mission_id: str, missions_meta: Mapping[str, dict]) -> str:
    """
    Choose a folder name for a mission id.
    - Reuse existing folder if mission already in missions_meta.
    - Otherwise prefer the short prefix (first segment of UUID) unless that is already
      used by a different mission, in which case return the full id.
    """
    existing = missions_meta.get(mission_id, {})
    if isinstance(existing, dict):
        folder = existing.get("folder")
        if folder:
            return folder

    short = mission_id.split("-")[0]

    for mid, entry in missions_meta.items():
        if mid == "_aliases":
            continue
        if mid != mission_id and isinstance(entry, dict) and entry.get("folder") == short:
            return mission_id

    return short

def resolve_mission(
    mission: str,
    P: Mapping[str, str | Path],
    *,
    mission_name: str | None = None,
    allow_new: bool = False,
) -> MissionPaths:
    """
    Resolve alias/UUID to canonical paths + display name, using config/missions.json.
    If allow_new is True, fall back to a provisional MissionPaths for missions that
    are not yet recorded in missions.json (useful for first-time downloads).
    """
    missions_json = missions_json_path(P)
    try:
        meta, aliases = _load_missions(missions_json)
    except FileNotFoundError:
        if not allow_new:
            raise
        meta, aliases = {}, {}
    except json.JSONDecodeError:
        if not allow_new:
            raise
        meta, aliases = {}, {}

    mission_id = aliases.get(mission, mission)
    entry = meta.get(mission_id)
    if not entry:
        if not allow_new:
            raise MissionNotFound(f"Mission '{mission}' not found in {missions_json}")
        folder = pick_folder_name(mission_id, meta)
        display = mission_name or (mission if mission != mission_id else folder)
        tables = Path(P["TABLES"]) / folder
        synced = Path(P["SYNCED"]) / folder
        raw    = Path(P["RAW"])    / folder
        maps   = Path(P["MAPS"])   / folder
        return MissionPaths(tables=tables, synced=synced, raw=raw,
                            mission_id=mission_id, folder=folder, display=display, maps=maps)

    folder = entry["folder"]

    # pick a nice display name
    if mission in aliases and aliases[mission] == mission_id:
        display = mission
    else:
        rev = next((a for a, mid in aliases.items() if mid == mission_id), None)
        display = rev or entry.get("name") or folder

    tables = Path(P["TABLES"]) / folder
    synced = Path(P["SYNCED"]) / folder
    raw    = Path(P["RAW"])    / folder
    maps   = Path(P["MAPS"])   / folder

    return MissionPaths(tables=tables, synced=synced, raw=raw,
                        mission_id=mission_id, folder=folder, display=display, maps=maps)


def missions_json_path(paths: Mapping[str, str | Path]) -> Path:
    """Return missions.json path from the paths mapping (defaults to config/missions.json)."""
    repo_root = Path(paths["REPO_ROOT"])
    return Path(paths.get("MISSIONS_JSON", repo_root / "config" / "missions.json"))


def load_missions_file(missions_json: Path) -> dict[str, dict]:
    """Load missions.json safely; return {} if missing or unparsable."""
    if not missions_json.exists():
        return {}
    try:
        return json.loads(missions_json.read_text())
    except Exception:
        return {}


def write_missions_file(missions_json: Path, data: Mapping) -> None:
    """Atomic write of missions.json + cache clear for resolve_mission()."""
    missions_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = missions_json.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(missions_json)
    _load_missions.cache_clear()


def upsert_mission_entry(
    missions_json: Path,
    mission_id: str,
    folder: str,
    mission_name: str | None,
) -> None:
    """Update missions.json with folder + alias, preserving existing entries."""
    cur = load_missions_file(missions_json)

    entry = cur.get(mission_id, {})
    entry.update({"folder": folder, "name": mission_name})
    cur[mission_id] = entry

    if mission_name:
        aliases = cur.get("_aliases", {})
        aliases[mission_name] = mission_id
        cur["_aliases"] = aliases

    write_missions_file(missions_json, cur)
