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

def resolve_mission(mission: str, P: Mapping[str, str | Path]) -> MissionPaths:
    """Resolve alias/UUID to canonical paths + display name, using config/missions.json."""
    repo_root = Path(P["REPO_ROOT"])
    missions_json = repo_root / "config" / "missions.json"
    meta, aliases = _load_missions(missions_json)

    mission_id = aliases.get(mission, mission)
    entry = meta.get(mission_id)
    if not entry:
        raise MissionNotFound(f"Mission '{mission}' not found in {missions_json}")

    folder = entry["folder"]

    # pick a nice display name
    if mission in aliases and aliases[mission] == mission_id:
        display = mission
    else:
        rev = next((a for a, mid in aliases.items() if mid == mission_id), None)
        display = rev or folder

    tables = Path(P["TABLES"]) / folder
    synced = Path(P["SYNCED"]) / folder
    raw    = Path(P["RAW"])    / folder
    maps    = Path(P["MAPS"])    / folder

    return MissionPaths(tables=tables, synced=synced, raw=raw,
                        mission_id=mission_id, folder=folder, display=display, maps = maps)


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
