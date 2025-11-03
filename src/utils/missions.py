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
