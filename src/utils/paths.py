# src/utils/paths.py
from pathlib import Path
import os, yaml

def get_paths():
    repo_root = Path(__file__).resolve().parents[2]
    paths_yaml = repo_root / "config" / "paths.yaml"
    data_root = os.getenv("DATA_ROOT", "/Volumes/T7/TRAV_DATA")
    if paths_yaml.exists():
        cfg = yaml.safe_load(paths_yaml.read_text()) or {}
        data_root = os.getenv("DATA_ROOT", cfg.get("data_root", data_root))
    root = Path(data_root)
    return {
        "RAW": root / "raw", "TABLES": root / "tables", "SYNCED": root / "synced",
        "MAPS": root / "maps", "CACHE": root / "cache", "LOGS": root / "logs",
        "REPO_ROOT": repo_root, "MISSIONS_JSON": repo_root / "config" / "missions.json"
    }
