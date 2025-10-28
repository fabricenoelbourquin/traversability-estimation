# src/utils/paths.py
from pathlib import Path
import os, yaml

def get_paths():
    repo_root = Path(__file__).resolve().parents[2]
    paths_yaml = repo_root / "config" / "paths.yaml"

    # defaults
    data_root = os.getenv("DATA_ROOT", "/Volumes/T7/TRAV_DATA")
    subdirs = {
        "raw": "raw",
        "tables": "tables",
        "synced": "synced",
        "maps": "maps",
        "cache": "cache",
        "logs": "logs",
        "models": "models",  # NEW
    }

    if paths_yaml.exists():
        cfg = yaml.safe_load(paths_yaml.read_text()) or {}
        # allow YAML to set data_root unless env var overrides it
        data_root = os.getenv("DATA_ROOT", cfg.get("data_root", data_root))
        if "subdirs" in cfg and isinstance(cfg["subdirs"], dict):
            subdirs.update({k: v for k, v in cfg["subdirs"].items() if isinstance(v, str) and v})

    root = Path(data_root)

    return {
        "RAW": root / subdirs["raw"],
        "TABLES": root / subdirs["tables"],
        "SYNCED": root / subdirs["synced"],
        "MAPS": root / subdirs["maps"],
        "CACHE": root / subdirs["cache"],
        "LOGS": root / subdirs["logs"],
        "MODELS": root / subdirs["models"],  # ← comma was missing before
        "REPO_ROOT": repo_root,
        "MISSIONS_JSON": repo_root / "config" / "missions.json",
    }
