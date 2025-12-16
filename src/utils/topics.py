"""Helpers for resolving topic names with fallbacks from config/topics.yaml."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import yaml

from utils.paths import get_paths


def _normalize_list(val) -> list[str]:
    """Return a flat list of strings from str|list; ignore non-str items."""
    if val is None:
        return []
    if isinstance(val, str):
        return [val]
    if isinstance(val, Iterable):
        return [str(v) for v in val if isinstance(v, str)]
    return []


def load_topic_candidates(key: str, fallback: list[str]) -> list[str]:
    """
    Load an ordered list of candidate topics from config/topics.yaml.
    If the key is missing or the file cannot be read, return the provided fallback list.
    """
    candidates: list[str] = []
    try:
        cfg_path = Path(get_paths()["REPO_ROOT"]) / "config" / "topics.yaml"
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
        candidates = _normalize_list((cfg.get("topics") or {}).get(key))
    except Exception:
        return list(fallback)

    return candidates or list(fallback)
