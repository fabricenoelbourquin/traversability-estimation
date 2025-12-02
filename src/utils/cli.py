"""Shared CLI helpers for mission-aware scripts."""
from __future__ import annotations

import argparse
from typing import Mapping

from utils.missions import MissionPaths, resolve_mission


def add_mission_arguments(ap: argparse.ArgumentParser, *, required: bool = True) -> None:
    """
    Add mutually exclusive --mission / --mission-id arguments.
    By default one of them is required.
    """
    grp = ap.add_mutually_exclusive_group(required=required)
    grp.add_argument("--mission", help="Mission alias or UUID (matches missions.json aliases).")
    grp.add_argument("--mission-id", help="Mission UUID (overrides --mission if both are provided).")


def add_hz_argument(ap: argparse.ArgumentParser, *, default: int | None = None, help_text: str | None = None) -> None:
    """Standard --hz argument used by synced parquet selectors."""
    ap.add_argument(
        "--hz",
        type=int,
        default=default,
        help=help_text or "Select synced_<Hz>Hz[_metrics].parquet (default: latest).",
    )


def resolve_mission_from_args(args, paths: Mapping[str, str]) -> MissionPaths:
    """
    Resolve MissionPaths from parsed args and a paths mapping.
    Prefers --mission-id over --mission if both are present.
    """
    mission_key = getattr(args, "mission_id", None) or getattr(args, "mission", None)
    if not mission_key:
        raise ValueError("Mission argument missing; call add_mission_arguments on the parser.")
    return resolve_mission(mission_key, paths)
