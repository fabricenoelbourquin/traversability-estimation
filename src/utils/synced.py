"""Helpers for picking synced parquet files consistently across scripts."""
from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable


def infer_hz_from_path(p: Path) -> int | None:
    """
    Try to extract the Hz component from filenames like synced_<Hz>Hz[_metrics].parquet.
    Returns None if the pattern is not found.
    """
    m = re.search(r"synced_(\d+)Hz", p.name)
    return int(m.group(1)) if m else None


def _dedupe_by_base(paths: Iterable[Path]) -> list[Path]:
    """De-duplicate by stripping the optional _metrics suffix while preserving order."""
    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        key = p.name.replace("_metrics", "")
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def resolve_synced_parquet(
    sync_dir: Path,
    hz: int | None = None,
    *,
    prefer_metrics: bool = False,
    metrics_only: bool = False,
) -> Path:
    """
    Return a synced parquet path under `sync_dir`, optionally selecting a specific Hz.

    Args:
        sync_dir: Mission's synced directory.
        hz: Desired sampling rate. If None, pick the newest matching file.
        prefer_metrics: Check synced_*_metrics.parquet before plain synced_*.parquet.
        metrics_only: Require *_metrics.parquet; if not present, raise.

    Raises:
        FileNotFoundError if no matching file is found.
    """
    sync_dir = Path(sync_dir)
    if hz is not None:
        hz = int(hz)
        metrics = sync_dir / f"synced_{hz}Hz_metrics.parquet"
        plain = sync_dir / f"synced_{hz}Hz.parquet"

        order = [metrics, plain] if prefer_metrics else [plain, metrics]
        if metrics_only:
            order = [metrics]

        for cand in order:
            if cand.exists():
                return cand
        suffix = "_metrics.parquet" if metrics_only else ".parquet"
        raise FileNotFoundError(f"synced_{hz}Hz{suffix} not found in {sync_dir}")

    metrics_list = sorted(sync_dir.glob("synced_*Hz_metrics.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    plain_list = sorted(sync_dir.glob("synced_*Hz.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)

    ordered = metrics_list + plain_list if prefer_metrics else plain_list + metrics_list
    if metrics_only:
        ordered = metrics_list

    ordered = _dedupe_by_base(ordered)
    if not ordered:
        kind = "metrics" if metrics_only else "synced"
        raise FileNotFoundError(f"No {kind} parquets found in {sync_dir}")

    return ordered[0]
