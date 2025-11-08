"""
Utilities for handling ROS time stamps. 
Use header stamp for correct synchronization of messages
"""

from __future__ import annotations
import logging

_log = logging.getLogger(__name__)
_WARNED_ON_FALLBACK: set[str] = set()

def _to_ns(stamp) -> int | None:
    """Best-effort convert ROS time-like object to ns."""
    if stamp is None:
        return None
    # ROS 1: .secs, .nsecs
    if hasattr(stamp, "secs") and hasattr(stamp, "nsecs"):
        try:
            return int(stamp.secs) * 1_000_000_000 + int(stamp.nsecs)
        except Exception:
            return None
    # ROS 2: .sec, .nanosec
    if hasattr(stamp, "sec") and hasattr(stamp, "nanosec"):
        try:
            return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        except Exception:
            return None
    # Rare variants: .sec + .nsec
    if hasattr(stamp, "sec") and hasattr(stamp, "nsec"):
        try:
            return int(stamp.sec) * 1_000_000_000 + int(stamp.nsec)
        except Exception:
            return None
    return None

def header_stamp_ns(msg) -> int | None:
    """Return header.stamp in ns if present and > 0, else None."""
    hdr = getattr(msg, "header", None)
    if hdr is None:
        return None
    ns = _to_ns(getattr(hdr, "stamp", None))
    if ns and ns > 0:
        return ns
    return None

def message_time_ns(
    msg,
    log_ts: int | None,
    *,
    stream: str = "stream",
    warn_if_fallback: bool = True,
    logger: logging.Logger | None = None,
) -> int:
    """
    Prefer header.stamp (publisher time). Fall back to bag log time (record time).
    Prints a clear warning exactly once per stream on fallback.
    """
    ns = header_stamp_ns(msg)
    if ns is not None:
        return ns
    if warn_if_fallback and stream not in _WARNED_ON_FALLBACK:
        (logger or _log).warning(
            "[time] %s: header.stamp missing/zero — falling back to bag log time. "
            "This adds transport/IO delay; fix upstream if possible.",
            stream,
        )
        _WARNED_ON_FALLBACK.add(stream)
    if log_ts is None:
        raise RuntimeError(f"{stream}: no header.stamp and no log_ts provided")
    return int(log_ts)
