from __future__ import annotations
from pathlib import Path
from typing import Iterable, List
from rosbags.highlevel import AnyReader
from utils.ros_time import message_time_ns

def filter_valid_rosbags(paths: Iterable[Path]) -> list[Path]:
    """
    Filter the given paths, keeping only valid ROS bag files (ROS1 .bag or .mcap).
    Skips files that cannot be opened or do not start with the expected signature.
    Including hidden files (starting with "._") is avoided (MacOS).
    """
    keep: list[Path] = []
    for p in paths:
        if not p.is_file() or p.name.startswith("._"):
            continue
        suffix = p.suffix.lower()
        if suffix == ".bag":
            try:
                with p.open("rb") as f:
                    if f.read(13).startswith(b"#ROSBAG V2.0"):
                        keep.append(p)
            except OSError:
                pass
        elif suffix == ".mcap":
            try:
                with p.open("rb") as f:
                    head = f.read(8)
                    # MCAP files start with 'MCAP' (optionally preceded by 0x89)
                    if head.startswith(b"MCAP") or head[1:5] == b"MCAP":
                        keep.append(p)
            except OSError:
                pass
    return keep

def expand_bag_patterns(pattern: str) -> list[str]:
    """
    Expand a single bag glob to also cover numbered splits and .mcap variants.
    Example: "*_hdr_front.bag" -> ["*_hdr_front.bag", "*_hdr_front.mcap",
                                   "*_hdr_front*.bag", "*_hdr_front*.mcap"]
    """
    out: list[str] = []
    seen: set[str] = set()

    def add(p: str):
        if p not in seen:
            seen.add(p)
            out.append(p)

    add(pattern)

    # Swap extensions to include ROS1/.mcap twins.
    if ".bag" in pattern:
        add(pattern.replace(".bag", ".mcap"))
    if ".mcap" in pattern:
        add(pattern.replace(".mcap", ".bag"))

    # Add trailing-wildcard variants so "*_hdr_front.bag" also catches *_hdr_front_0.*
    for ext, trim in ((".bag", 4), (".mcap", 5)):
        if pattern.endswith(ext):
            base = pattern[:-trim]
            if not base.endswith("*"):
                add(base + "*" + ext)
                # also add counterpart extension with wildcard
                add(base + ("*.mcap" if ext == ".bag" else "*.bag"))

    return out

def iter_msgs_with_time(reader, connections, *, stream="stream", warn_if_fallback=True):
    """
    Iterate over messages from the given reader and connections,
    yielding (connection, timestamp_ns, message) tuples."""
    for conn, ts, raw in reader.messages(connections=connections):
        msg = reader.deserialize(raw, conn.msgtype)
        ns  = message_time_ns(msg, ts, stream=stream, warn_if_fallback=warn_if_fallback)
        yield conn, ns, msg
