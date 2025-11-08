from __future__ import annotations
from pathlib import Path
from typing import Iterable, List
from rosbags.highlevel import AnyReader
from utils.ros_time import message_time_ns

def filter_valid_rosbags(paths: Iterable[Path]) -> list[Path]:
    """
    Filter the given paths, keeping only valid ROS bag files (ROS1, version 2).
    Skips files that cannot be opened or do not start with the ROS bag signature.
    Including hidden files (starting with "._") is avoided (MacOS).
    """
    keep: list[Path] = []
    for p in paths:
        if not p.is_file() or p.name.startswith("._"):
            continue
        try:
            with p.open("rb") as f:
                if f.read(13).startswith(b"#ROSBAG V2.0"):
                    keep.append(p)
        except OSError:
            pass
    return keep

def iter_msgs_with_time(reader, connections, *, stream="stream", warn_if_fallback=True):
    """
    Iterate over messages from the given reader and connections,
    yielding (connection, timestamp_ns, message) tuples."""
    for conn, ts, raw in reader.messages(connections=connections):
        msg = reader.deserialize(raw, conn.msgtype)
        ns  = message_time_ns(msg, ts, stream=stream, warn_if_fallback=warn_if_fallback)
        yield conn, ns, msg