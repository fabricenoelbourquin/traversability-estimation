#!/usr/bin/env python3
"""
Extract key signals from a mission's rosbags -> Parquet tables.

Examples:
  python src/extract_bag.py --mission HEAP-1
  python src/extract_bag.py --mission-id e97e35ad-dd7b-49c4-a158-95aba246520e
  # Overwrite existing Parquet:
  python src/extract_bag.py --mission HEAP-1 --overwrite

Requires explicit topic names in config/topics.yaml (cmd_vel, odom, gps, imu, anymal_state).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
import fnmatch
from itertools import chain

import numpy as np
import pandas as pd
import yaml

from utils.paths import get_paths
from utils.cli import add_mission_arguments, resolve_mission_from_args
from utils.ros_time import header_stamp_ns, message_time_ns
from utils.rosbag_tools import filter_valid_rosbags


from rosbags.highlevel import AnyReader

# Topics we expect to extract; set explicitly in config/topics.yaml
REQUIRED_TOPICS = ("cmd_vel", "odom", "gps", "imu", "anymal_state")
SUPPORTED_BAG_SUFFIXES = (".bag", ".mcap")

def _norm_list(val) -> list[str]:
    if val is None:
        return []
    if isinstance(val, str):
        return [val]
    if isinstance(val, (list, tuple)):
        return [str(x) for x in val if x]
    return [str(val)]

# --------------------- helpers ---------------------

def load_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text()) if p.exists() else {}

def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def load_missions_json(rr: Path) -> dict:
    mj = rr / "config" / "missions.json"
    return json.loads(mj.read_text()) if mj.exists() else {}

def _available_topics_summary(conns, limit: int = 30) -> str:
    """Pretty-print the topics present in the opened bags (for error messages)."""
    pairs = sorted({(c.topic, c.msgtype) for c in conns})
    lines = [f"  - {t} ({mt})" for t, mt in pairs[:limit]]
    if len(pairs) > limit:
        lines.append(f"  ... {len(pairs) - limit} more")
    return "\n".join(lines) if lines else "  (no topics found)"

def _expand_bag_patterns(patterns: list[str]) -> list[str]:
    """Add .mcap variants for any .bag pattern and allow numbered splits."""
    expanded: list[str] = []
    for pat in patterns:
        expanded.append(pat)
        if pat.endswith(".bag") and not pat.endswith("*.bag"):
            expanded.append(pat[:-4] + "*.bag")
        if ".bag" in pat:
            expanded.append(pat.replace(".bag", ".mcap"))
            if pat.endswith(".bag") and not pat.endswith("*.bag"):
                expanded.append(pat[:-4] + "*.mcap")
    seen: set[str] = set()
    out: list[str] = []
    for pat in expanded:
        if pat not in seen:
            seen.add(pat)
            out.append(pat)
    return out

def _glob_supported_bags(raw_dir: Path, patterns: list[str]) -> list[Path]:
    """Find bag/mcap files matching provided patterns or any supported suffix."""
    bag_paths: list[Path] = []
    for pat in _expand_bag_patterns(patterns):
        bag_paths.extend(sorted(raw_dir.glob(pat)))
    bag_paths = list(dict.fromkeys(bag_paths))
    if bag_paths:
        return bag_paths

    # Fallback: gather any supported suffix in the folder
    for suf in SUPPORTED_BAG_SUFFIXES:
        bag_paths.extend(sorted(raw_dir.glob(f"*{suf}")))
    return list(dict.fromkeys(bag_paths))


def choose_topics(conns, topics_cfg: dict | None) -> dict[str, str]:
    """
    Choose topic names for cmd_vel, odom, gps, imu, anymal_state.
    expects explicit topic names in topics.yaml like:
      topics:
        cmd_vel: /anymal/twist_command
        odom:    /anymal/state_estimator/odometry
        gps:     /cpt7/ie/gnss/navsatfix
        imu:     /anymal/imu
        anymal_state: /anymal/state_estimator/anymal_state
    Returns a dict like {"cmd_vel": "/anymal/twist_command", "odom": "...", ...}
    """
    # Create a quick lookup dictionary of {topic_name: message_type}
    # iterate over this 'available' set later to find matches.
    available = {c.topic: c.msgtype for c in conns}
    # Safely extract the 'topics' dictionary from the topics.yaml, defaulting to empty if None
    explicit = (topics_cfg or {}).get("topics") or {}

    if not explicit:
        raise SystemExit(
            "[error] No topics configured in topics.yaml.\n"
            "Add a `topics` mapping with at least cmd_vel, odom, gps, imu, anymal_state.\n"
            "Available topics in these bags:\n"
            f"{_available_topics_summary(conns)}"
        )

    def _heuristic_pick(key: str) -> str | None:
        # Simple fallbacks when explicit names are absent: pick the first topic
        # containing a key token. Keeps selection deterministic and transparent.
        key_tokens = {
            "cmd_vel": ("twist", "cmd"),
            "odom": ("odom", "state_estimator"),
            "gps": ("navsatfix", "gnss"),
            "imu": ("imu",),
            "anymal_state": ("anymal_state", "state_estimator"),
        }.get(key, ())
        # sort the available topics to keep selection deterministic
        for topic in sorted(available):
            low = topic.lower()
            if any(tok in low for tok in key_tokens):
                return topic
        return None

    chosen: dict[str, str] = {}
    missing: list[str] = []
    for key in REQUIRED_TOPICS:
        # _norm_list ensures we can handle both "topic: /foo" and "topic: [/foo, /bar]"
        cand_list = _norm_list(explicit.get(key))
        selected = None
        tried: list[str] = []

        for cand in cand_list:
            # Iterate through every candidate string provided for this key
            tried.append(cand)
            # STRATEGY A: Wildcard Match
            # Only runs if explicitly uses a wildcard character.
            if "*" in cand or "?" in cand:
                matches = [t for t in available if fnmatch.fnmatch(t, cand)]
                if matches:
                    selected = matches[0]
                    break
            # STRATEGY B: Exact Match
            elif cand in available:
                selected = cand
                break
            # STRATEGY C: Suffix (Fuzzy) Match, handles namespace issues.
            # If asked for "odometry", this accepts "/robot_1/odometry".
            else:
                # Try suffix match to cope with missing leading slash or namespace tweaks
                matches = [t for t in available if t.endswith(cand)]
                if matches:
                    selected = matches[0]
                    break
        # Fallback: heuristic pick if none of the configured candidates matched
        if not selected:
            heuristic = _heuristic_pick(key)
            if heuristic:
                selected = heuristic
                if tried:
                    tried_str = ", ".join(tried)
                    print(
                        f"[warn] {key}: none of the configured topics present ({tried_str}); "
                        f"using heuristic match '{heuristic}'.",
                        file=sys.stderr,
                    )
            elif cand_list:
                print(
                    f"[warn] {key}: configured topic(s) not found ({', '.join(cand_list)}).",
                    file=sys.stderr,
                )
            else:
                missing.append(key)
                continue

        if selected:
            chosen[key] = selected

    if missing:
        print(
            "[warn] topics.yaml missing entries for: " + ", ".join(missing),
            file=sys.stderr,
        )

    return chosen

# --- TF / quaternion helpers ---

def _sanitize_frame(name: str) -> str:
    return name[1:] if name.startswith("/") else name

def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    )

def _quat_conj(q):
    w, x, y, z = q
    return (w, -x, -y, -z)

def _quat_norm(q):
    w, x, y, z = q
    n = float(np.sqrt(w*w + x*x + y*y + z*z))
    if n == 0:
        return q
    return (w/n, x/n, y/n, z/n)

# --------------------- extractors ---------------------

def extract_cmd_vel(reader: AnyReader, topic: str) -> pd.DataFrame:
    rows = []
    conns = [c for c in reader.connections if c.topic == topic]
    for conn, t_log, raw in reader.messages(connections=conns):
        msg = reader.deserialize(raw, conn.msgtype)
        ns = message_time_ns(msg, t_log, stream="cmd_vel")
        twist = getattr(msg, "twist", msg)  # TwistStamped.twist or Twist
        rows.append({
            "stamp_ns": int(ns),
            "t": ns * 1e-9,
            "v_cmd_x": float(twist.linear.x),
            "v_cmd_y": float(getattr(twist.linear, "y", 0.0)),
            "w_cmd_z": float(twist.angular.z),
        })
    return pd.DataFrame(rows).sort_values("stamp_ns").reset_index(drop=True)

def extract_odom(reader: AnyReader, topic: str) -> pd.DataFrame:
    rows = []
    conns = [c for c in reader.connections if c.topic == topic]
    for conn, t_log, raw in reader.messages(connections=conns):
        msg = reader.deserialize(raw, conn.msgtype)  # nav_msgs/Odometry
        ns = message_time_ns(msg, t_log, stream="odom")
        vx = float(msg.twist.twist.linear.x)
        vy = float(getattr(msg.twist.twist.linear, "y", 0.0))
        rows.append({
            "stamp_ns": int(ns),
            "t": ns * 1e-9,
            "vx": vx,
            "vy": vy,
            "speed": float(np.hypot(vx, vy)),
            "x": float(msg.pose.pose.position.x),
            "y": float(msg.pose.pose.position.y),
            "qw": float(msg.pose.pose.orientation.w),
            "qx": float(msg.pose.pose.orientation.x),
            "qy": float(msg.pose.pose.orientation.y),
            "qz": float(msg.pose.pose.orientation.z),
        })
    return pd.DataFrame(rows).sort_values("stamp_ns").reset_index(drop=True)

def extract_gps(reader: AnyReader, topic: str) -> pd.DataFrame:
    rows = []
    conns = [c for c in reader.connections if c.topic == topic]
    for conn, t_log, raw in reader.messages(connections=conns):
        msg = reader.deserialize(raw, conn.msgtype)  # NavSatFix
        ns = message_time_ns(msg, t_log, stream="gps")
        rows.append({
            "stamp_ns": int(ns),
            "t": ns * 1e-9,
            "lat": float(msg.latitude),
            "lon": float(msg.longitude),
            "alt": float(msg.altitude),
            "status": int(getattr(msg.status, "status", 0)),
        })
    return pd.DataFrame(rows).sort_values("stamp_ns").reset_index(drop=True)

def extract_imu(reader: AnyReader, topic: str) -> pd.DataFrame:
    rows = []
    conns = [c for c in reader.connections if c.topic == topic]
    for conn, t_log, raw in reader.messages(connections=conns):
        msg = reader.deserialize(raw, conn.msgtype)  # Imu
        ns = message_time_ns(msg, t_log, stream="imu")
        rows.append({
            "stamp_ns": int(ns),
            "t": ns * 1e-9,
            "ax": float(msg.linear_acceleration.x),
            "ay": float(msg.linear_acceleration.y),
            "az": float(msg.linear_acceleration.z),
            "wx": float(msg.angular_velocity.x),
            "wy": float(msg.angular_velocity.y),
            "wz": float(msg.angular_velocity.z),
            "qw": float(msg.orientation.w),
            "qx": float(msg.orientation.x),
            "qy": float(msg.orientation.y),
            "qz": float(msg.orientation.z),
        })
    return pd.DataFrame(rows).sort_values("stamp_ns").reset_index(drop=True)

def extract_base_orientation_from_tf(
    reader: AnyReader,
    world_frame: str,
    base_frame: str,
    *,
    world_fallbacks: list[str] | None = None,
) -> tuple[pd.DataFrame, str | None]:
    """
    Build q_WB (base→world) from /tf and /tf_static.
    Works whether hops come as a->b or only b->a (uses inverse as needed).
    (uses inverse)
    Emits a sparse time-series (records when q_WB changes).
    Time columns use transform header.stamp when available (fallback: bag log time).
    """
    tf_conns = []
    for conn in reader.connections:
        msg_type = (conn.msgtype or "").lower()
        if conn.topic in ("/tf", "/tf_static") and "tfmessage" in msg_type:
            tf_conns.append(conn)

    entries = []
    for conn in tf_conns:
        is_static = conn.topic == "/tf_static"
        for _, t_log, raw in reader.messages(connections=[conn]):
            msg = reader.deserialize(raw, conn.msgtype)
            transforms = getattr(msg, "transforms", ()) or ()
            for transform in transforms:
                parent = _sanitize_frame(getattr(getattr(transform, "header", None), "frame_id", "") or "")
                child = _sanitize_frame(getattr(transform, "child_frame_id", "") or "")
                if not parent or not child:
                    continue
                rot = getattr(getattr(transform, "transform", None), "rotation", None)
                if rot is None:
                    continue
                q = (float(rot.w), float(rot.x), float(rot.y), float(rot.z))
                try:
                    stamp_ns = header_stamp_ns(transform)
                except Exception:
                    stamp_ns = None
                ns = int(stamp_ns) if (stamp_ns and stamp_ns > 0) else int(t_log)
                entries.append((is_static, ns * 1e-9, parent, child, q))

    if not entries:
        return pd.DataFrame(columns=["stamp_ns", "t", "qw", "qx", "qy", "qz"]), None

    entries.sort(key=lambda entry: (0 if entry[0] else 1, entry[1]))

    latest = {}
    adjacency = {}

    def add_edge(parent: str, child: str, quat) -> None:
        # Track the most recent transform and maintain an undirected connectivity graph
        latest[(parent, child)] = quat
        adjacency.setdefault(parent, set()).add(child)
        adjacency.setdefault(child, set()).add(parent)

    def compose(world: str, base: str):
        # Find a path world->base in the graph and compose the quaternions along it
        if world not in adjacency or base not in adjacency:
            return None
        queue = deque([world])
        prev = {world: None}
        while queue:
            node = queue.popleft()
            if node == base:
                break
            for neighbor in adjacency.get(node, ()):
                if neighbor not in prev:
                    prev[neighbor] = node
                    queue.append(neighbor)
        if base not in prev:
            return None
        path = []
        node = base
        while node is not None:
            path.append(node)
            node = prev[node]
        path.reverse()
        q_total = (1.0, 0.0, 0.0, 0.0)
        for frm, to in zip(path[:-1], path[1:]):
            hop = latest.get((frm, to))
            if hop is None:
                reverse = latest.get((to, frm))
                if reverse is None:
                    return None
                hop = _quat_conj(reverse)
            q_total = _quat_mul(q_total, hop)
        q_total = _quat_norm(q_total)
        if q_total[0] < 0.0:
            q_total = tuple(-c for c in q_total)
        return q_total

    rows = []
    last_q = None
    wf = _sanitize_frame(world_frame)
    bf = _sanitize_frame(base_frame)
    candidates: list[str] = []
    for cand in chain([wf], (world_fallbacks or []), ["map", "odom", "enu_origin"]):
        cand_norm = _sanitize_frame(cand)
        if cand_norm and cand_norm not in candidates:
            candidates.append(cand_norm)
    used_world = None

    for is_static, t_val, parent, child, quat in entries:
        add_edge(parent, child, quat)
        q_wb = None
        for cand in candidates:
            q_wb = compose(cand, bf)
            if q_wb is not None:
                used_world = cand
                break
        if q_wb is None:
            continue
        if is_static and rows:
            continue
        if last_q is not None:
            same = (
                np.allclose(q_wb, last_q, atol=1e-6)
                or np.allclose(q_wb, tuple(-c for c in last_q), atol=1e-6)
            )
            if same:
                continue
        last_q = q_wb
        rows.append({
            "stamp_ns": int(round(t_val * 1e9)),
            "t": t_val,
            "qw": q_wb[0],
            "qx": q_wb[1],
            "qy": q_wb[2],
            "qz": q_wb[3],
        })

    return pd.DataFrame(rows).sort_values("stamp_ns").reset_index(drop=True), used_world

# --- ANYmalState extractor: q_*, qd_*, tau_* from msg.joints ---
def extract_anymal_state(reader: AnyReader, topic: str) -> pd.DataFrame:
    """
    anymal_msgs/AnymalState on /anymal/state_estimator/anymal_state

    Uses:
      - msg.joints.name        -> list[str]
      - msg.joints.position    -> list/ndarray (rad)
      - msg.joints.velocity    -> list/ndarray (rad/s)
      - msg.joints.effort      -> list/ndarray (Nm)

    Writes wide columns:
      - q_<name>, qd_<name>, tau_<name>
    """
    tall = []
    conns = [c for c in reader.connections if c.topic == topic]

    for conn, t_log, raw in reader.messages(connections=conns):
        try:
            msg = reader.deserialize(raw, conn.msgtype)
        except Exception:
            continue

        joints = getattr(msg, "joints", None)
        if joints is None:
            continue

        names_raw = getattr(joints, "name", None)
        names = list(names_raw) if names_raw is not None else []
        if len(names) == 0:
            continue

        # Normalize possibly numpy-based sequences to plain lists; avoid truthiness on arrays.
        def _as_list(seq) -> list:
            if seq is None:
                return []
            try:
                return list(seq)
            except Exception:
                return []

        positions = _as_list(getattr(joints, "position", None))
        velocities = _as_list(getattr(joints, "velocity", None))
        efforts = _as_list(getattr(joints, "effort", None))

        ns = header_stamp_ns(getattr(msg, "header", None)) or int(t_log)
        t = float(ns) * 1e-9

        for idx, joint_name in enumerate(names):
            row = {"t": t, "joint": str(joint_name)}
            try:
                if idx < len(positions):
                    row["q"] = float(positions[idx])
            except (TypeError, ValueError):
                pass
            try:
                if idx < len(velocities):
                    row["qd"] = float(velocities[idx])
            except (TypeError, ValueError):
                pass
            try:
                if idx < len(efforts):
                    row["tau"] = float(efforts[idx])
            except (TypeError, ValueError):
                pass

            # Only append rows that contain at least one numeric measurement
            if len(row) > 2:
                tall.append(row)

    if not tall:
        return pd.DataFrame(columns=["t"])

    df = pd.DataFrame(tall)

    wide_frames = []
    def _pivot(val_col: str, prefix: str) -> pd.DataFrame:
        # Pivot tall joint rows into wide form (one column per joint) for the given value
        sub = df.pivot_table(index="t", columns="joint", values=val_col, aggfunc="last")
        sub.columns = [f"{prefix}{c}" for c in sub.columns]
        return sub

    for col, pref in (("q", "q_"), ("qd", "qd_"), ("tau", "tau_")):
        if col in df and df[col].notna().any():
            sub = _pivot(col, pref)
            if not sub.empty:
                wide_frames.append(sub)

    if not wide_frames:
        return pd.DataFrame(columns=["t"])

    wide = pd.concat(wide_frames, axis=1).sort_index().reset_index()
    wide = wide[wide["t"].notna()].drop_duplicates("t", keep="last").reset_index(drop=True)
    return wide

# --------------------- main ---------------------

def main():
    ap = argparse.ArgumentParser(description="Extract key topics from mission rosbags to Parquet.")
    add_mission_arguments(ap)
    ap.add_argument("--topics-cfg", default=str(repo_root() / "config" / "topics.yaml"),
                    help="Topics mapping YAML (explicit topic names).")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing Parquet files.")
    args = ap.parse_args()

    P = get_paths()
    rr = P["REPO_ROOT"]
    missions = load_missions_json(rr)
    mp = resolve_mission_from_args(args, P)
    mission_id, mission_folder, raw_dir, out_dir = mp.mission_id, mp.folder, mp.raw, mp.tables
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine which bag files to read.
    # Prefer rosbags.yaml defaults; if missing, just use all supported bag formats in the mission folder.
    rosbags_yaml = load_yaml(rr / "config" / "rosbags.yaml")
    patterns = rosbags_yaml.get("defaults", ["*.bag"])
    bag_paths = _glob_supported_bags(raw_dir, patterns)
    if not bag_paths:
        raise SystemExit(f"No bags or mcap files found in {raw_dir}")
    
    bag_paths = filter_valid_rosbags(bag_paths)
    if not bag_paths:
        raise SystemExit(f"No valid ROS bags/MCAP files found in {raw_dir} after filtering.")
    print("[info] Bag paths (filtered)")
    for p in bag_paths:
        print(" -", p)

    # Open all selected bags as a single logical dataset
    with AnyReader(bag_paths) as reader:
        conns = list(reader.connections)
        topics_cfg = load_yaml(Path(args.topics_cfg))
        topics = choose_topics(conns, topics_cfg)

        # Informative print
        print("[info] Topics selected")
        keys = ("cmd_vel", "odom", "gps", "imu", "anymal_state")
        pad = max(len(k) for k in keys)  # keep columns aligned even for longer names
        for k in keys:
            if k in topics:
                print(f"{k:>{pad}}: {topics[k]}")
            else:
                print(f"{k:>{pad}}: (not found)")
        
        # Frames from topics.yaml (optional; falls back to defaults)
        frames_cfg = (topics_cfg or {}).get("frames", {})
        world_frame = frames_cfg.get("world", "enu_origin")
        base_frame  = frames_cfg.get("base",  "base")
        print(f"[info] Frames: world='{world_frame}'  base='{base_frame}'")

        # Extract & save
        outputs = {}
        def write_table(name: str, df: pd.DataFrame, filename: str, *, extra=None, empty_msg: str | None = None) -> None:
            if df.empty:
                if empty_msg:
                    print(empty_msg)
                return
            path = out_dir / filename
            if args.overwrite or not path.exists():
                df.to_parquet(path, index=False)
            outputs[name] = {"rows": int(len(df)), "path": str(path), **(extra or {})}

        odom_df = None
        if topics.get("cmd_vel"):
            write_table("cmd_vel", extract_cmd_vel(reader, topics["cmd_vel"]), "cmd_vel.parquet")
        if topics.get("odom"):
            odom_df = extract_odom(reader, topics["odom"])
            write_table("odom", odom_df, "odom.parquet")
        if topics.get("gps"):
            write_table("gps", extract_gps(reader, topics["gps"]), "gps.parquet")
        if topics.get("imu"):
            write_table("imu", extract_imu(reader, topics["imu"]), "imu.parquet")
        # --- Extract base orientation q_WB (base→world) from TF and save as a table
        used_tf_world = None
        try:
            df_q, used_tf_world = extract_base_orientation_from_tf(
                reader,
                world_frame,
                base_frame,
                world_fallbacks=(frames_cfg or {}).get("world_fallbacks"),
            )
        except Exception as e:
            df_q = pd.DataFrame(columns=["stamp_ns","t","qw","qx","qy","qz"])
            print(f"[warn] TF orientation extraction failed: {e}")

        if used_tf_world and used_tf_world != world_frame:
            print(f"[warn] TF world frame fallback: requested '{world_frame}' but using '{used_tf_world}'.")
        if not used_tf_world and not df_q.empty:
            used_tf_world = world_frame

        # If TF is missing but odom is present, fall back to odom orientation
        if df_q.empty and odom_df is not None and not odom_df.empty:
            df_q = odom_df[["stamp_ns", "t", "qw", "qx", "qy", "qz"]].copy()
            used_tf_world = "odom"
            print("[warn] No TF-based orientation recovered; falling back to odom quaternion (world='odom').")

        write_table(
            "base_orientation",
            df_q,
            "base_orientation.parquet",
            extra={"world_frame": used_tf_world or world_frame, "base_frame": base_frame},
            empty_msg="[warn] No TF-based orientation recovered (q_WB).",
        )

        topic_state = topics.get("anymal_state")
        if topic_state:
            try:
                df_js = extract_anymal_state(reader, topic_state)
                write_table(
                    "joint_states",
                    df_js,
                    "joint_states.parquet",
                    empty_msg=f"[info] No joint positions/velocities/torques extracted from {topic_state}.",
                )
            except Exception as e:
                print(f"[warn] anymal_state extractor failed: {e}")

    # Save per-mission metadata
    topics_used = {
        "mission_id": mission_id,
        "mission_folder": mission_folder,
        "bags_used": [p.name for p in bag_paths],
        "topics": topics,
        "outputs": outputs,
    }
    save_json(out_dir / "topics_used.json", topics_used)

    manifest = {
        "mission_id": mission_id,
        "mission_name": missions.get(mission_id, {}).get("name"),
        "created_at": int(time.time()),
        "git_commit": os.popen("git rev-parse --short HEAD 2>/dev/null").read().strip(),
    }
    save_json(out_dir / "manifest.json", manifest)

    print("\n[done] Extracted tables:")
    for k, v in outputs.items():
        print(f"  - {k}: {v['rows']} rows -> {v['path']}")

if __name__ == "__main__":
    main()
