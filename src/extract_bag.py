#!/usr/bin/env python3
"""
Extract key signals from a mission's rosbags -> Parquet tables.

Examples:
  python src/extract_bag.py --mission HEAP-1
  python src/extract_bag.py --mission-id e97e35ad-dd7b-49c4-a158-95aba246520e
  # Overwrite existing Parquet:
  python src/extract_bag.py --mission HEAP-1 --overwrite
"""

from __future__ import annotations
from pathlib import Path
import argparse, json, os, re, sys, time

import pandas as pd
import numpy as np
import yaml

from utils.paths import get_paths
from utils.missions import resolve_mission
from utils.ros_time import message_time_ns, header_stamp_ns
from utils.rosbag_tools import filter_valid_rosbags

from rosbags.highlevel import AnyReader

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

def choose_topics(conns, topics_cfg: dict) -> dict[str, str]:
    """
    Choose topic names for cmd_vel, odom, gps, imu.
    Preference:
      - explicit names in topics.yaml (if provided)
      - heuristics by name + msg type
    Returns a dict like {"cmd_vel": "/anymal/twist_command", "odom": "...", ...}
    """
    # explicit config (optional)
    explicit = (topics_cfg or {}).get("topics", {})
    chosen: dict[str, str | None] = {
        "cmd_vel": explicit.get("cmd_vel"),
        "odom": explicit.get("odom"),
        "gps": explicit.get("gps"),
        "imu": explicit.get("imu"),
    }

    # If something is missing, try to guess from connections
    # Build list of (topic, msgtype)
    items = [(c.topic, c.msgtype) for c in conns]

    def pick(pred_name_re: re.Pattern, type_whitelist: tuple[str, ...], prefer_re: re.Pattern | None = None) -> str | None:
        cand = []
        for topic, mtype in items:
            if pred_name_re.search(topic) and mtype in type_whitelist:
                cand.append(topic)
        if not cand:
            # relax: allow name-only OR type-only matches
            for topic, mtype in items:
                if pred_name_re.search(topic) or (mtype in type_whitelist):
                    cand.append(topic)
        if not cand:
            return None
        if prefer_re:
            for t in cand:
                if prefer_re.search(t):
                    return t
        # otherwise first (stable order from AnyReader.connections)
        return cand[0]

    # Regexes + msg types
    RE_CMD = re.compile(r'(^|/)cmd_vel$|command.*(twist|vel)|twist_command', re.I)
    RE_ODOM = re.compile(r'odom|odometry|state_estimator/odometry', re.I)
    RE_GPS = re.compile(r'navsatfix|gps|gnss|fix', re.I)
    RE_IMU = re.compile(r'(^|/)imu($|/)', re.I)

    TYPE_TWIST = ("geometry_msgs/msg/Twist", "geometry_msgs/msg/TwistStamped",
                  "geometry_msgs/Twist", "geometry_msgs/TwistStamped")
    TYPE_ODOM  = ("nav_msgs/msg/Odometry", "nav_msgs/Odometry")
    TYPE_GPS   = ("sensor_msgs/msg/NavSatFix", "sensor_msgs/NavSatFix")
    TYPE_IMU   = ("sensor_msgs/msg/Imu", "sensor_msgs/Imu")

    prefer_anymal = re.compile(r'anymal', re.I)
    prefer_cpt7 = re.compile(r'cpt7|inertial_explorer', re.I)

    if not chosen["cmd_vel"]:
        chosen["cmd_vel"] = pick(RE_CMD, TYPE_TWIST)
    if not chosen["odom"]:
        chosen["odom"] = pick(RE_ODOM, TYPE_ODOM, prefer_anymal)
    if not chosen["gps"]:
        chosen["gps"] = pick(RE_GPS, TYPE_GPS, prefer_cpt7)
    if not chosen["imu"]:
        chosen["imu"] = pick(RE_IMU, TYPE_IMU, re.compile(r'anymal|adis|stim', re.I))

    return {k: v for k, v in chosen.items() if v}

# --- TF / quaternion helpers ---

def _sanitize_frame(name: str) -> str:
    return name[1:] if name.startswith("/") else name

def _quat_mul(q1, q2):
    w1,x1,y1,z1 = q1; w2,x2,y2,z2 = q2
    return (
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    )

def _quat_conj(q):
    w,x,y,z = q
    return (w, -x, -y, -z)

def _quat_norm(q):
    w,x,y,z = q
    n = float(np.sqrt(w*w + x*x + y*y + z*z))
    if n == 0: return q
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
) -> pd.DataFrame:
    """
    Build q_WB (base→world) from /tf and /tf_static.
    Works whether hops come as a->b or only b->a (uses inverse as needed).
    (uses inverse)
    Emits a sparse time-series (records when q_WB changes).
    Time columns use transform header.stamp when available (fallback: bag log time).
    """
    # collect all TF samples
    entries = []
    for conn in reader.connections:
        if ("TFMessage" in conn.msgtype or "tfMessage" in conn.msgtype) and conn.topic in ("/tf", "/tf_static"):
            is_static = (conn.topic == "/tf_static")
            for _, t_log, raw in reader.messages(connections=[conn]):
                msg = reader.deserialize(raw, conn.msgtype)
                transforms = getattr(msg, "transforms", None)
                if not transforms:
                    continue
                for tr in transforms:
                    parent = _sanitize_frame(getattr(getattr(tr, "header", None), "frame_id", "") or "")
                    child  = _sanitize_frame(getattr(tr, "child_frame_id", "") or "")
                    if not parent or not child:
                        continue
                    rot = getattr(getattr(tr, "transform", None), "rotation", None)
                    if rot is None:
                        continue
                    q = (float(rot.w), float(rot.x), float(rot.y), float(rot.z))
                    ns_tr = header_stamp_ns(tr)  # header.stamp on the TransformStamped
                    ns = int(ns_tr) if (ns_tr is not None and ns_tr > 0) else int(t_log)
                    entries.append((bool(is_static), ns * 1e-9, parent, child, q))

    if not entries:
        return pd.DataFrame(columns=["stamp_ns","t","qw","qx","qy","qz"])


    # process statics first, then dynamics by time
    entries.sort(key=lambda e: (0 if e[0] else 1, e[1]))

    latest = {}          # (parent, child) -> quat(parent->child)
    adj = {}             # undirected adjacency for BFS

    def _add_edge(p, c, q):
        latest[(p,c)] = q
        adj.setdefault(p, set()).add(c)
        adj.setdefault(c, set()).add(p)

    def _compose_q(a: str, b: str):
        # BFS on current adj to get path a->...->b
        if a not in adj or b not in adj:
            return None
        prev = {a: None}
        q = [a]
        from collections import deque
        dq = deque(q)
        while dq:
            u = dq.popleft()
            if u == b: break
            for v in adj.get(u, ()):
                if v not in prev:
                    prev[v] = u
                    dq.append(v)
        if b not in prev:
            return None
        # reconstruct path and compose
        path = []
        u = b
        while u is not None:
            path.append(u)
            u = prev[u]
        path.reverse()
        qtot = (1.0, 0.0, 0.0, 0.0)
        for x, y in zip(path[:-1], path[1:]):   # desired hop x->y
            q_xy = latest.get((x,y))
            if q_xy is None:
                q_yx = latest.get((y,x))
                if q_yx is None:
                    return None
                q_xy = _quat_conj(q_yx)
            qtot = _quat_mul(qtot, q_xy)
        qtot = _quat_norm(qtot)
        if qtot[0] < 0.0:  # sign-canonicalize
            qtot = tuple(-c for c in qtot)
        return qtot

    rows = []
    last_q = None
    wf = _sanitize_frame(world_frame)
    bf = _sanitize_frame(base_frame)

    for is_static, t, parent, child, q in entries:
        _add_edge(parent, child, q)
        q_WB = _compose_q(wf, bf)
        if q_WB is None:
            continue
        if is_static and rows:
            continue
        if last_q is not None:
            # skip if identical (including q ~ -q)
            same = (np.allclose(q_WB, last_q, atol=1e-6) or
                    np.allclose(q_WB, tuple(-c for c in last_q), atol=1e-6))
            if same:
                continue
        last_q = q_WB
        rows.append({
            "stamp_ns": int(round(t * 1e9)),
            "t": t,
            "qw": q_WB[0], "qx": q_WB[1], "qy": q_WB[2], "qz": q_WB[3],
        })

    return pd.DataFrame(rows).sort_values("stamp_ns").reset_index(drop=True)

# --------------------- main ---------------------

def main():
    ap = argparse.ArgumentParser(description="Extract key topics from mission rosbags to Parquet.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--mission", help="Mission alias (from config/missions.json)")
    g.add_argument("--mission-id", help="Mission UUID (from config/missions.json)")
    ap.add_argument("--topics-cfg", default=str(repo_root() / "config" / "topics.yaml"),
                    help="Optional topics override YAML (explicit topic names).")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing Parquet files.")
    args = ap.parse_args()

    P = get_paths()
    rr = P["REPO_ROOT"]
    missions = load_missions_json(rr)
    mp = resolve_mission(args.mission, P)
    mission_id, mission_folder, raw_dir, out_dir = mp.mission_id, mp.folder, mp.raw, mp.tables
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine which bag files to read.
    # Prefer rosbags.yaml defaults; if missing, just use all .bag in the mission folder.
    rosbags_yaml = load_yaml(rr / "config" / "rosbags.yaml")
    patterns = rosbags_yaml.get("defaults", ["*.bag"])
    # Expand patterns relative to the mission folder
    bag_paths: list[Path] = []
    for pat in patterns:
        bag_paths.extend(sorted(raw_dir.glob(pat)))
    # De-duplicate while preserving order
    seen = set()
    bag_paths = [p for p in bag_paths if not (p in seen or seen.add(p))]
    if not bag_paths:
        # Fallback to all bags
        bag_paths = sorted(raw_dir.glob("*.bag"))
    if not bag_paths:
        raise SystemExit(f"No bags found in {raw_dir}")

    bag_paths = filter_valid_rosbags(bag_paths)
    if not bag_paths:
        raise SystemExit(f"No valid ROS1 bags found in {raw_dir} after filtering.")
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
        for k in ("cmd_vel", "odom", "gps", "imu"):
            if k in topics:
                print(f"{k:>8}: {topics[k]}")
            else:
                print(f"{k:>8}: (not found)")
        
        # Frames from topics.yaml (optional; falls back to defaults)
        frames_cfg = (topics_cfg or {}).get("frames", {}) if topics_cfg else {}
        world_frame = frames_cfg.get("world", "enu_origin")
        base_frame  = frames_cfg.get("base",  "base")
        print(f"[info] Frames: world='{world_frame}'  base='{base_frame}'")

        # Extract & save
        outputs = {}
        if topics.get("cmd_vel"):
            df = extract_cmd_vel(reader, topics["cmd_vel"])
            path = out_dir / "cmd_vel.parquet"
            if args.overwrite or not path.exists():
                df.to_parquet(path, index=False)
            outputs["cmd_vel"] = {"rows": int(len(df)), "path": str(path)}
        if topics.get("odom"):
            df = extract_odom(reader, topics["odom"])
            path = out_dir / "odom.parquet"
            if args.overwrite or not path.exists():
                df.to_parquet(path, index=False)
            outputs["odom"] = {"rows": int(len(df)), "path": str(path)}
        if topics.get("gps"):
            df = extract_gps(reader, topics["gps"])
            path = out_dir / "gps.parquet"
            if args.overwrite or not path.exists():
                df.to_parquet(path, index=False)
            outputs["gps"] = {"rows": int(len(df)), "path": str(path)}
        if topics.get("imu"):
            df = extract_imu(reader, topics["imu"])
            path = out_dir / "imu.parquet"
            if args.overwrite or not path.exists():
                df.to_parquet(path, index=False)
            outputs["imu"] = {"rows": int(len(df)), "path": str(path)}
        # --- Extract base orientation q_WB (base→world) from TF and save as a table
        try:
            df_q = extract_base_orientation_from_tf(reader, world_frame, base_frame)
        except Exception as e:
            df_q = pd.DataFrame(columns=["t","qw","qx","qy","qz"])
            print(f"[warn] TF orientation extraction failed: {e}")

        if len(df_q):
            path = out_dir / "base_orientation.parquet"
            if args.overwrite or not path.exists():
                df_q.to_parquet(path, index=False)
            outputs["base_orientation"] = {"rows": int(len(df_q)), "path": str(path),
                                           "world_frame": world_frame, "base_frame": base_frame}
        else:
            print("[warn] No TF-based orientation recovered (q_WB).")


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
