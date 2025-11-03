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

from rosbags.highlevel import AnyReader

# --------------------- helpers ---------------------

def load_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text()) if p.exists() else {}

def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)

def t_sec(nanos: int) -> float:
    return nanos * 1e-9

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

def filter_valid_rosbags(paths: list[Path]) -> list[Path]:
    """Keep only real ROS1 .bag files; drop macOS '._' and anything non-ROS1."""
    keep = []
    for p in paths:
        if not p.is_file():
            continue
        if p.name.startswith("._"):
            # macOS AppleDouble sidecar (resource fork) -> skip
            continue
        try:
            with p.open('rb') as f:
                head = f.read(13)
            if head.startswith(b"#ROSBAG V2.0"):
                keep.append(p)
        except Exception:
            # unreadable -> skip
            pass
    return keep

# --------------------- extractors ---------------------

def extract_cmd_vel(reader: AnyReader, topic: str) -> pd.DataFrame:
    rows = []
    for conn, t, raw in reader.messages(connections=[c for c in reader.connections if c.topic == topic]):
        msg = reader.deserialize(raw, conn.msgtype)
        # Support Twist and TwistStamped-ish payloads
        twist = getattr(msg, "twist", msg)  # TwistStamped.twist or Twist
        rows.append({
            "t": t_sec(t),
            "v_cmd_x": float(twist.linear.x),
            "v_cmd_y": float(twist.linear.y) if hasattr(twist.linear, "y") else 0.0,
            "w_cmd_z": float(twist.angular.z),
        })
    return pd.DataFrame(rows).sort_values("t")

def extract_odom(reader: AnyReader, topic: str) -> pd.DataFrame:
    rows = []
    for conn, t, raw in reader.messages(connections=[c for c in reader.connections if c.topic == topic]):
        msg = reader.deserialize(raw, conn.msgtype)  # nav_msgs/Odometry
        vx = float(msg.twist.twist.linear.x)
        vy = float(getattr(msg.twist.twist.linear, "y", 0.0))
        rows.append({
            "t": t_sec(t),
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
    return pd.DataFrame(rows).sort_values("t")

def extract_gps(reader: AnyReader, topic: str) -> pd.DataFrame:
    rows = []
    for conn, t, raw in reader.messages(connections=[c for c in reader.connections if c.topic == topic]):
        msg = reader.deserialize(raw, conn.msgtype)  # NavSatFix
        rows.append({
            "t": t_sec(t),
            "lat": float(msg.latitude),
            "lon": float(msg.longitude),
            "alt": float(msg.altitude),
            "status": int(getattr(msg.status, "status", 0)),
        })
    return pd.DataFrame(rows).sort_values("t")

def extract_imu(reader: AnyReader, topic: str) -> pd.DataFrame:
    rows = []
    for conn, t, raw in reader.messages(connections=[c for c in reader.connections if c.topic == topic]):
        msg = reader.deserialize(raw, conn.msgtype)  # Imu
        rows.append({
            "t": t_sec(t),
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
    return pd.DataFrame(rows).sort_values("t")

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
    mission_id, mission_folder = mp.mission_id, mp.folder
    raw_dir = P["RAW"] / mission_folder
    out_dir = P["TABLES"] / mission_folder
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
