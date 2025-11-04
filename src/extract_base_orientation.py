#!/usr/bin/env python3
"""
Extract the robot base orientation q_WB (base→world) from TF streams in mission rosbags.

Conventions
-----------
- q_WB maps base-frame vectors into the world frame (active rotation):
    ^W v = q_WB ⊗ ^B v ⊗ q_WB*
- Default frames: world='enu_origin', base='base'  (both overridable via CLI)
- We also record a 'world_axes' flag (enu|neu) in the JSON as metadata so downstream
  tools know whether to project yaw as atan2(N, E) [ENU] or atan2(E, N) [NEU].
  (This does NOT affect the TF math.)

Assumption (based on your inspection)
-------------------------------------
All TF edges along the desired chain world→…→base are present **only in the reverse
direction** in the bag files (i.e., we observe (b→a) when we need (a→b)). Therefore:
- We build the world→base path via undirected adjacency,
- For each hop a→b on the path, we REQUIRE that the observed edge is (b→a),
- We use the quaternion inverse (conjugate) at each hop.

This keeps the extractor simple and easy to reason about.

Outputs
-------
- tables/<mission>/base_orientation.parquet   (columns: t, qw, qx, qy, qz)  # q_WB
- tables/<mission>/tf_orientation.json         (metadata, including 'world_axes')

Usage
-----
  python src/extract_base_orientation.py --mission TRIM-1
  python src/extract_base_orientation.py --mission-id ... --base-frame anymal/base
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import argparse
import json
import math
import os
import sys
import time
from collections import deque, defaultdict

import numpy as np
import pandas as pd

from rosbags.highlevel import AnyReader

from utils.paths import get_paths
from utils.missions import resolve_mission

from extract_bag import filter_valid_rosbags, save_json  # reuse helpers


# --------------------- dataclasses ---------------------

@dataclass
class TransformSample:
    """Minimal TF sample container (we only need quaternions)."""
    time: float
    parent: str
    child: str
    quat: Tuple[float, float, float, float]  # (w, x, y, z)
    is_static: bool

# --- quick-test helpers (no side effects unless --quick-test is used) ---

def rot_vec_q(q: tuple[float,float,float,float], v: tuple[float,float,float]) -> tuple[float,float,float]:
    """Active rotation: v_W = q ⊗ v_B ⊗ q*  (q = w,x,y,z)."""
    w,x,y,z = q
    vx,vy,vz = v
    qv = np.array([x,y,z], dtype=float)
    v3 = np.array([vx,vy,vz], dtype=float)
    t  = 2.0 * np.cross(qv, v3)
    v2 = v3 + w * t + np.cross(qv, t)
    return (float(v2[0]), float(v2[1]), float(v2[2]))

def q_from_yaw(rad: float) -> tuple[float,float,float,float]:
    """Pure yaw about +z."""
    return (math.cos(rad/2.0), 0.0, 0.0, math.sin(rad/2.0))

def yaw_enu(v: tuple[float,float,float]) -> float:
    """Yaw with ENU convention: atan2(N, E) == atan2(y, x)."""
    return math.atan2(v[1], v[0])

def yaw_neu(v: tuple[float,float,float]) -> float:
    """Yaw if world were NEU (x=N, y=E): atan2(E, N) == swap."""
    return math.atan2(v[0], v[1])

def deg(a: float) -> float:
    return math.degrees(a)

def unit(v: tuple[float,float,float]) -> tuple[float,float,float]:
    n = math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]) or 1.0
    return (v[0]/n, v[1]/n, v[2]/n)

# --------------------- small helpers ---------------------

def stamp_to_seconds(stamp) -> float:
    """Convert ROS time (ROS1 or ROS2) into seconds as float."""
    sec = getattr(stamp, "sec", None)
    nsec = getattr(stamp, "nanosec", None)
    if sec is None:
        sec = getattr(stamp, "secs", None)
    if nsec is None:
        nsec = getattr(stamp, "nsec", None)
    if sec is None:
        raise ValueError("Cannot extract `sec` from ROS time stamp.")
    if nsec is None:
        nsec = 0
    return float(sec) + float(nsec) * 1e-9


def quaternion_from_msg(rot) -> Tuple[float, float, float, float]:
    """Extract quaternion in (w, x, y, z) order from a TF transform.rotation message."""
    x = float(getattr(rot, "x"))
    y = float(getattr(rot, "y"))
    z = float(getattr(rot, "z"))
    w = float(getattr(rot, "w"))
    return (w, x, y, z)


def quat_multiply(q1: Tuple[float, float, float, float],
                  q2: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Hamilton product (q1 * q2) in (w, x, y, z) order (active rotation composition)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return (w, x, y, z)


def quat_normalize(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Normalize a quaternion; raise if zero-length."""
    norm = math.sqrt(sum(comp * comp for comp in q))
    if norm == 0.0:
        raise ValueError("Cannot normalize zero-length quaternion.")
    return tuple(comp / norm for comp in q)  # type: ignore[return-value]


def quat_conjugate(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Conjugate (inverse for unit quaternions)."""
    w, x, y, z = q
    return (w, -x, -y, -z)


def quat_equivalent(q1: Tuple[float, float, float, float],
                    q2: Tuple[float, float, float, float],
                    atol: float = 1e-6) -> bool:
    """Allow q and -q as identical rotations."""
    diff = np.abs(np.array(q1) - np.array(q2))
    diff_neg = np.abs(np.array(q1) + np.array(q2))
    return bool((diff <= atol).all() or (diff_neg <= atol).all())


def sanitize_frame_name(frame: str) -> str:
    """Drop a leading slash from frame names for consistency."""
    return frame[1:] if frame.startswith("/") else frame


def load_tf_entries(reader: AnyReader) -> Tuple[List[TransformSample], set[str], set[str]]:
    """
    Stream TF messages from /tf and /tf_static into light samples.
    Returns:
      entries: list of TransformSample
      parent_frames: set of seen parent frame names
      child_frames:  set of seen child frame names
    """
    entries: List[TransformSample] = []
    parent_frames: set[str] = set()
    child_frames: set[str] = set()

    tf_conns = [
        c for c in reader.connections
        if (
            ("TFMessage" in c.msgtype and c.topic in ("/tf", "/tf_static"))
            or ("tfMessage" in c.msgtype and c.topic in ("/tf", "/tf_static"))
        )
    ]
    if not tf_conns:
        return entries, parent_frames, child_frames

    for conn in tf_conns:
        topic = conn.topic
        is_static = "tf_static" in topic
        for _, _, raw in reader.messages(connections=[conn]):
            msg = reader.deserialize(raw, conn.msgtype)
            transforms = getattr(msg, "transforms", None)
            if not transforms:
                continue
            for tr in transforms:
                header = getattr(tr, "header", None)
                transform = getattr(tr, "transform", None)
                if header is None or transform is None:
                    continue
                parent = sanitize_frame_name(getattr(header, "frame_id", ""))
                child = sanitize_frame_name(getattr(tr, "child_frame_id", ""))
                if not parent or not child:
                    continue
                try:
                    t = stamp_to_seconds(header.stamp)
                except Exception:
                    t = 0.0
                quat = quaternion_from_msg(transform.rotation)

                entries.append(TransformSample(time=t, parent=parent, child=child,
                                               quat=quat, is_static=is_static))
                parent_frames.add(parent)
                child_frames.add(child)

    # Ensure static transforms initialize first for deterministic composition.
    entries.sort(key=lambda e: (0 if e.is_static else 1, e.time))
    return entries, parent_frames, child_frames


# ---------- TF composition -> q_WB (base→world), assuming reverse-only edges ----------

def compose_q_WB_reverse_only(
    latest: Dict[Tuple[str, str], Tuple[float, float, float, float]],
    world_frame: str,
    base_frame: str,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Compose q_WB (base→world) along the path world→...→base, under the assumption that
    the ONLY available stored edges are in the reverse direction for each hop.

    For a hop a→b (desired), we REQUIRE that latest contains (b, a); we then use
    q_ab = (q_ba)^*  (the conjugate) to compose.

    Returns the composed, normalized, sign-canonicalized quaternion, or None if no path yet.
    """
    # Build undirected adjacency from the directed reverse-only edges we observed.
    nbrs = defaultdict(set)
    for (p, c) in latest.keys():
        nbrs[p].add(c)
        nbrs[c].add(p)

    # BFS to find a path world -> base
    prev = {world_frame: None}
    q = deque([world_frame])
    while q:
        u = q.popleft()
        if u == base_frame:
            break
        for v in nbrs.get(u, ()):
            if v not in prev:
                prev[v] = u
                q.append(v)
    if base_frame not in prev:
        return None  # no path found yet

    # Reconstruct path nodes: world -> ... -> base
    path = []
    u = base_frame
    while u is not None:
        path.append(u)
        u = prev[u]
    path.reverse()

    # Compose along path using ONLY inverse of reverse edges.
    q_total = (1.0, 0.0, 0.0, 0.0)  # identity
    for a, b in zip(path[:-1], path[1:]):  # desired hop a->b
        q_ba = latest.get((b, a))
        if q_ba is None:
            # By assumption, we expect reverse edges only; fail fast for clarity.
            raise SystemExit(f"Expected reverse TF edge ({b}→{a}) not found while composing {a}→{b}.")
        q_ab = quat_conjugate(q_ba)       # inverse for unit quaternion
        q_total = quat_multiply(q_total, q_ab)

    q_total = quat_normalize(q_total)
    # Canonicalize the sign (scalar part nonnegative) to avoid sudden sign flips in time series.
    if q_total[0] < 0.0:
        q_total = tuple(-c for c in q_total)
    return q_total  # == q_WB


def extract_q_WB_series(
    entries: Sequence[TransformSample],
    world_frame: str,
    base_frame: str,
) -> pd.DataFrame:
    """
    Stream TF updates and emit a minimally sampled time series of q_WB (base→world).

    Implementation notes:
    - We keep the latest quaternion per directed edge in a dict: latest[(parent, child)] = q_pc.
    - We assume edges we need (a→b along world→base) are **not** directly present; instead we
      REQUIRE the reverse (b→a) and use its inverse each hop.
    - We record:
        * the first sample overall
        * any later sample that changes q_WB (beyond sign ambiguity)
        * at least one sample after all statics are processed
    """
    latest: Dict[Tuple[str, str], Tuple[float, float, float, float]] = {}
    rows: List[Dict[str, float]] = []
    last_quat: Optional[Tuple[float, float, float, float]] = None

    for sample in entries:
        latest[(sample.parent, sample.child)] = sample.quat

        q_WB = compose_q_WB_reverse_only(latest, world_frame, base_frame)
        if q_WB is None:
            continue

        should_record = not sample.is_static or not rows
        if not should_record:
            continue

        if last_quat is not None and quat_equivalent(q_WB, last_quat):
            continue

        last_quat = q_WB
        rows.append({"t": sample.time, "qw": q_WB[0], "qx": q_WB[1], "qy": q_WB[2], "qz": q_WB[3]})

    if not rows:
        raise SystemExit("No orientation samples could be recovered for the requested frames.")

    df = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
    return df


# --------------------- CLI ---------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Extract q_WB (base→world) quaternion from TF streams.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--mission", help="Mission alias (from config/missions.json)")
    g.add_argument("--mission-id", help="Mission UUID (from config/missions.json)")

    ap.add_argument("--world-frame", default="enu_origin",
                    help="World frame name (default: enu_origin).")
    ap.add_argument("--base-frame", default="base",
                    help="Base frame name (default: base).")

    ap.add_argument("--world-axes", choices=["enu", "neu"], default="enu",
                    help="Record world axis convention (metadata only; TF math is unaffected).")

    ap.add_argument("--bag-glob", default="*_tf_minimal.bag",
                    help="Glob to select bags (default: '*_tf_minimal.bag').")

    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing orientation parquet if present.")
    ap.add_argument("--quick-test", action="store_true",
                help="Run synthetic + single-sample orientation sanity checks and exit.")

    args = ap.parse_args()

    # --- Mission paths
    P = get_paths()
    mission_key = args.mission or args.mission_id
    mp = resolve_mission(mission_key, P)
    mission_id, raw_dir, out_dir = mp.mission_id, mp.raw, mp.tables
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Pick TF bag(s): default only *_tf_minimal.bag
    bag_paths: List[Path] = sorted(raw_dir.glob(args.bag_glob))
    if not bag_paths: 
        print(f"[warn] No bags matched '{args.bag_glob}'. Falling back to all .bag in {raw_dir}")
        bag_paths = sorted(raw_dir.glob("*.bag"))
    if not bag_paths:
        raise SystemExit(f"No bags found in {raw_dir}")

    bag_paths = filter_valid_rosbags(bag_paths)
    if not bag_paths:
        raise SystemExit(f"No valid ROS1 bags found in {raw_dir} after filtering.")

    print("[info] Bag paths (filtered)")
    for p in bag_paths:
        print(" -", p)

    # --- Stream TF
    with AnyReader(bag_paths) as reader:
        entries, parent_frames, child_frames = load_tf_entries(reader)

    if not entries:
        raise SystemExit("No TF messages were found in the selected bags.")

    # --- Resolve frames (no candidates; just defaults/CLI override) and existence checks
    world_frame = sanitize_frame_name(args.world_frame)
    base_frame  = sanitize_frame_name(args.base_frame)

    all_frames = set(parent_frames) | set(child_frames)
    if world_frame not in all_frames:
        raise SystemExit(f"World frame '{world_frame}' not found in TF tree.")
    if base_frame not in all_frames:
        raise SystemExit(f"Base frame '{base_frame}' not found in TF tree.")

    print("\n[info] Frames selected (producing q_WB = base→world; reverse-only edges assumed)")
    print(f"{'world':>8}: {world_frame}")
    print(f"{'base':>8}: {base_frame}")

    # --- Compose time series q_WB
    df = extract_q_WB_series(entries, world_frame, base_frame)
    if args.quick_test:
        print("\n[quick-test] Quaternion semantics sanity checks")

        # --- Synthetic sanity: yaw 0° (East) & +90° (North) applied to +x_B
        qE = q_from_yaw(0.0)
        vE = rot_vec_q(qE, (1.0,0.0,0.0))
        print(f"  Synth East:  q=(1,0,0,0)  x_B→W={unit(vE)}  yaw_ENU={deg(yaw_enu(vE)):.1f}° (expect ~0°)")

        qN = q_from_yaw(math.pi/2.0)
        vN = rot_vec_q(qN, (1.0,0.0,0.0))
        print(f"  Synth North: q≈(√2/2,0,0,√2/2)  x_B→W={unit(vN)}  yaw_ENU={deg(yaw_enu(vN)):.1f}° (expect ~+90°)")

        # --- Real sample from extracted series (middle row)
        mid = max(0, len(df)//2 - 1)
        q = (float(df.loc[mid,"qw"]), float(df.loc[mid,"qx"]),
             float(df.loc[mid,"qy"]), float(df.loc[mid,"qz"]))
        qc = (q[0], -q[1], -q[2], -q[3])  # conjugate (for direction sanity)

        vx_w = rot_vec_q(q, (1,0,0))     # +x_B
        vy_w = rot_vec_q(q, (0,1,0))     # +y_B
        vmy_w= rot_vec_q(q, (0,-1,0))    # -y_B

        print("\n  Real sample @t≈{:.3f}s".format(float(df.loc[mid,"t"])))
        print("   Using q_WB (as extracted):")
        print("     +x_B → W:", unit(vx_w), "yaw_ENU={:.1f}°, yaw_NEU={:.1f}°"
              .format(deg(yaw_enu(vx_w)), deg(yaw_neu(vx_w))))
        print("     +y_B → W:", unit(vy_w), "yaw_ENU={:.1f}°, yaw_NEU={:.1f}°"
              .format(deg(yaw_enu(vy_w)), deg(yaw_neu(vy_w))))
        print("     -y_B → W:", unit(vmy_w), "yaw_ENU={:.1f}°, yaw_NEU={:.1f}°"
              .format(deg(yaw_enu(vmy_w)), deg(yaw_neu(vmy_w))))

        vx_w_conj = rot_vec_q(qc, (1,0,0))
        print("\n   If we (incorrectly) used q_BW (= q_WB*):")
        print("     +x_B → W:", unit(vx_w_conj),
              "yaw_ENU={:.1f}°".format(deg(yaw_enu(vx_w_conj))))

        print("\n[quick-test] Notes:")
        print("  • If +x_B yaw_ENU tracks your GPS/path yaw, your base forward is +x (REP-103).")
        print("  • If -y_B yaw_ENU tracks best, your published base forward is -y.")
        print("  • If yaw_NEU is the one that looks right (vs yaw_ENU), your world is NEU,")
        print("    which will appear as an ~±90° bias if you use ENU formulas.")
        sys.exit(0)


    # --- Save parquet
    out_path = out_dir / "base_orientation.parquet"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"{out_path} already exists. Use --overwrite to replace it.")
    df.to_parquet(out_path, index=False)

    # --- Save compact JSON metadata
    outputs = {
        "rows": int(len(df)),
        "path": str(out_path),
        "world_frame": world_frame,
        "base_frame": base_frame,
        "quaternion_semantics": "q_WB (base→world)",
        "edge_direction_assumption": "reverse_only_per_hop",
        "world_axes": args.world_axes,
    }

    # We also store list of frames seen for quick audit
    tf_meta = {
        "available_parents": sorted(parent_frames),
        "available_children": sorted(child_frames),
    }

    save_json(out_dir / "tf_orientation.json", {
        "mission_id": mission_id,
        "created_at": int(time.time()),
        "git_commit": os.popen("git rev-parse --short HEAD 2>/dev/null").read().strip(),
        "bags_used": [p.name for p in bag_paths],
        "frames": tf_meta,
        "outputs": outputs,
    })

    print("\n[done] Extracted q_WB (base→world):")
    print(f"  - rows: {len(df)}")
    print(f"  - save: {out_path}")


if __name__ == "__main__":
    main()
