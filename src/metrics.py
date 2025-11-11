# src/metrics.py
from __future__ import annotations
from typing import Callable, Iterable
import numpy as np
import pandas as pd

# --- registry ---
REGISTRY: dict[str, Callable[[pd.DataFrame, dict], pd.Series]] = {}

def metric(name: str):
    def wrap(fn: Callable[[pd.DataFrame, dict], pd.Series]):
        REGISTRY[name] = fn
        return fn
    return wrap

def _ensure_speed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create v_cmd and v_actual if missing, using vector norms."""
    out = df.copy()
    if "v_cmd" not in out.columns:
        if {"v_cmd_x","v_cmd_y"}.issubset(out.columns):
            out["v_cmd"] = np.hypot(out["v_cmd_x"], out["v_cmd_y"])
    if "v_actual" not in out.columns:
        if {"vx","vy"}.issubset(out.columns):
            out["v_actual"] = np.hypot(out["vx"], out["vy"])
    return out

# -------- helpers for joint-based metrics ------------------------------------

def _dt_seconds(t: pd.Series) -> pd.Series:
    dt = t.astype(float).diff()
    if dt.isna().all():
        return pd.Series(np.nan, index=t.index)
    return dt.fillna(0.0).clip(lower=0.0)

def _find_joint_columns(df: pd.DataFrame):
    qd_cols  = [c for c in df.columns if c.startswith("qd_")]
    tau_cols = [c for c in df.columns if c.startswith("tau_")]
    return qd_cols, tau_cols


# Metrics
#   Each metric:
#     - takes (df, cfg)
#     - returns a 1D Series (aligned with df.index)
#     - should be robust to missing columns (return NaNs where inputs are missing)
@metric("speed_error_abs")
def speed_error_abs(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """| ||cmd|| - ||actual|| |"""
    d = _ensure_speed_columns(df)
    return (d["v_cmd"] - d["v_actual"]).abs()

@metric("speed_error_signed")
def speed_error_signed(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """( ||cmd|| - ||actual|| ), useful to see bias."""
    d = _ensure_speed_columns(df)
    return d["v_cmd"] - d["v_actual"]

@metric("speed_tracking_score")
def speed_tracking_score(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """
    1.0 = perfect tracking, 0 = poor (clipped by max_speed_mps).
    score = max(0, 1 - |err|/max_speed)
    """
    d = _ensure_speed_columns(df)
    vmax = float(cfg.get("max_speed_mps", 1.0))
    err = (d["v_cmd"] - d["v_actual"]).abs()
    return (1.0 - err / max(vmax, 1e-9)).clip(lower=0.0, upper=1.0)


# ---- Kinematic "effort" proxies ---------------------------------------------

@metric("joint_travel_rate_L1")
def joint_travel_rate_L1(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """
    Sum of absolute joint velocities [rad/s]:  sum_j |qd_j|.
    Useful as a per-sample 'kinematic fussiness' signal.
    """
    qd_cols, _ = _find_joint_columns(df)
    if not qd_cols:
        return pd.Series(np.nan, index=df.index)
    return df[qd_cols].abs().sum(axis=1)

@metric("joint_travel_cum_L1")
def joint_travel_cum_L1(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """
    Cumulative integral of L1 joint speed [rad]: ∫ sum_j |qd_j| dt.
    """
    qd_cols, _ = _find_joint_columns(df)
    if not qd_cols or "t" not in df:
        return pd.Series(np.nan, index=df.index)
    rate = df[qd_cols].abs().sum(axis=1)
    dt = _dt_seconds(df["t"])
    return (rate * dt).fillna(0.0).cumsum()

# ---- Actuation effort & energy ----------------------------------------------

@metric("torque_L1_rate")
def torque_L1_rate(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """
    Sum of absolute joint torques [Nm]: sum_j |tau_j|.
    """
    _, tau_cols = _find_joint_columns(df)
    if not tau_cols:
        return pd.Series(np.nan, index=df.index)
    return df[tau_cols].abs().sum(axis=1)

@metric("power_mech")
def power_mech(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """
    Mechanical power [W]: sum_j tau_j * qd_j (can be negative with regen).
    """
    qd_cols, tau_cols = _find_joint_columns(df)
    if not qd_cols or not tau_cols:
        return pd.Series(np.nan, index=df.index)
    # align columns by joint name
    joints = sorted(set(c[3:] for c in qd_cols) & set(c[4:] for c in tau_cols))
    if not joints:
        return pd.Series(np.nan, index=df.index)
    terms = []
    for j in joints:
        terms.append(df[f"tau_{j}"] * df[f"qd_{j}"])
    return pd.concat(terms, axis=1).sum(axis=1)

@metric("energy_pos_cum")
def energy_pos_cum(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """
    Cumulative positive mechanical energy [J]: ∫ sum_j max(0, tau_j*qd_j) dt.
    """
    if "t" not in df:
        return pd.Series(np.nan, index=df.index)
    P = REGISTRY["power_mech"](df, cfg)
    if P.isna().all():
        return pd.Series(np.nan, index=df.index)
    dt = _dt_seconds(df["t"])
    e_step = np.maximum(P, 0.0) * dt
    return e_step.fillna(0.0).cumsum()

@metric("cot_running")
def cot_running(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """
    Running Cost of Transport (dimensionless):
      COT(t) = Energy_pos_cum(t) / (m * g * dist_m(t))
    Requires:
      - energy_pos_cum (computed here) or recomputed inline
      - dist_m from sync (see add_distance_traveled())
      - cfg: mass_kg (float), gravity (defaults 9.81)
    """
    robot = cfg.get("robot", {})
    m = float(robot.get("mass_kg", np.nan))
    g = float(robot.get("gravity", 9.81))
    if not np.isfinite(m) or "dist_m" not in df:
        return pd.Series(np.nan, index=df.index)
    Epos = REGISTRY["energy_pos_cum"](df, cfg)
    denom = m * g * df["dist_m"].replace(0.0, np.nan)
    return (Epos / denom).replace([np.inf, -np.inf], np.nan)

# Public API: compute a set of metrics and return a new DataFrame with columns
def compute(df: pd.DataFrame, names: Iterable[str], cfg: dict) -> pd.DataFrame:
    """
    Append selected metrics as columns to a copy of df and return it.
    Missing inputs yield NaNs.
    """
    out = df.copy()
    for n in names:
        fn = REGISTRY.get(n)
        if fn is None:
            raise KeyError(f"Unknown metric '{n}'. Available: {list(REGISTRY)}")
        out[n] = fn(out, cfg)
    return out
