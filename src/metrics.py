# src/metrics.py
from __future__ import annotations
from typing import Callable, Iterable
import numpy as np
import pandas as pd
from utils.filtering import filter_signal

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

def _nan_series(df: pd.DataFrame) -> pd.Series:
    return pd.Series(np.nan, index=df.index)

def _joint_power_terms(df: pd.DataFrame) -> pd.DataFrame | None:
    qd_cols, tau_cols = _find_joint_columns(df)
    if not qd_cols or not tau_cols:
        return None
    joints = sorted(set(c[3:] for c in qd_cols) & set(c[4:] for c in tau_cols))
    if not joints:
        return None
    data = {j: df[f"tau_{j}"] * df[f"qd_{j}"] for j in joints}
    return pd.DataFrame(data, index=df.index)


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
    # read from nested params (metrics.yaml: params.max_speed_mps)
    vmax = float((cfg.get("params") or {}).get("max_speed_mps", 1.0))
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
    Mechanical power usage [W]: sum_j |tau_j * qd_j| (no regen credit).
    """
    terms = _joint_power_terms(df)
    if terms is None:
        return _nan_series(df)
    return terms.abs().sum(axis=1)

@metric("power_mech_signed")
def power_mech_signed(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """
    Signed mechanical power [W]: sum_j tau_j * qd_j (regen captured).
    """
    terms = _joint_power_terms(df)
    if terms is None:
        return _nan_series(df)
    return terms.sum(axis=1)

@metric("cost_of_transport")
def cost_of_transport(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """
    Instantaneous Cost of Transport (dimensionless):
      COT = power_mech / (m * g * |v_actual|).
    Samples with |v_actual| below a configurable threshold are clamped to 0.
    """
    d = _ensure_speed_columns(df)
    if "v_actual" not in d.columns:
        return pd.Series(np.nan, index=df.index)

    robot = cfg.get("robot") or {}
    m = float(robot.get("mass_kg", np.nan))
    g = float(robot.get("gravity", 9.81))
    if not np.isfinite(m) or m <= 0.0 or not np.isfinite(g) or g <= 0.0:
        return pd.Series(np.nan, index=df.index)

    power = REGISTRY["power_mech"](df, cfg)
    if power.isna().all():
        return pd.Series(np.nan, index=df.index)

    params = cfg.get("params") or {}
    min_speed = float(params.get("min_speed_for_power_norm", 0.1))
    min_speed = max(min_speed, 1e-4)

    filters_cfg = cfg.get("filters", {})

    speed_raw = d["v_actual"].to_numpy(dtype=np.float64)
    speed_filtered = filter_signal(speed_raw, "cost_of_transport_speed", filters_cfg=filters_cfg, log_fn=None)
    speed_use = speed_filtered if speed_filtered is not None else speed_raw
    speed_mag = np.abs(speed_use)

    power_vals = power.to_numpy(dtype=np.float64)
    power_filtered = filter_signal(power_vals, "cost_of_transport_power", filters_cfg=filters_cfg, log_fn=None)
    power_use = power_vals if power_filtered is None else power_filtered

    out = np.full_like(speed_mag, np.nan, dtype=np.float64)
    valid = np.isfinite(speed_mag) & np.isfinite(power_use)
    if not np.any(valid):
        return pd.Series(out, index=df.index)

    slow_mask = valid & (speed_mag < min_speed)
    fast_mask = valid & ~slow_mask
    out[slow_mask] = 0.0
    out[fast_mask] = power_use[fast_mask] / (m * g * speed_mag[fast_mask])

    ratio_filtered = filter_signal(out, "cost_of_transport", filters_cfg=filters_cfg, log_fn=None)
    final_vals = out if ratio_filtered is None else ratio_filtered
    return pd.Series(final_vals, index=df.index)

@metric("energy_pos_cum")
def energy_pos_cum(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """
    Cumulative positive mechanical energy [J]: ∫ sum_j max(0, tau_j*qd_j) dt.
    """
    if "t" not in df:
        return pd.Series(np.nan, index=df.index)
    # Raw instantaneous mechanical power (unfiltered)
    P = REGISTRY["power_mech"](df, cfg)
    if P.isna().all():
        return pd.Series(np.nan, index=df.index)
    # Optionally smooth power before clamping/integration using filters.power_mech
    filters_cfg = cfg.get("filters", {})
    P_filt = filter_signal(P.to_numpy(dtype=np.float64), "power_mech", filters_cfg=filters_cfg, log_fn=None)
    P_use = P.to_numpy(dtype=np.float64) if P_filt is None else P_filt
    dt = _dt_seconds(df["t"])
    e_step = np.maximum(P_use, 0.0) * dt
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
