# src/metrics.py
from __future__ import annotations
from typing import Callable, Dict, Iterable
import numpy as np
import pandas as pd

# --- registry ---
REGISTRY: Dict[str, Callable[[pd.DataFrame, dict], pd.Series]] = {}

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
