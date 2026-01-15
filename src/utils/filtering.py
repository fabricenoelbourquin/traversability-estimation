from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


FilterStage = Mapping[str, object]
FilterContext = Mapping[str, object] | None


@dataclass
class FilterResult:
    """Container describing the filtered values plus the spec that was applied."""
    values: np.ndarray | None
    chain: list[FilterStage]


def _to_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Convert input values to a 1D numpy array of floats."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        return arr.reshape(-1)
    return arr.copy()


def _rolling(series: pd.Series, stage: FilterStage, method: str) -> pd.Series:
    """ Apply a rolling operation to a pandas Series. """
    window = max(1, int(stage.get("window", 5)))
    center = bool(stage.get("center", True))
    min_periods = int(stage.get("min_periods", 1))
    if window <= 1:
        return series
    rolled = series.rolling(window=window, min_periods=min_periods, center=center)
    return getattr(rolled, method)()


def _moving_average(values: np.ndarray, stage: FilterStage, *, context: FilterContext = None) -> np.ndarray:
    """ Apply Moving Average to the input values. """
    series = pd.Series(values)
    return _rolling(series, stage, method="mean").to_numpy()


def _moving_median(values: np.ndarray, stage: FilterStage, *, context: FilterContext = None) -> np.ndarray:
    """ Apply Moving Median to the input values. """
    series = pd.Series(values)
    return _rolling(series, stage, method="median").to_numpy()


def _ema(values: np.ndarray, stage: FilterStage, *, context: FilterContext = None) -> np.ndarray:
    """ Apply Exponential Moving Average (EMA) to the input values. """
    alpha = float(stage.get("alpha", 0.3))
    if not (0.0 < alpha <= 1.0):
        raise ValueError("EMA alpha must be in (0, 1].")
    series = pd.Series(values)
    min_periods = int(stage.get("min_periods", 1))
    adjust = bool(stage.get("adjust", False))
    return series.ewm(alpha=alpha, adjust=adjust, min_periods=min_periods).mean().to_numpy()


def _hampel(values: np.ndarray, stage: FilterStage, *, context: FilterContext = None) -> np.ndarray:
    """
    Hampel filter: rolling median with MAD-based outlier replacement.
    Points with |x - median| > k * MAD are replaced by the median.
    """
    window = max(1, int(stage.get("window", 11)))
    if window <= 1:
        return values.copy()
    k = float(stage.get("k", 3.0))
    center = bool(stage.get("center", True))
    min_periods = int(stage.get("min_periods", 1))

    series = pd.Series(values)
    rolled = series.rolling(window=window, center=center, min_periods=min_periods)
    med = rolled.median()
    mad = (series - med).abs().rolling(window=window, center=center, min_periods=min_periods).median()
    mad = mad.replace(0.0, np.nan)  # avoid zero division; NaNs will fail the mask
    thresh = k * 1.4826 * mad
    mask = (series - med).abs() > thresh
    out = series.copy()
    out[mask] = med[mask]
    return out.to_numpy()


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values.astype(np.float64, copy=True)
    return (
        pd.Series(values)
        .rolling(window=window, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=np.float64)
    )


def mad_outlier_reject(values: np.ndarray, z_thresh: float = 3.5) -> np.ndarray:
    out = values.astype(np.float64, copy=True)
    finite = out[np.isfinite(out)]
    if finite.size < 3:
        return out
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    if not np.isfinite(mad) or mad <= 0.0:
        return out
    z = 0.6745 * (out - med) / mad
    out[np.abs(z) > z_thresh] = np.nan
    return out


def gaussian_kernel_1d(sigma: float, radius: int) -> np.ndarray:
    if sigma <= 0.0 or radius <= 0:
        return np.array([1.0], dtype=np.float64)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    ksum = float(np.sum(kernel))
    if not np.isfinite(ksum) or ksum <= 0.0:
        return np.array([1.0], dtype=np.float64)
    return (kernel / ksum).astype(np.float64)


def gaussian_smooth_1d_nan(arr: np.ndarray, sigma: float, window: int | None = None) -> np.ndarray:
    """Gaussian smooth with NaN-aware normalization."""
    if arr.size == 0 or not np.isfinite(sigma) or sigma <= 0.0:
        return arr.astype(np.float64, copy=True)
    if window is None:
        radius = int(round(3.0 * sigma))
    else:
        radius = int(max(1, window // 2))
    kernel = gaussian_kernel_1d(sigma, radius)
    pad = radius
    values = arr.astype(np.float64, copy=True)
    mask = np.isfinite(values).astype(np.float64)
    values[~np.isfinite(values)] = 0.0

    values_pad = np.pad(values, pad, mode="reflect")
    mask_pad = np.pad(mask, pad, mode="reflect")
    num = np.convolve(values_pad, kernel, mode="valid")
    den = np.convolve(mask_pad, kernel, mode="valid")
    out = num / np.clip(den, 1e-12, None)
    out[den == 0.0] = np.nan
    return out.astype(np.float64)


def kalman_cv_1d(
    t_s: np.ndarray,
    z: np.ndarray,
    *,
    process_var: float,
    meas_var: float,
    init_pos_var: float = 1.0,
    init_vel_var: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Constant-velocity Kalman filter + RTS smoother for 1D position.
    Returns (smoothed_position, smoothed_velocity).
    """
    t = np.asarray(t_s, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    n = len(z)
    if n == 0:
        return z.copy(), z.copy()

    q = float(process_var)
    r = float(meas_var)
    if not np.isfinite(q) or q < 0.0:
        raise ValueError("process_var must be finite and >= 0.")
    if not np.isfinite(r) or r <= 0.0:
        raise ValueError("meas_var must be finite and > 0.")

    x_f = np.zeros((n, 2), dtype=np.float64)
    P_f = np.zeros((n, 2, 2), dtype=np.float64)
    x_pred = np.zeros((n, 2), dtype=np.float64)
    P_pred = np.zeros((n, 2, 2), dtype=np.float64)

    x = np.array([z[0] if np.isfinite(z[0]) else 0.0, 0.0], dtype=np.float64)
    P = np.diag([float(init_pos_var), float(init_vel_var)]).astype(np.float64)

    for k in range(n):
        if k > 0:
            dt = t[k] - t[k - 1]
            if not np.isfinite(dt) or dt <= 0.0:
                dt = 0.0
            F = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
            q11 = (dt**3) / 3.0
            q12 = (dt**2) / 2.0
            q22 = dt
            Q = q * np.array([[q11, q12], [q12, q22]], dtype=np.float64)
            x = F @ x
            P = F @ P @ F.T + Q

        x_pred[k] = x
        P_pred[k] = P

        if np.isfinite(z[k]):
            y = z[k] - x[0]
            S = P[0, 0] + r
            if S <= 0.0 or not np.isfinite(S):
                S = r
            K = P[:, 0] / S
            x = x + K * y
            P = P - np.outer(K, P[0, :])

        x_f[k] = x
        P_f[k] = P

    x_s = x_f.copy()
    P_s = P_f.copy()
    for k in range(n - 2, -1, -1):
        dt = t[k + 1] - t[k]
        if not np.isfinite(dt) or dt <= 0.0:
            dt = 0.0
        F = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        P_pred_k1 = P_pred[k + 1]
        if not np.all(np.isfinite(P_pred_k1)):
            continue
        C = P_f[k] @ F.T @ np.linalg.pinv(P_pred_k1)
        x_s[k] = x_f[k] + C @ (x_s[k + 1] - x_pred[k + 1])
        P_s[k] = P_f[k] + C @ (P_s[k + 1] - P_pred_k1) @ C.T

    return x_s[:, 0], x_s[:, 1]


def kalman_smooth_xy(
    t_s: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    process_var: float,
    meas_var: float,
    init_pos_var: float = 1.0,
    init_vel_var: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run 1D constant-velocity Kalman smoothing on x and y separately."""
    x_s, vx_s = kalman_cv_1d(
        t_s,
        x,
        process_var=process_var,
        meas_var=meas_var,
        init_pos_var=init_pos_var,
        init_vel_var=init_vel_var,
    )
    y_s, vy_s = kalman_cv_1d(
        t_s,
        y,
        process_var=process_var,
        meas_var=meas_var,
        init_pos_var=init_pos_var,
        init_vel_var=init_vel_var,
    )
    return x_s, y_s, vx_s, vy_s


def rate_limit_angles(angle_deg: np.ndarray, t_s: np.ndarray, max_rate_deg_s: float) -> np.ndarray:
    """Clamp per-sample changes to a maximum rate (deg/s)."""
    out = angle_deg.astype(np.float64, copy=True)
    if out.size == 0 or not np.isfinite(max_rate_deg_s) or max_rate_deg_s <= 0.0:
        return out
    dt = np.diff(t_s, prepend=t_s[0])
    for i in range(1, len(out)):
        if not np.isfinite(out[i]) or not np.isfinite(out[i - 1]) or not np.isfinite(dt[i]):
            continue
        if dt[i] <= 0.0:
            continue
        max_delta = max_rate_deg_s * dt[i]
        delta = out[i] - out[i - 1]
        if delta > max_delta:
            out[i] = out[i - 1] + max_delta
        elif delta < -max_delta:
            out[i] = out[i - 1] - max_delta
    return out


def despike_hampel(angle_deg: np.ndarray, window: int, z_thresh: float) -> np.ndarray:
    """Replace spikes using a Hampel filter (rolling median + MAD)."""
    if window <= 1 or angle_deg.size == 0:
        return angle_deg.astype(np.float64, copy=True)
    series = pd.Series(angle_deg)
    med = series.rolling(window=window, center=True, min_periods=1).median()
    mad = (series - med).abs().rolling(window=window, center=True, min_periods=1).median()
    mad = mad.replace(0.0, np.nan)
    z = 0.6745 * (series - med) / mad
    out = series.to_numpy(dtype=np.float64)
    z_vals = z.to_numpy(dtype=np.float64)
    med_vals = med.to_numpy(dtype=np.float64)
    out[np.abs(z_vals) > z_thresh] = med_vals[np.abs(z_vals) > z_thresh]
    return out


def _mad_outlier(values: np.ndarray, stage: FilterStage, *, context: FilterContext = None) -> np.ndarray:
    z_thresh = float(stage.get("z_thresh", stage.get("k", 3.5)))
    if not np.isfinite(z_thresh) or z_thresh <= 0.0:
        return values.copy()
    return mad_outlier_reject(values, z_thresh)


def _gaussian_smooth(values: np.ndarray, stage: FilterStage, *, context: FilterContext = None) -> np.ndarray:
    sigma = float(stage.get("sigma", stage.get("value_gauss_sigma", 0.0)))
    window = stage.get("window")
    return gaussian_smooth_1d_nan(values, sigma, window)


def _rate_limit(values: np.ndarray, stage: FilterStage, *, context: FilterContext = None) -> np.ndarray:
    max_rate = float(stage.get("max_rate_deg_s", stage.get("max_rate", 0.0)))
    if not np.isfinite(max_rate) or max_rate <= 0.0:
        return values.copy()
    if context is None or "t_s" not in context:
        raise ValueError("rate_limit filter requires context['t_s'].")
    t_s = np.asarray(context["t_s"], dtype=np.float64)
    return rate_limit_angles(values, t_s, max_rate)


FILTER_FUNCS: dict[str, Callable[..., np.ndarray]] = {
    # Dictionary mapping filter type names to their corresponding functions
    "moving_average": _moving_average,
    "moving_median": _moving_median,
    "ema": _ema,
    "exponential": _ema,
    "hampel": _hampel,
    "mad_outlier": _mad_outlier,
    "gaussian_smooth": _gaussian_smooth,
    "rate_limit": _rate_limit,
    "none": lambda values, stage, *, context=None: values.copy(),
}


def build_dem_pitch_roll_chain(filter_params: Mapping[str, Any] | None) -> list[FilterStage]:
    if not filter_params:
        return []
    params = dict(filter_params)
    chain: list[FilterStage] = []

    smooth_window = int(params.get("smooth_window", 0))
    if smooth_window >= 1:
        chain.append({
            "type": "moving_median",
            "window": smooth_window,
            "center": True,
            "min_periods": 1,
        })

    mad_z_thresh = float(params.get("mad_z_thresh", 0.0))
    if np.isfinite(mad_z_thresh) and mad_z_thresh > 0.0:
        chain.append({
            "type": "mad_outlier",
            "z_thresh": mad_z_thresh,
        })

    if bool(params.get("rate_hampel_filter", False)):
        hampel_window = int(params.get("rate_hampel_window", 0))
        hampel_z = float(params.get("rate_hampel_z", 0.0))
        if hampel_window >= 1 and np.isfinite(hampel_z) and hampel_z > 0.0:
            chain.append({
                "type": "hampel",
                "window": hampel_window,
                "k": hampel_z,
                "center": True,
                "min_periods": 1,
            })

        max_rate = float(params.get("rate_hampel_max_rate", 0.0))
        if np.isfinite(max_rate) and max_rate > 0.0:
            chain.append({
                "type": "rate_limit",
                "max_rate_deg_s": max_rate,
            })

    value_gauss_sigma = float(params.get("value_gauss_sigma", 0.0))
    if np.isfinite(value_gauss_sigma) and value_gauss_sigma > 0.0:
        chain.append({
            "type": "gaussian_smooth",
            "sigma": value_gauss_sigma,
        })

    return chain


def _normalize_chain(spec: object) -> list[FilterStage]:
    """ Normalize a filter specification into a list of filter stages. """
    if spec is None:
        return []
    if isinstance(spec, str):
        return [{"type": spec}]
    if isinstance(spec, Mapping):
        if "chain" in spec:
            return _normalize_chain(spec["chain"])
        if "type" not in spec:
            raise ValueError(f"Filter spec {spec} missing 'type'.")
        if not spec.get("enabled", True):
            return []
        return [spec]
    if isinstance(spec, Sequence) and not isinstance(spec, (bytes, bytearray)):
        chain: list[FilterStage] = []
        for item in spec:
            chain.extend(_normalize_chain(item))
        return chain
    raise TypeError(f"Unsupported filter specification: {spec!r}")


def apply_filter_chain(
    values: Sequence[float] | np.ndarray,
    chain: list[FilterStage],
    *,
    context: FilterContext = None,
) -> np.ndarray:
    """ Apply a chain of filters to the input values. """
    arr = _to_array(values)
    out = arr
    for stage in chain:
        ftype = stage.get("type")
        if ftype is None or ftype == "none":
            continue
        fn = FILTER_FUNCS.get(ftype)
        if fn is None:
            raise KeyError(f"Unknown filter type '{ftype}'. Available: {sorted(FILTER_FUNCS)}")
        out = fn(out, stage, context=context)
    return out


def apply_named_filter(
    values: Sequence[float] | np.ndarray | None,
    filters_cfg: Mapping[str, object] | None,
    signal_name: str,
    fallback_key: str | None = "default",
    *,
    context: FilterContext = None,
) -> FilterResult:
    """ Apply a named filter chain to the input values, optionally using context."""
    if values is None:
        return FilterResult(values=None, chain=[])
    chain = resolve_filter_chain(filters_cfg, signal_name, fallback_key)
    if not chain:
        return FilterResult(values=_to_array(values), chain=[])
    return FilterResult(values=apply_filter_chain(values, chain, context=context), chain=chain)


def resolve_filter_chain(
    filters_cfg: Mapping[str, object] | None,
    signal_name: str,
    fallback_key: str | None = "default",
) -> list[FilterStage]:
    """ Resolve the filter chain for a given signal name from the filters configuration. """
    if not filters_cfg:
        return []
    spec = filters_cfg.get(signal_name)
    if spec is None and fallback_key:
        spec = filters_cfg.get(fallback_key)
    return _normalize_chain(spec)


def format_chain(chain: Sequence[FilterStage]) -> str:
    """ Format a filter chain into a human-readable string. """
    parts = []
    for stage in chain:
        ftype = stage.get("type")
        if ftype in (None, "none"):
            continue
        desc = ftype
        extras = {k: v for k, v in stage.items() if k not in {"type", "enabled"}}
        if extras:
            extras_str = ", ".join(f"{k}={v}" for k, v in extras.items())
            desc = f"{desc}({extras_str})"
        parts.append(desc)
    return " -> ".join(parts)


def load_metrics_config(cfg_path: Path) -> dict[str, object]:
    """ Load the metrics configuration from a YAML file. """
    if not cfg_path.exists():
        return {}
    data = yaml.safe_load(cfg_path.read_text())
    return data or {}


def filter_signal(
    values: Sequence[float] | np.ndarray | None,
    signal_name: str,
    *,
    filters_cfg: Mapping[str, object] | None = None,
    fallback_key: str | None = "default",
    log_fn: Callable[[str], None] | None = None,
    context: FilterContext = None,
) -> np.ndarray | None:
    """
    Convenience wrapper that applies the configured filter chain for `signal_name`
    and optionally logs the description. Context can supply extra data (e.g., time).
    """
    res = apply_named_filter(values, filters_cfg, signal_name, fallback_key=fallback_key, context=context)
    desc = format_chain(res.chain)
    if desc and log_fn is not None:
        log_fn(f"[info] Filtering '{signal_name}': {desc}")
    return res.values
