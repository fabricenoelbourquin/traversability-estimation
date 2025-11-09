from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml


FilterStage = Mapping[str, Any]


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


def _moving_average(values: np.ndarray, stage: FilterStage) -> np.ndarray:
    """ Apply Moving Average to the input values. """
    series = pd.Series(values)
    return _rolling(series, stage, method="mean").to_numpy()


def _moving_median(values: np.ndarray, stage: FilterStage) -> np.ndarray:
    """ Apply Moving Median to the input values. """
    series = pd.Series(values)
    return _rolling(series, stage, method="median").to_numpy()


def _ema(values: np.ndarray, stage: FilterStage) -> np.ndarray:
    """ Apply Exponential Moving Average (EMA) to the input values. """
    alpha = float(stage.get("alpha", 0.3))
    if not (0.0 < alpha <= 1.0):
        raise ValueError("EMA alpha must be in (0, 1].")
    series = pd.Series(values)
    min_periods = int(stage.get("min_periods", 1))
    adjust = bool(stage.get("adjust", False))
    return series.ewm(alpha=alpha, adjust=adjust, min_periods=min_periods).mean().to_numpy()


FILTER_FUNCS: dict[str, Any] = {
    # Dictionary mapping filter type names to their corresponding functions
    "moving_average": _moving_average,
    "moving_median": _moving_median,
    "ema": _ema,
    "exponential": _ema,
    "none": lambda values, stage: values.copy(),
}


def _normalize_chain(spec: Any) -> list[FilterStage]:
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


def apply_filter_chain(values: Sequence[float] | np.ndarray, chain: list[FilterStage]) -> np.ndarray:
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
        out = fn(out, stage)
    return out


def apply_named_filter(
    values: Sequence[float] | np.ndarray | None,
    filters_cfg: Mapping[str, Any] | None,
    signal_name: str,
    fallback_key: str | None = "default",
) -> FilterResult:
    """ Apply a named filter chain to the input values."""
    if values is None:
        return FilterResult(values=None, chain=[])
    chain = resolve_filter_chain(filters_cfg, signal_name, fallback_key)
    if not chain:
        return FilterResult(values=_to_array(values), chain=[])
    return FilterResult(values=apply_filter_chain(values, chain), chain=chain)


def resolve_filter_chain(
    filters_cfg: Mapping[str, Any] | None,
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


def load_metrics_config(cfg_path: Path) -> dict[str, Any]:
    """ Load the metrics configuration from a YAML file. """
    if not cfg_path.exists():
        return {}
    data = yaml.safe_load(cfg_path.read_text())
    return data or {}


def filter_signal(
    values: Sequence[float] | np.ndarray | None,
    signal_name: str,
    *,
    filters_cfg: Mapping[str, Any] | None = None,
    fallback_key: str | None = "default",
    log_fn: Callable[[str], None] | None = None,
) -> np.ndarray | None:
    """
    Convenience wrapper that applies the configured filter chain for `signal_name`
    and optionally logs the description.
    """
    res = apply_named_filter(values, filters_cfg, signal_name, fallback_key=fallback_key)
    desc = format_chain(res.chain)
    if desc and log_fn is not None:
        log_fn(f"[info] Filtering '{signal_name}': {desc}")
    return res.values
