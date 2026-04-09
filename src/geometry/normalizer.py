"""
Normalizer — converts raw metric values to personal baseline ratios.

Every metric becomes a dimensionless number centered at 1.0:
  - 1.0 = at personal baseline (your normal)
  - >1.0 = above baseline (better than your normal)
  - <1.0 = below baseline (worse than your normal)

For higher_is_better=False metrics (spending, stress, etc.), the ratio
is inverted so "good" is always >1.0 regardless of the metric's polarity.
"""

from __future__ import annotations

from typing import Any

# Avoid division by zero
_EPSILON = 0.001
# Clamp range to prevent outliers from warping the shape
_MIN_NORMALIZED = 0.0
_MAX_NORMALIZED = 2.5


def normalize_values(
    current_values: dict[str, float],
    trend_report: Any = None,
) -> dict[str, float]:
    """Normalize all current metric values to baseline ratios.

    Parameters
    ----------
    current_values : dict[str, float]
        Raw metric values keyed by metric_key.
    trend_report : TrendReport, optional
        Contains PersonalBaseline for each tracked metric.

    Returns
    -------
    dict[str, float]
        Normalized values centered at 1.0.
    """
    try:
        from ..metric_catalog.catalog import ALL_METRICS
    except ImportError:
        ALL_METRICS = {}

    normalized: dict[str, float] = {}

    for key, raw in current_values.items():
        spec = ALL_METRICS.get(key)
        baseline_mean = _get_baseline_mean(key, spec, trend_report)

        if baseline_mean is None or baseline_mean < _EPSILON:
            # No baseline — use 1.0 (at baseline by definition)
            normalized[key] = 1.0
            continue

        # Zero-valued inverse metrics inflate to MAX when dividing baseline/0.
        # Treat as "at baseline" — zero with no personal data is neutral, not thriving.
        if raw < _EPSILON and spec and not spec.higher_is_better:
            normalized[key] = 1.0
            continue

        if spec and not spec.higher_is_better:
            # Inverted: lower raw = better = higher normalized
            # spending baseline $100, today $50 → normalized = 100/50 = 2.0 (good)
            # spending baseline $100, today $200 → normalized = 100/200 = 0.5 (bad)
            ratio = baseline_mean / max(raw, _EPSILON)
        else:
            # Normal: higher raw = better = higher normalized
            # sleep baseline 7.5, today 8.0 → normalized = 8.0/7.5 = 1.07 (good)
            ratio = raw / baseline_mean

        normalized[key] = max(_MIN_NORMALIZED, min(_MAX_NORMALIZED, ratio))

    return normalized


def _get_baseline_mean(
    key: str,
    spec: Any,
    trend_report: Any,
) -> float | None:
    """Get the personal baseline mean for a metric.

    Priority:
    1. TrendEngine PersonalBaseline (personal data)
    2. Catalog healthy_range midpoint (population default)
    """
    # Try TrendEngine baseline first (personal data)
    if trend_report is not None:
        metrics = getattr(trend_report, "metrics", {})
        # Try both catalog key and legacy key formats
        snapshot = metrics.get(key)
        if snapshot is not None:
            baseline = getattr(snapshot, "baseline", None)
            if baseline is not None:
                mean = getattr(baseline, "mean", None)
                if mean is not None and mean > _EPSILON:
                    return mean

    # Fall back to catalog healthy_range midpoint
    if spec is not None:
        low, high = spec.healthy_range
        midpoint = (low + high) / 2
        if midpoint > _EPSILON:
            return midpoint

    return None


def evaluate_level(key: str, raw_value: float) -> str:
    """Evaluate raw value against catalog thresholds.

    Returns "critical" | "warning" | "healthy" | "thriving".
    """
    try:
        from ..metric_catalog.catalog import ALL_METRICS
        spec = ALL_METRICS.get(key)
        if spec:
            return spec.evaluate(raw_value)
    except ImportError:
        pass
    return "healthy"
