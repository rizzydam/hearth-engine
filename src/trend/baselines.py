"""
Baseline computer — builds a PersonalBaseline from historical data.

A personal baseline is YOUR normal, not a population norm.  It answers
the question: "given my history, what should this metric look like on a
typical day?"

Implementation uses only Python stdlib (math + statistics).  No numpy.

Data is weighted by staleness so recent weeks shape the baseline more
than months-old readings.  Minimum 5 data points required.
"""

from __future__ import annotations

import math
import statistics
from datetime import datetime, timezone

from .contracts import PersonalBaseline, TimestampedValue
from .staleness import StalenessCalculator


# Minimum data points needed to compute a meaningful baseline
MIN_DATA_POINTS = 5


class BaselineComputer:
    """Compute a PersonalBaseline from timestamped metric history."""

    def __init__(self, staleness: StalenessCalculator | None = None) -> None:
        self._staleness = staleness or StalenessCalculator()

    def compute(
        self,
        metric_key: str,
        values: list[TimestampedValue],
        window_days: int = 90,
        now: datetime | None = None,
    ) -> PersonalBaseline | None:
        """Build a staleness-weighted PersonalBaseline.

        Parameters
        ----------
        metric_key : str
            Identifier for the metric (used only for logging/debugging).
        values : list[TimestampedValue]
            Historical readings.  Need not be sorted.
        window_days : int
            How far back to look.  Data older than this is discarded entirely.
        now : datetime, optional
            Reference time.  Defaults to UTC now.

        Returns
        -------
        PersonalBaseline or None
            None if fewer than MIN_DATA_POINTS readings exist in the window.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        # Filter to window
        windowed = [v for v in values if v.age_days(now) <= window_days]
        if len(windowed) < MIN_DATA_POINTS:
            return None

        # Get staleness weights
        weighted_pairs = self._staleness.weight_values(windowed, now)

        # Raw (unweighted) values for percentiles — percentiles on weighted
        # distributions need interpolation that adds complexity without much
        # benefit at this scale.
        raw_vals = sorted(v.value for v in windowed)

        # Weighted mean
        total_weight = sum(w for _, w in weighted_pairs)
        if total_weight == 0:
            return None
        w_mean = sum(v.value * w for v, w in weighted_pairs) / total_weight

        # Weighted standard deviation
        w_var = sum(w * (v.value - w_mean) ** 2 for v, w in weighted_pairs) / total_weight
        w_std = math.sqrt(w_var)

        # Unweighted median and percentiles using stdlib
        median = statistics.median(raw_vals)
        p25 = _percentile(raw_vals, 0.25)
        p75 = _percentile(raw_vals, 0.75)

        return PersonalBaseline(
            mean=round(w_mean, 4),
            median=round(median, 4),
            std_dev=round(w_std, 4),
            p25=round(p25, 4),
            p75=round(p75, 4),
            data_points=len(windowed),
            window_days=window_days,
        )


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Compute a percentile from a pre-sorted list using linear interpolation.

    Parameters
    ----------
    sorted_values : list[float]
        Values already sorted ascending.
    pct : float
        Percentile as a fraction (0.25 = 25th percentile).

    Returns
    -------
    float
    """
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    k = pct * (n - 1)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return sorted_values[lo]
    frac = k - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac
