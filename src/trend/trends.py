"""
Trend computer — direction, velocity, and change-point detection.

Given a series of timestamped values and an optional baseline, this
module determines:
    - Direction: improving / stable / declining / insufficient_data
    - Velocity: how fast (slope in units/day)
    - Velocity label: rapid / moderate / gradual / flat
    - Change point: where the current trend began
    - Duration: how many days the current trend has lasted

Implementation uses simple least-squares linear regression and a
mean-difference change-point detector.  No numpy, no scipy.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta

from .contracts import (
    PersonalBaseline,
    TimestampedValue,
    TrendAnalysis,
    TrendDirection,
)
from .staleness import StalenessCalculator


# Minimum data points for a significant trend
MIN_DATA_POINTS = 5


class TrendComputer:
    """Analyse the trajectory of a metric over time."""

    def __init__(
        self,
        staleness: StalenessCalculator | None = None,
        higher_is_better: bool = True,
    ) -> None:
        self._staleness = staleness or StalenessCalculator()
        self.higher_is_better = higher_is_better

    def analyze(
        self,
        metric_key: str,
        values: list[TimestampedValue],
        baseline: PersonalBaseline | None = None,
        now: datetime | None = None,
    ) -> TrendAnalysis:
        """Compute a full TrendAnalysis for the given metric values.

        Parameters
        ----------
        metric_key : str
            Human identifier for this metric.
        values : list[TimestampedValue]
            Historical readings, need not be sorted.
        baseline : PersonalBaseline, optional
            If available, used for deviation calculation and significance.
        now : datetime, optional
            Reference time.  Defaults to UTC now.

        Returns
        -------
        TrendAnalysis
        """
        if now is None:
            now = datetime.now(timezone.utc)

        # Sort by timestamp ascending
        sorted_vals = sorted(values, key=lambda v: v.timestamp)
        current_value = sorted_vals[-1].value if sorted_vals else None

        # Not enough data?
        if len(sorted_vals) < MIN_DATA_POINTS:
            return TrendAnalysis(
                direction=TrendDirection.INSUFFICIENT_DATA,
                velocity=0.0,
                velocity_label="flat",
                trend_started=None,
                duration_days=0,
                is_significant=False,
                data_points=len(sorted_vals),
                confidence=0.0,
                current_value=current_value,
                values_window=sorted_vals,
                baseline=baseline,
                deviation_from_baseline=None,
            )

        # --- Linear regression (least-squares) ---
        slope, intercept, r_squared = _linear_regression(sorted_vals, now)

        # --- Significance ---
        # A trend is significant if we have enough data AND the slope
        # is large relative to the data's variability.
        std_dev = baseline.std_dev if baseline else _simple_std(sorted_vals)
        # Guard against zero std_dev (perfectly constant data)
        is_significant = (
            len(sorted_vals) >= MIN_DATA_POINTS
            and std_dev > 0
            and abs(slope) > 0.05 * std_dev  # slope must exceed 5% of std per day
        )

        # --- Direction ---
        if not is_significant:
            direction = TrendDirection.STABLE
        elif self.higher_is_better:
            direction = TrendDirection.IMPROVING if slope > 0 else TrendDirection.DECLINING
        else:
            # Lower is better (e.g. spending) — negative slope is improving
            direction = TrendDirection.IMPROVING if slope < 0 else TrendDirection.DECLINING

        # --- Velocity label ---
        if not is_significant:
            velocity_label = "flat"
        else:
            abs_slope = abs(slope)
            if std_dev > 0:
                daily_ratio = abs_slope / std_dev
            else:
                daily_ratio = 0.0
            if daily_ratio > 2.0:
                velocity_label = "rapid"
            elif daily_ratio > 1.0:
                velocity_label = "moderate"
            else:
                velocity_label = "gradual"

        # --- Change-point detection ---
        change_idx = _find_change_point(sorted_vals)
        if change_idx is not None and change_idx < len(sorted_vals):
            trend_started = sorted_vals[change_idx].timestamp
            duration_days = max(
                int((now - _tz_aware(trend_started)).total_seconds() / 86400),
                0,
            )
        else:
            # No clear change point — trend spans the whole window
            trend_started = sorted_vals[0].timestamp
            duration_days = max(
                int((now - _tz_aware(sorted_vals[0].timestamp)).total_seconds() / 86400),
                0,
            )

        # --- Deviation from baseline ---
        deviation = None
        if baseline is not None and current_value is not None:
            deviation = baseline.deviation_from_normal(current_value)

        # --- Confidence ---
        # Combines r-squared (fit quality) with data quantity
        data_factor = min(len(sorted_vals) / 30, 1.0)  # saturates at 30 points
        confidence = round(r_squared * data_factor, 4)

        return TrendAnalysis(
            direction=direction,
            velocity=round(slope, 6),
            velocity_label=velocity_label,
            trend_started=trend_started,
            duration_days=duration_days,
            is_significant=is_significant,
            data_points=len(sorted_vals),
            confidence=confidence,
            current_value=current_value,
            values_window=sorted_vals,
            baseline=baseline,
            deviation_from_baseline=round(deviation, 4) if deviation is not None else None,
        )


# ---------------------------------------------------------------------------
# Pure-math helpers (no external dependencies)
# ---------------------------------------------------------------------------

def _tz_aware(dt: datetime) -> datetime:
    """Ensure a datetime is timezone-aware (assume UTC if naive)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _linear_regression(
    values: list[TimestampedValue],
    now: datetime,
) -> tuple[float, float, float]:
    """Simple least-squares regression: value = slope * day_index + intercept.

    The x-axis is "days since the earliest reading" so the slope is in
    units-per-day.

    Returns (slope, intercept, r_squared).
    """
    if len(values) < 2:
        return 0.0, 0.0, 0.0

    t0 = _tz_aware(values[0].timestamp)
    xs: list[float] = []
    ys: list[float] = []
    for v in values:
        day = (_tz_aware(v.timestamp) - t0).total_seconds() / 86400
        xs.append(day)
        ys.append(v.value)

    n = len(xs)
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    sum_x2 = sum(x * x for x in xs)

    denom = n * sum_x2 - sum_x * sum_x
    if denom == 0:
        return 0.0, sum_y / n if n else 0.0, 0.0

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n

    # R-squared
    mean_y = sum_y / n
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    r_squared = max(r_squared, 0.0)  # clamp negative values

    return slope, intercept, r_squared


def _find_change_point(values: list[TimestampedValue]) -> int | None:
    """Find the index where the trend most likely reversed.

    Uses the maximum mean-difference method: for each candidate split
    point, compute the difference between the mean of the left segment
    and the mean of the right segment.  The split with the largest
    absolute difference is the most likely change point.

    We require at least 3 data points on each side of the split.

    Returns the index of the first element of the "new" segment,
    or None if no meaningful split exists.
    """
    n = len(values)
    min_segment = 3
    if n < min_segment * 2:
        return None

    raw = [v.value for v in values]
    total = sum(raw)
    best_idx = None
    best_diff = 0.0

    left_sum = sum(raw[:min_segment])
    for i in range(min_segment, n - min_segment + 1):
        left_sum += raw[i - 1] if i > min_segment else 0
        # Recompute left_sum properly: sum of raw[0:i]
        pass

    # Simpler and clearer: precompute prefix sums
    prefix = [0.0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + raw[i]

    for i in range(min_segment, n - min_segment + 1):
        left_mean = prefix[i] / i
        right_mean = (prefix[n] - prefix[i]) / (n - i)
        diff = abs(right_mean - left_mean)
        if diff > best_diff:
            best_diff = diff
            best_idx = i

    return best_idx


def _simple_std(values: list[TimestampedValue]) -> float:
    """Population standard deviation from raw values (no weighting)."""
    if len(values) < 2:
        return 0.0
    vals = [v.value for v in values]
    mean = sum(vals) / len(vals)
    variance = sum((x - mean) ** 2 for x in vals) / len(vals)
    return math.sqrt(variance)
