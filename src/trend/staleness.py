"""
Staleness calculator — exponential decay for data freshness.

Recent data matters more than old data. A sleep reading from last night
tells you more than one from three weeks ago. This module applies that
intuition mathematically using exponential half-life decay.

    freshness = 2^(-age_days / half_life_days)

With a 7-day half life:
    - Today's data:   weight ~1.0
    - 7 days ago:     weight ~0.5
    - 14 days ago:    weight ~0.25
    - 30 days ago:    weight ~0.06
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from .contracts import TimestampedValue


class StalenessCalculator:
    """Compute freshness scores and apply staleness weighting."""

    def __init__(self, half_life_days: float = 7.0) -> None:
        if half_life_days <= 0:
            raise ValueError("half_life_days must be positive")
        self.half_life_days = half_life_days

    def freshness(self, age_days: float) -> float:
        """Return a 0–1 freshness score for a data point of a given age.

        Parameters
        ----------
        age_days : float
            How many days old the data point is.  Zero means "right now".

        Returns
        -------
        float
            1.0 for brand-new data, decaying toward 0 for very old data.
        """
        if age_days < 0:
            age_days = 0.0
        return math.pow(2.0, -age_days / self.half_life_days)

    def freshness_at(self, timestamp: datetime, now: datetime | None = None) -> float:
        """Return freshness for a specific timestamp."""
        if now is None:
            now = datetime.now(timezone.utc)
        # Ensure both are tz-aware
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        age = max((now - timestamp).total_seconds() / 86400, 0.0)
        return self.freshness(age)

    def weight_values(
        self,
        values: list[TimestampedValue],
        now: datetime | None = None,
    ) -> list[tuple[TimestampedValue, float]]:
        """Pair each value with its staleness-based weight.

        Parameters
        ----------
        values : list[TimestampedValue]
            Raw timestamped data points.
        now : datetime, optional
            Reference "now" for age calculation.  Defaults to UTC now.

        Returns
        -------
        list[tuple[TimestampedValue, float]]
            Each value paired with its freshness weight (0–1).
        """
        if now is None:
            now = datetime.now(timezone.utc)
        return [
            (v, self.freshness(v.age_days(now)))
            for v in values
        ]
