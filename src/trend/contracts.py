"""
Trend data structures — the vocabulary of trajectory intelligence.

These dataclasses define the shape of every trend analysis artifact.
They carry no logic; logic lives in the computers and scorers.

Design principle: every field is either measured from data or computed
from measured fields. No AI inference at this layer — just math.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TrendDirection(str, Enum):
    """Which way a metric is heading."""
    IMPROVING           = "improving"
    STABLE              = "stable"
    DECLINING           = "declining"
    INSUFFICIENT_DATA   = "insufficient_data"


class UrgencyLevel(str, Enum):
    """How urgently something needs attention."""
    NONE        = "none"
    WATCH       = "watch"
    INVESTIGATE = "investigate"
    ACT         = "act"
    URGENT      = "urgent"


# ---------------------------------------------------------------------------
# Timestamped value — the raw input
# ---------------------------------------------------------------------------

@dataclass
class TimestampedValue:
    """A single data point with a timestamp."""
    timestamp: datetime
    value: float

    def age_days(self, now: datetime | None = None) -> float:
        """How many days old this value is relative to *now*."""
        from datetime import timezone
        if now is None:
            now = datetime.now(timezone.utc)
        # Make both timezone-aware for safe subtraction
        if self.timestamp.tzinfo is None:
            ts = self.timestamp.replace(tzinfo=timezone.utc)
        else:
            ts = self.timestamp
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return max((now - ts).total_seconds() / 86400, 0.0)


# ---------------------------------------------------------------------------
# Metric definition — what to compute and where to find the data
# ---------------------------------------------------------------------------

@dataclass
class MetricDefinition:
    """Describes a single metric the TrendEngine tracks.

    Parameters
    ----------
    key : str
        Unique identifier like ``sleep.duration`` or ``spending.daily_total``.
    collection : str
        The LocalStore / SqliteStore collection name.
    field : str
        The key inside each record that holds the numeric value.
    higher_is_better : bool
        True for things like sleep hours; False for things like daily spend.
    aggregation : str
        How to aggregate multiple records per day: ``mean``, ``sum``, ``last``.
    """
    key: str
    collection: str
    field: str
    higher_is_better: bool = True
    aggregation: str = "mean"
    sense_name: str = ""  # Which sense's store to read from (e.g. "healthsense")


# ---------------------------------------------------------------------------
# Personal baseline
# ---------------------------------------------------------------------------

@dataclass
class PersonalBaseline:
    """Statistical profile of a metric over a personal history window.

    This is *your* normal, not a population average. All stats are
    computed from staleness-weighted data so recent values matter more.
    """
    mean: float
    median: float
    std_dev: float
    p25: float
    p75: float
    data_points: int
    window_days: int
    personal_target: float | None = None
    personal_acceptable_range: tuple[float, float] | None = None

    def deviation_from_normal(self, value: float) -> float:
        """How many standard deviations *value* is from the baseline mean.

        Positive = above mean, negative = below mean.
        Returns 0.0 if std_dev is zero (all identical values).
        """
        if self.std_dev == 0:
            return 0.0
        return (value - self.mean) / self.std_dev

    def is_within_normal(self, value: float) -> bool:
        """True if *value* falls within p25–p75 (the middle 50%)."""
        return self.p25 <= value <= self.p75


# ---------------------------------------------------------------------------
# Trend analysis
# ---------------------------------------------------------------------------

@dataclass
class TrendAnalysis:
    """Result of analysing the direction and velocity of a metric."""
    direction: TrendDirection
    velocity: float                          # slope in units/day
    velocity_label: str                      # rapid | moderate | gradual | flat
    trend_started: datetime | None           # when the current direction began
    duration_days: int                        # how long the current trend has lasted
    is_significant: bool                      # enough data and large enough effect?
    data_points: int
    confidence: float                         # 0–1
    current_value: float | None
    values_window: list[TimestampedValue]
    baseline: PersonalBaseline | None
    deviation_from_baseline: float | None     # in std_dev units


# ---------------------------------------------------------------------------
# Gap analysis
# ---------------------------------------------------------------------------

@dataclass
class GapAnalysis:
    """How far a metric is from its target — and whether you're closing in."""
    current_value: float
    current_label: str                        # e.g. "6.2 hours"
    target_value: float
    target_source: str                        # "threshold" | "baseline" | "personal"
    target_label: str                         # e.g. "7.5 hours"
    gap_size: float                           # absolute difference
    gap_percentage: float                     # gap as % of target
    gap_direction: str                        # "above" | "below" | "at_target"
    trend: TrendDirection
    closing: bool                             # is the gap shrinking?
    closing_rate: float | None                # units/day toward target (positive = closing)
    estimated_close_date: date | None         # when the gap would close at current rate
    story: str                                # human-readable narrative


# ---------------------------------------------------------------------------
# Trajectory urgency
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryUrgency:
    """How urgently a metric's trajectory demands attention."""
    level: UrgencyLevel
    score: float                              # 0–1 composite score
    factors: list[str]                        # human-readable reasons
    headline: str                             # one-line summary
    context: str                              # full narrative
    gap: GapAnalysis | None = None
    trend: TrendAnalysis | None = None


# ---------------------------------------------------------------------------
# Composite snapshot and report
# ---------------------------------------------------------------------------

@dataclass
class MetricSnapshot:
    """Everything the engine knows about one metric right now."""
    baseline: PersonalBaseline | None
    trend: TrendAnalysis | None
    gap: GapAnalysis | None
    urgency: TrajectoryUrgency | None
    staleness: float                          # 0–1 freshness score


@dataclass
class TrendReport:
    """The complete output of a TrendEngine.compute() call."""
    metrics: dict[str, MetricSnapshot] = field(default_factory=dict)
    overall_trajectory: str = ""              # one-paragraph summary

    def urgent_metrics(self) -> dict[str, MetricSnapshot]:
        """Return only metrics at 'act' or 'urgent' level."""
        return {
            k: v for k, v in self.metrics.items()
            if v.urgency and v.urgency.level in (UrgencyLevel.ACT, UrgencyLevel.URGENT)
        }

    def declining_metrics(self) -> dict[str, MetricSnapshot]:
        """Return only metrics whose trend is declining."""
        return {
            k: v for k, v in self.metrics.items()
            if v.trend and v.trend.direction == TrendDirection.DECLINING
        }


# ---------------------------------------------------------------------------
# Threshold config (used by thresholds.py loader)
# ---------------------------------------------------------------------------

@dataclass
class ThresholdConfig:
    """Personal target and acceptable range for one metric."""
    target: float
    acceptable_range: tuple[float, float]
    source: str = "default"                   # "default" | "personal" | "discovered"
