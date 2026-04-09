"""
Gap analyser — how far from target, and is the gap closing?

A gap is the distance between where you are and where you want to be.
This module calculates the gap, determines whether your current trend
is closing or widening it, and projects when you'll reach your target
at the current rate.

It also builds a human-readable "story" that makes the numbers personal.
"""

from __future__ import annotations

from datetime import datetime, date, timezone, timedelta

from .contracts import (
    GapAnalysis,
    PersonalBaseline,
    ThresholdConfig,
    TrendAnalysis,
    TrendDirection,
)


class GapAnalyzer:
    """Analyse the gap between current performance and target."""

    def __init__(self, higher_is_better: bool = True) -> None:
        self.higher_is_better = higher_is_better

    def analyze(
        self,
        metric_key: str,
        current: float,
        baseline: PersonalBaseline | None,
        trend: TrendAnalysis | None,
        threshold: ThresholdConfig | None = None,
        now: datetime | None = None,
    ) -> GapAnalysis | None:
        """Compute a GapAnalysis.

        Parameters
        ----------
        metric_key : str
            Human identifier for this metric.
        current : float
            The most recent value.
        baseline : PersonalBaseline, optional
            Historical baseline for this metric.
        trend : TrendAnalysis, optional
            Current trend analysis.
        threshold : ThresholdConfig, optional
            Personal target and acceptable range.
        now : datetime, optional
            Reference time.

        Returns
        -------
        GapAnalysis or None
            None if no target can be determined.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        # Determine target
        target, target_source = self._resolve_target(baseline, threshold)
        if target is None:
            return None

        # Gap calculation
        gap_size = abs(current - target)
        gap_percentage = (gap_size / abs(target) * 100) if target != 0 else 0.0

        # Direction
        if abs(current - target) < 0.001:
            gap_direction = "at_target"
        elif self.higher_is_better:
            gap_direction = "below" if current < target else "above"
        else:
            gap_direction = "above" if current > target else "below"

        # Trend info
        trend_dir = trend.direction if trend else TrendDirection.INSUFFICIENT_DATA
        velocity = trend.velocity if trend else 0.0

        # Is the gap closing?
        closing = self._is_closing(current, target, velocity)

        # Closing rate (positive = closing, negative = widening)
        if closing:
            closing_rate = abs(velocity)
        elif velocity != 0:
            closing_rate = -abs(velocity)
        else:
            closing_rate = None

        # Estimated close date
        estimated_close = None
        if closing and closing_rate and closing_rate > 0 and gap_size > 0:
            days_to_close = gap_size / closing_rate
            estimated_close = (now + timedelta(days=days_to_close)).date()

        # Labels
        current_label = _format_value(metric_key, current)
        target_label = _format_value(metric_key, target)

        # Story
        story = self._build_story(
            metric_key, current, target, baseline, trend,
            gap_size, gap_direction, closing, estimated_close,
        )

        return GapAnalysis(
            current_value=round(current, 4),
            current_label=current_label,
            target_value=round(target, 4),
            target_source=target_source,
            target_label=target_label,
            gap_size=round(gap_size, 4),
            gap_percentage=round(gap_percentage, 2),
            gap_direction=gap_direction,
            trend=trend_dir,
            closing=closing,
            closing_rate=round(closing_rate, 6) if closing_rate is not None else None,
            estimated_close_date=estimated_close,
            story=story,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_target(
        self,
        baseline: PersonalBaseline | None,
        threshold: ThresholdConfig | None,
    ) -> tuple[float | None, str]:
        """Determine the target value and its source."""
        if threshold is not None:
            return threshold.target, "threshold"
        if baseline is not None and baseline.personal_target is not None:
            return baseline.personal_target, "personal"
        if baseline is not None:
            return baseline.mean, "baseline"
        return None, ""

    def _is_closing(self, current: float, target: float, velocity: float) -> bool:
        """True if the current velocity is moving toward the target."""
        if velocity == 0:
            return False
        if self.higher_is_better:
            # current < target → need to go up → positive velocity = closing
            if current < target:
                return velocity > 0
            # current > target → already above → any direction is "closing" or stable
            elif current > target:
                return velocity < 0
        else:
            # Lower is better
            if current > target:
                return velocity < 0
            elif current < target:
                return velocity > 0
        return False

    def _build_story(
        self,
        metric_key: str,
        current: float,
        target: float,
        baseline: PersonalBaseline | None,
        trend: TrendAnalysis | None,
        gap_size: float,
        gap_direction: str,
        closing: bool,
        estimated_close: date | None,
    ) -> str:
        """Compose a human-readable narrative about this gap."""
        name = _metric_display_name(metric_key)
        parts: list[str] = []

        # Baseline context
        if baseline is not None:
            parts.append(
                f"You averaged {_format_value(metric_key, baseline.mean)} "
                f"over the past {baseline.window_days} days."
            )

        # Current vs target
        if gap_direction == "at_target":
            parts.append(f"Your current {name} is right at your target.")
        else:
            direction_word = gap_direction  # "above" or "below"
            parts.append(
                f"Currently at {_format_value(metric_key, current)} — "
                f"{_format_value(metric_key, gap_size)} {direction_word} "
                f"your target of {_format_value(metric_key, target)}."
            )

        # Trend context
        if trend is not None and trend.direction != TrendDirection.INSUFFICIENT_DATA:
            if trend.trend_started:
                started_str = trend.trend_started.strftime("%b %d")
                parts.append(f"The trend started {started_str}.")

            if trend.direction == TrendDirection.DECLINING:
                if closing:
                    parts.append("It's declining, but moving toward your target.")
                else:
                    parts.append("It's still declining.")
            elif trend.direction == TrendDirection.IMPROVING:
                if closing:
                    parts.append("It's improving and closing the gap.")
                else:
                    parts.append("It's improving, though moving away from your target.")
            elif trend.direction == TrendDirection.STABLE:
                parts.append("It's been stable recently.")

        # Estimated close
        if estimated_close is not None:
            parts.append(
                f"At the current rate, you'd reach your target around {estimated_close.strftime('%b %d')}."
            )

        return " ".join(parts)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _metric_display_name(key: str) -> str:
    """Turn a metric key like 'sleep.duration' into 'sleep duration'."""
    return key.replace(".", " ").replace("_", " ")


def _format_value(metric_key: str, value: float) -> str:
    """Format a numeric value with context-appropriate units.

    This is intentionally simple — a future version may use MetricDefinition
    to look up units. For now, round to one decimal.
    """
    if abs(value) >= 100:
        return f"{value:,.0f}"
    return f"{value:.1f}"
