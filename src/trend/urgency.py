"""
Trajectory urgency scorer — turns trend data into action priority.

The core insight: urgency comes from trajectory, not current value.

    "Bad but improving" = LOW urgency (it's already getting better)
    "Fine but deteriorating" = HIGH urgency (it's about to become a problem)

The score is a 0–1 composite that factors in:
    - Direction (declining > stable > improving)
    - Velocity (rapid > moderate > gradual)
    - Duration (longer trends are more urgent — not just a bad day)
    - Gap status (widening gap amplifies, closing gap dampens)
    - Staleness (stale data reduces confidence in urgency)
    - Compound pressure (multiple declining metrics amplify each other)
"""

from __future__ import annotations

from .contracts import (
    GapAnalysis,
    TrendAnalysis,
    TrendDirection,
    TrajectoryUrgency,
    UrgencyLevel,
)


# ---------------------------------------------------------------------------
# Weight tables
# ---------------------------------------------------------------------------

_DIRECTION_WEIGHTS: dict[TrendDirection, float] = {
    TrendDirection.DECLINING:           0.7,
    TrendDirection.STABLE:              0.3,
    TrendDirection.IMPROVING:           0.1,
    TrendDirection.INSUFFICIENT_DATA:   0.0,
}

_VELOCITY_WEIGHTS: dict[str, float] = {
    "rapid":    1.0,
    "moderate": 0.6,
    "gradual":  0.3,
    "flat":     0.1,
}


class TrajectoryScorer:
    """Score the urgency of a metric's trajectory."""

    def score(
        self,
        trend: TrendAnalysis | None,
        gap: GapAnalysis | None,
        staleness_freshness: float = 1.0,
        other_declining_count: int = 0,
    ) -> TrajectoryUrgency:
        """Compute a TrajectoryUrgency from trend, gap, and staleness data.

        Parameters
        ----------
        trend : TrendAnalysis, optional
            The current trend analysis.  If None, returns a "none" urgency.
        gap : GapAnalysis, optional
            Gap analysis (closing/widening status).
        staleness_freshness : float
            The data freshness score (0–1) from StalenessCalculator.
        other_declining_count : int
            Number of OTHER metrics currently declining.  Used for compound
            amplification.

        Returns
        -------
        TrajectoryUrgency
        """
        factors: list[str] = []

        # No trend data — nothing to score
        if trend is None or trend.direction == TrendDirection.INSUFFICIENT_DATA:
            return TrajectoryUrgency(
                level=UrgencyLevel.NONE,
                score=0.0,
                factors=["Insufficient data for trend analysis"],
                headline="Not enough data to assess trajectory",
                context="Need at least 5 data points to compute a meaningful trend.",
                gap=gap,
                trend=trend,
            )

        # --- Base score ---
        dir_w = _DIRECTION_WEIGHTS.get(trend.direction, 0.0)
        vel_w = _VELOCITY_WEIGHTS.get(trend.velocity_label, 0.1)
        dur_w = min(trend.duration_days / 30, 1.0)

        base = dir_w * vel_w * dur_w
        factors.append(
            f"Direction: {trend.direction.value} (weight {dir_w})"
        )
        factors.append(
            f"Velocity: {trend.velocity_label} (weight {vel_w})"
        )
        factors.append(
            f"Duration: {trend.duration_days} days (weight {dur_w:.2f})"
        )

        # --- Gap modifier ---
        if gap is not None:
            if gap.closing:
                base *= 0.3
                factors.append("Gap closing — urgency reduced")
            else:
                base *= 1.5
                factors.append("Gap widening — urgency amplified")

        # --- Staleness discount ---
        base *= staleness_freshness
        if staleness_freshness < 0.5:
            factors.append(
                f"Data staleness: freshness {staleness_freshness:.2f} — "
                f"urgency discounted"
            )

        # --- Compound amplifier ---
        if other_declining_count > 0:
            amplifier = 0.1 * other_declining_count
            base += amplifier
            factors.append(
                f"Compound pressure: {other_declining_count} other metrics "
                f"declining (+{amplifier:.2f})"
            )

        # Clamp to 0–1
        final_score = max(0.0, min(base, 1.0))

        # --- Level classification ---
        level = _classify_level(final_score)

        # --- Headline ---
        headline = _build_headline(trend, gap)

        # --- Context ---
        context = _build_context(trend, gap, factors, final_score, level)

        return TrajectoryUrgency(
            level=level,
            score=round(final_score, 4),
            factors=factors,
            headline=headline,
            context=context,
            gap=gap,
            trend=trend,
        )


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _classify_level(score: float) -> UrgencyLevel:
    """Map a 0–1 score to an UrgencyLevel."""
    if score > 0.75:
        return UrgencyLevel.URGENT
    if score > 0.5:
        return UrgencyLevel.ACT
    if score > 0.25:
        return UrgencyLevel.INVESTIGATE
    if score >= 0.1:
        return UrgencyLevel.WATCH
    return UrgencyLevel.NONE


# ---------------------------------------------------------------------------
# Narrative builders
# ---------------------------------------------------------------------------

def _build_headline(trend: TrendAnalysis, gap: GapAnalysis | None) -> str:
    """One-line urgency headline."""
    parts: list[str] = []

    # Direction + velocity
    dir_str = trend.direction.value.capitalize()
    if trend.velocity_label in ("rapid", "moderate"):
        parts.append(f"{dir_str} and {trend.velocity_label}")
    else:
        parts.append(dir_str)

    # Deviation from baseline
    if trend.deviation_from_baseline is not None and abs(trend.deviation_from_baseline) > 0.5:
        dev = trend.deviation_from_baseline
        direction = "above" if dev > 0 else "below"
        parts.append(f"{abs(dev):.1f} std dev {direction} normal")

    # Gap context
    if gap is not None and gap.gap_direction != "at_target":
        parts.append(
            f"{gap.gap_size:.1f} {gap.gap_direction} target"
        )

    return " — ".join(parts)


def _build_context(
    trend: TrendAnalysis,
    gap: GapAnalysis | None,
    factors: list[str],
    score: float,
    level: UrgencyLevel,
) -> str:
    """Full narrative context."""
    lines: list[str] = []

    lines.append(f"Urgency level: {level.value} (score: {score:.2f})")
    lines.append("")

    if gap and gap.story:
        lines.append(gap.story)
        lines.append("")

    lines.append("Factors:")
    for f in factors:
        lines.append(f"  - {f}")

    if trend.trend_started:
        lines.append(
            f"\nTrend started: {trend.trend_started.strftime('%Y-%m-%d')}"
        )
    lines.append(f"Data points: {trend.data_points}")
    lines.append(f"Confidence: {trend.confidence:.2f}")

    return "\n".join(lines)
