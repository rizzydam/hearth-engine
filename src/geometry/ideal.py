"""
Ideal shape — where the person wants to be.

Built from three sources:
1. Thriving thresholds (catalog) — what research says is optimal
2. Active goals (GoalStore) — what the user chose to pursue
3. User priorities (bus) — what the user emphasizes

The delta between actual and ideal IS the gap analysis.
"""

from __future__ import annotations

from typing import Any

from .contracts import AxisValue, IdealShape
from .axis_layout import get_axis_angles, get_domain_for_metric
from .normalizer import _get_baseline_mean


def compute_ideal(
    current_axes: list[AxisValue],
    goals: list[dict] | None = None,
    room_priorities: dict[str, float] | None = None,
) -> IdealShape:
    """Compute the ideal shape from catalog thresholds + goals + priorities.

    Parameters
    ----------
    current_axes : list[AxisValue]
        The current axis layout (provides baseline and angle for each metric).
    goals : list[dict], optional
        Active goals with "domain" field.
    room_priorities : dict[str, float], optional
        Domain priority weights from Room (0-1 scale).
    """
    try:
        from ..metric_catalog.catalog import ALL_METRICS, ALL_SENSES
    except ImportError:
        return IdealShape(axes=current_axes)

    angles = get_axis_angles()
    goal_domains = set()
    if goals:
        for g in goals:
            domain = g.get("domain", "")
            if domain:
                goal_domains.add(domain)

    priorities = room_priorities or {}
    ideal_axes = []

    for ax in current_axes:
        spec = ALL_METRICS.get(ax.metric_key)
        if spec is None:
            ideal_axes.append(AxisValue(
                metric_key=ax.metric_key,
                domain=ax.domain,
                raw_value=ax.baseline,
                baseline=ax.baseline,
                normalized=1.0,
                coupled=1.0,
                angle_radians=ax.angle_radians,
                level="healthy",
            ))
            continue

        # Start with thriving threshold as the ideal
        ideal_normalized = 1.0
        if spec.thriving_threshold is not None and ax.baseline > 0:
            if spec.higher_is_better:
                ideal_normalized = spec.thriving_threshold / ax.baseline
            else:
                ideal_normalized = ax.baseline / max(spec.thriving_threshold, 0.001)
            ideal_normalized = max(0.5, min(2.5, ideal_normalized))

        # Boost domains with active goals (+10%)
        if ax.domain in goal_domains:
            ideal_normalized *= 1.10

        # Weight by room priorities
        priority = priorities.get(ax.domain, 0.5)
        # Higher priority → ideal pushed further from 1.0 (more ambitious)
        if ideal_normalized > 1.0:
            ideal_normalized = 1.0 + (ideal_normalized - 1.0) * (0.5 + priority)

        ideal_axes.append(AxisValue(
            metric_key=ax.metric_key,
            domain=ax.domain,
            raw_value=spec.thriving_threshold if spec.thriving_threshold else ax.baseline,
            baseline=ax.baseline,
            normalized=ideal_normalized,
            coupled=ideal_normalized,  # Ideal shape is uncoupled (target, not physics)
            angle_radians=ax.angle_radians,
            level="thriving",
        ))

    # Compute priority weights per domain
    priority_weights = {}
    for ax in ideal_axes:
        if ax.domain not in priority_weights:
            base_priority = priorities.get(ax.domain, 0.5)
            if ax.domain in goal_domains:
                base_priority = min(1.0, base_priority + 0.2)
            priority_weights[ax.domain] = round(base_priority, 3)

    return IdealShape(axes=ideal_axes, priority_weights=priority_weights)
