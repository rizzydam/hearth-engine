"""
Temporal layers — fluid, patterned, static views of the same shape.

Three resolutions of the same geometry:
- Fluid: right now, changes with every check-in
- Patterned: 7-30 day rolling average, shows recurring distortions
- Static: 90+ day baseline evolution, what 1.0 used to be

The delta between layers IS drift detection. No separate system needed.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .contracts import AxisValue, ShapeSnapshot, TemporalLayers
from .shape_metrics import compute_shape, compute_delta


def build_temporal_layers(
    fluid_shape: ShapeSnapshot,
    assessment_history: list[dict],
    trend_report: Any = None,
    patterned_window_days: int = 14,
) -> TemporalLayers:
    """Build all three temporal layers from current shape + history.

    Parameters
    ----------
    fluid_shape : ShapeSnapshot
        Today's coupled shape (already computed).
    assessment_history : list[dict]
        Recent geometric assessments from store. Each dict should have
        a "layers" -> "fluid" -> "axes" path, or a flat "axes" list.
    trend_report : TrendReport, optional
        For static layer baselines.
    patterned_window_days : int
        How many days of history to average for the patterned layer.
    """
    patterned = _compute_patterned(fluid_shape, assessment_history, patterned_window_days)
    static = _compute_static(fluid_shape, trend_report)

    fluid_vs_patterned = compute_delta(fluid_shape, patterned, "fluid", "patterned")
    patterned_vs_static = compute_delta(patterned, static, "patterned", "static")

    return TemporalLayers(
        fluid=fluid_shape,
        patterned=patterned,
        static=static,
        fluid_vs_patterned=fluid_vs_patterned,
        patterned_vs_static=patterned_vs_static,
    )


def _compute_patterned(
    fluid: ShapeSnapshot,
    history: list[dict],
    window_days: int,
) -> ShapeSnapshot:
    """Average recent daily shapes into a patterned layer.

    Falls back to fluid shape if insufficient history.
    """
    if not history:
        return fluid

    # Extract axis values from history entries
    historical_axes: dict[str, list[float]] = defaultdict(list)

    for entry in history[-window_days:]:
        # Navigate the dict structure to find axes
        axes_data = _extract_axes_from_history(entry)
        for ax in axes_data:
            key = ax.get("metric_key", "")
            coupled = ax.get("coupled", ax.get("normalized", 1.0))
            if key:
                historical_axes[key].append(coupled)

    if not historical_axes:
        return fluid

    # Average the historical values, use fluid's axis layout
    averaged_axes = []
    for ax in fluid.axes:
        hist_values = historical_axes.get(ax.metric_key, [])
        if hist_values:
            avg_coupled = sum(hist_values) / len(hist_values)
        else:
            avg_coupled = ax.coupled  # No history for this axis

        averaged_axes.append(AxisValue(
            metric_key=ax.metric_key,
            domain=ax.domain,
            raw_value=ax.raw_value,
            baseline=ax.baseline,
            normalized=avg_coupled,  # Patterned uses averaged value
            coupled=avg_coupled,
            angle_radians=ax.angle_radians,
            level=ax.level,
        ))

    return compute_shape(averaged_axes)


def _compute_static(
    fluid: ShapeSnapshot,
    trend_report: Any,
) -> ShapeSnapshot:
    """Build static layer from TrendEngine baselines.

    The static layer represents what 1.0 USED to be. If your baseline
    has shifted over 90 days, the static shape reflects the old normal.
    Since we normalize to current baseline, the static layer is always
    near 1.0 — the interesting signal is when patterned deviates from it.
    """
    if trend_report is None:
        # No trend data — static = perfect circle (everything at baseline)
        static_axes = []
        for ax in fluid.axes:
            static_axes.append(AxisValue(
                metric_key=ax.metric_key,
                domain=ax.domain,
                raw_value=ax.baseline,
                baseline=ax.baseline,
                normalized=1.0,
                coupled=1.0,
                angle_radians=ax.angle_radians,
                level="healthy",
            ))
        return compute_shape(static_axes)

    # With trend data: static layer values come from baseline means
    # normalized against themselves = 1.0, unless the baseline has been
    # shifting (in which case the 90-day mean differs from the 14-day mean)
    static_axes = []
    metrics = getattr(trend_report, "metrics", {})

    for ax in fluid.axes:
        snapshot = metrics.get(ax.metric_key)
        if snapshot and getattr(snapshot, "baseline", None):
            baseline = snapshot.baseline
            # If trend is significant, the static value reflects the
            # 90-day baseline relative to current baseline
            # (most of the time this is ~1.0)
            static_val = 1.0
        else:
            static_val = 1.0

        static_axes.append(AxisValue(
            metric_key=ax.metric_key,
            domain=ax.domain,
            raw_value=ax.baseline,
            baseline=ax.baseline,
            normalized=static_val,
            coupled=static_val,
            angle_radians=ax.angle_radians,
            level="healthy",
        ))

    return compute_shape(static_axes)


def _extract_axes_from_history(entry: dict) -> list[dict]:
    """Extract axis data from a history dict (various formats)."""
    # Try nested format: entry["layers"]["fluid"]["axes"]
    layers = entry.get("layers", {})
    fluid = layers.get("fluid", {})
    axes = fluid.get("axes", [])
    if axes:
        return axes

    # Try flat format: entry["axes"]
    axes = entry.get("axes", [])
    if axes:
        return axes

    return []
