"""
Geometry data model — the shapes that describe a person's state.

No logic. Just structures. The geometry module computes these;
every downstream consumer reads them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ───────────────────────────────────────────────────────────────────────
# AxisValue — one radial axis in the shape
# ───────────────────────────────────────────────────────────────────────

@dataclass
class AxisValue:
    metric_key: str             # "fitness.sleep_hours"
    domain: str                 # "fitness"
    raw_value: float            # The original metric value
    baseline: float             # Computed baseline (mean or healthy midpoint)
    normalized: float           # raw / baseline, centered at 1.0
    coupled: float              # After force propagation through synapses
    angle_radians: float        # Position on the radial chart
    level: str = "healthy"      # "critical" | "warning" | "healthy" | "thriving"

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_key": self.metric_key,
            "domain": self.domain,
            "raw_value": round(self.raw_value, 4),
            "baseline": round(self.baseline, 4),
            "normalized": round(self.normalized, 4),
            "coupled": round(self.coupled, 4),
            "angle_radians": round(self.angle_radians, 4),
            "level": self.level,
        }


# ───────────────────────────────────────────────────────────────────────
# ShapeSnapshot — a complete shape at one temporal resolution
# ───────────────────────────────────────────────────────────────────────

@dataclass
class ShapeSnapshot:
    timestamp: str = ""
    axes: list[AxisValue] = field(default_factory=list)
    area_ratio: float = 1.0         # area(actual) / area(unit_circle)
    circularity_index: float = 1.0  # 0.0 (lopsided) to 1.0 (perfect circle)
    dominant_distortion: str = ""   # Domain with most deviation
    dominant_direction: str = ""    # "inward" or "outward"
    per_domain_means: dict[str, float] = field(default_factory=dict)
    per_domain_balance: dict[str, float] = field(default_factory=dict)
    vertex_coordinates: list[tuple[float, float]] = field(default_factory=list)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = _now_iso()

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "area_ratio": round(self.area_ratio, 4),
            "circularity_index": round(self.circularity_index, 4),
            "dominant_distortion": self.dominant_distortion,
            "dominant_direction": self.dominant_direction,
            "per_domain_means": {k: round(v, 4) for k, v in self.per_domain_means.items()},
            "per_domain_balance": {k: round(v, 4) for k, v in self.per_domain_balance.items()},
            "axes": [a.to_dict() for a in self.axes],
            "vertex_coordinates": [
                (round(x, 4), round(y, 4)) for x, y in self.vertex_coordinates
            ],
        }


# ───────────────────────────────────────────────────────────────────────
# ShapeDelta — difference between two shapes
# ───────────────────────────────────────────────────────────────────────

@dataclass
class ShapeDelta:
    layer_a: str = ""               # "fluid" | "patterned" | "static" | "ideal"
    layer_b: str = ""
    area_delta: float = 0.0         # area_ratio_a - area_ratio_b
    circularity_delta: float = 0.0
    per_axis_delta: dict[str, float] = field(default_factory=dict)
    shifting_domains: list[str] = field(default_factory=list)
    interpretation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_a": self.layer_a,
            "layer_b": self.layer_b,
            "area_delta": round(self.area_delta, 4),
            "circularity_delta": round(self.circularity_delta, 4),
            "per_axis_delta": {k: round(v, 4) for k, v in self.per_axis_delta.items()},
            "shifting_domains": self.shifting_domains,
            "interpretation": self.interpretation,
        }


# ───────────────────────────────────────────────────────────────────────
# TemporalLayers — three views of the same shape
# ───────────────────────────────────────────────────────────────────────

@dataclass
class TemporalLayers:
    fluid: ShapeSnapshot = field(default_factory=ShapeSnapshot)
    patterned: ShapeSnapshot = field(default_factory=ShapeSnapshot)
    static: ShapeSnapshot = field(default_factory=ShapeSnapshot)
    fluid_vs_patterned: ShapeDelta = field(default_factory=ShapeDelta)
    patterned_vs_static: ShapeDelta = field(default_factory=ShapeDelta)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fluid": self.fluid.to_dict(),
            "patterned": self.patterned.to_dict(),
            "static": self.static.to_dict(),
            "fluid_vs_patterned": self.fluid_vs_patterned.to_dict(),
            "patterned_vs_static": self.patterned_vs_static.to_dict(),
        }


# ───────────────────────────────────────────────────────────────────────
# IdealShape — where the person wants to be
# ───────────────────────────────────────────────────────────────────────

@dataclass
class IdealShape:
    axes: list[AxisValue] = field(default_factory=list)
    priority_weights: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "axes": [a.to_dict() for a in self.axes],
            "priority_weights": {k: round(v, 4) for k, v in self.priority_weights.items()},
        }


# ───────────────────────────────────────────────────────────────────────
# GeometricAssessment — the complete output
# ───────────────────────────────────────────────────────────────────────

@dataclass
class GeometricAssessment:
    timestamp: str = ""
    layers: TemporalLayers = field(default_factory=TemporalLayers)
    ideal: IdealShape = field(default_factory=IdealShape)
    delta_to_ideal: ShapeDelta = field(default_factory=ShapeDelta)

    # Derived from shape
    mode: str = "stable"
    capacity: float = 1.0
    overall_health: float = 1.0         # area_ratio of fluid layer
    overall_balance: float = 1.0        # circularity_index of fluid layer

    # Preserved from synapse assessment
    hard_rule_fires: list = field(default_factory=list)
    thriving_metrics: list[dict] = field(default_factory=list)
    active_streaks: list[dict] = field(default_factory=list)
    momentum_signals: list[dict] = field(default_factory=list)
    cascade_predictions: list = field(default_factory=list)
    reinforcement_cascades: list = field(default_factory=list)
    coverage: Any = None
    resilience: list = field(default_factory=list)
    recovery: Any = None
    data_requests: list[dict] = field(default_factory=list)

    # Hypothesis/delta data
    hypothesis_stats: dict = field(default_factory=dict)
    deltas: list = field(default_factory=list)

    # Tracing
    activated_synapses: list = field(default_factory=list)
    propagation_iterations: int = 0
    convergence_residual: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = _now_iso()

    def to_dict(self) -> dict[str, Any]:
        from ..synapse.contracts import CoverageReport, RecoveryState
        return {
            "timestamp": self.timestamp,
            "mode": self.mode,
            "capacity": round(self.capacity, 3),
            "overall_health": round(self.overall_health, 4),
            "overall_balance": round(self.overall_balance, 4),
            "layers": self.layers.to_dict(),
            "ideal": self.ideal.to_dict(),
            "delta_to_ideal": self.delta_to_ideal.to_dict(),
            "hard_rule_fires": [
                {"rule_id": h.rule_id, "severity": h.severity, "message": h.message}
                if hasattr(h, "rule_id") else h
                for h in self.hard_rule_fires
            ],
            "thriving_metrics": self.thriving_metrics,
            "active_streaks": self.active_streaks,
            "momentum_signals": self.momentum_signals,
            "propagation_iterations": self.propagation_iterations,
            "convergence_residual": round(self.convergence_residual, 6),
            "coverage": self.coverage.to_dict() if hasattr(self.coverage, "to_dict") else None,
            "resilience": [
                {"domain": r.domain, "net": r.net_resilience}
                if hasattr(r, "domain") else r
                for r in self.resilience
            ],
            "data_requests": self.data_requests[:5],
        }
