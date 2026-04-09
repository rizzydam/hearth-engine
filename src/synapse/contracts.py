"""
Synapse data model — every structure the synapse layer operates on.

No logic here. Just shapes. Logic lives in the engines that consume them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import uuid


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str = "syn") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# ───────────────────────────────────────────────────────────────────────
# Synapse — a weighted connection between two metrics
# ───────────────────────────────────────────────────────────────────────

@dataclass
class Synapse:
    synapse_id: str
    metric_a: str                            # source metric key
    metric_b: str                            # target metric key
    weight: float                            # 0.0-1.0 connection strength
    direction: str                           # "positive" | "negative" | "complex"
    relationship: str = "threatens"          # "threatens" | "protects" | "reinforces"
    lag_window: tuple[int, int] = (0, 7)    # (min_days, max_days) B follows A
    threshold_a: float | None = None         # activate only when metric_a exceeds
    threshold_b: float | None = None         # activate only when metric_b exceeds
    source: str = "research_prior"           # "research_prior" | "observed" | "ai_discovered"
    research_basis: str | None = None
    observations: int = 0                    # data points supporting this connection
    contradictions: int = 0                  # data points contradicting this connection
    confidence: float = 0.5
    user_confirmed: bool = False
    last_reinforced: str = ""
    last_fired: str = ""
    created_at: str = ""
    decay_rate: float = 0.005               # weight reduction per day without reinforcement
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = _now_iso()
        if not self.last_reinforced:
            self.last_reinforced = self.created_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "synapse_id": self.synapse_id,
            "metric_a": self.metric_a,
            "metric_b": self.metric_b,
            "weight": self.weight,
            "direction": self.direction,
            "relationship": self.relationship,
            "lag_window": list(self.lag_window),
            "threshold_a": self.threshold_a,
            "threshold_b": self.threshold_b,
            "source": self.source,
            "research_basis": self.research_basis,
            "observations": self.observations,
            "contradictions": self.contradictions,
            "confidence": self.confidence,
            "user_confirmed": self.user_confirmed,
            "last_reinforced": self.last_reinforced,
            "last_fired": self.last_fired,
            "created_at": self.created_at,
            "decay_rate": self.decay_rate,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Synapse:
        lw = d.get("lag_window", [0, 7])
        return cls(
            synapse_id=d["synapse_id"],
            metric_a=d["metric_a"],
            metric_b=d["metric_b"],
            weight=d.get("weight", 0.5),
            direction=d.get("direction", "complex"),
            relationship=d.get("relationship", "threatens"),
            lag_window=tuple(lw) if isinstance(lw, (list, tuple)) else (0, 7),
            threshold_a=d.get("threshold_a"),
            threshold_b=d.get("threshold_b"),
            source=d.get("source", "research_prior"),
            research_basis=d.get("research_basis"),
            observations=d.get("observations", 0),
            contradictions=d.get("contradictions", 0),
            confidence=d.get("confidence", 0.5),
            user_confirmed=d.get("user_confirmed", False),
            last_reinforced=d.get("last_reinforced", ""),
            last_fired=d.get("last_fired", ""),
            created_at=d.get("created_at", ""),
            decay_rate=d.get("decay_rate", 0.005),
            metadata=d.get("metadata", {}),
        )


# ───────────────────────────────────────────────────────────────────────
# MetricObservation — a timestamped value from a sense
# ───────────────────────────────────────────────────────────────────────

@dataclass
class MetricObservation:
    metric_key: str
    value: float
    timestamp: str = ""
    source_sense: str = ""
    data_quality: float = 0.6

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = _now_iso()

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_key": self.metric_key,
            "value": self.value,
            "timestamp": self.timestamp,
            "source_sense": self.source_sense,
            "data_quality": self.data_quality,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MetricObservation:
        return cls(
            metric_key=d["metric_key"],
            value=d["value"],
            timestamp=d.get("timestamp", ""),
            source_sense=d.get("source_sense", ""),
            data_quality=d.get("data_quality", 0.6),
        )


# ───────────────────────────────────────────────────────────────────────
# DataExpectation — what we expect from a metric and what's missing
# ───────────────────────────────────────────────────────────────────────

@dataclass
class DataExpectation:
    metric_key: str
    expected_cadence_hours: float = 24.0
    last_seen: str | None = None
    gap_count: int = 0
    gap_z_score: float = 0.0
    is_dropout: bool = False
    total_observations: int = 0
    avg_gap_hours: float = 24.0
    gap_stddev_hours: float = 12.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_key": self.metric_key,
            "expected_cadence_hours": self.expected_cadence_hours,
            "last_seen": self.last_seen,
            "gap_count": self.gap_count,
            "gap_z_score": self.gap_z_score,
            "is_dropout": self.is_dropout,
            "total_observations": self.total_observations,
            "avg_gap_hours": self.avg_gap_hours,
            "gap_stddev_hours": self.gap_stddev_hours,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DataExpectation:
        return cls(**{k: d[k] for k in d if k in cls.__dataclass_fields__})


# ───────────────────────────────────────────────────────────────────────
# CoverageReport — what we can see vs what we're blind to
# ───────────────────────────────────────────────────────────────────────

@dataclass
class CoverageReport:
    total_synapses: int = 0
    visible_synapses: int = 0
    partially_visible: int = 0
    blind_synapses: int = 0
    coverage_fraction: float = 0.0
    highest_value_blind: list[str] = field(default_factory=list)
    per_domain_coverage: dict[str, float] = field(default_factory=dict)


# ───────────────────────────────────────────────────────────────────────
# ResilienceVector — per-domain protective vs vulnerability balance
# ───────────────────────────────────────────────────────────────────────

@dataclass
class ResilienceVector:
    domain: str
    buffer_score: float = 0.0
    protective_factors: list[dict] = field(default_factory=list)
    vulnerability_factors: list[dict] = field(default_factory=list)
    net_resilience: float = 0.0
    scenario_answers: dict[str, float] = field(default_factory=dict)


# ───────────────────────────────────────────────────────────────────────
# RecoveryState — where someone is in the recovery sequence
# ───────────────────────────────────────────────────────────────────────

RECOVERY_STAGES = ("sleep", "activity", "social", "work", "financial", "identity")

@dataclass
class RecoveryState:
    current_stage: str = "sleep"
    stage_index: int = 0
    mode: str = "stable"
    trajectory: str = "stable"               # "deteriorating" | "stabilizing" | "recovering" | "stable"
    stage_progress: float = 0.0
    regression_count: int = 0
    time_in_stage_days: int = 0
    intervention_matches: list[dict] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_stage": self.current_stage,
            "stage_index": self.stage_index,
            "mode": self.mode,
            "trajectory": self.trajectory,
            "stage_progress": self.stage_progress,
            "regression_count": self.regression_count,
            "time_in_stage_days": self.time_in_stage_days,
            "intervention_matches": self.intervention_matches,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RecoveryState:
        return cls(**{k: d[k] for k in d if k in cls.__dataclass_fields__})


# ───────────────────────────────────────────────────────────────────────
# HardRule — non-negotiable clinical/financial safety thresholds
# ───────────────────────────────────────────────────────────────────────

@dataclass
class HardRule:
    rule_id: str
    metric_key: str
    threshold: float
    comparison: str = "lt"                   # "lt" | "gt" | "eq"
    consecutive_days: int = 1
    severity: str = "warning"                # "warning" | "critical"
    message: str = ""
    clinical_basis: str = ""
    overrides_synapse: bool = True

    def check(self, value: float) -> bool:
        """Does the current value violate this rule?"""
        if self.comparison == "lt":
            return value < self.threshold
        elif self.comparison == "gt":
            return value > self.threshold
        elif self.comparison == "eq":
            return value == self.threshold
        return False


# ───────────────────────────────────────────────────────────────────────
# Activated synapse — a synapse that fired during assessment
# ───────────────────────────────────────────────────────────────────────

@dataclass
class ActivatedSynapse:
    synapse_id: str
    activation_level: float
    source_metric_value: float
    target_metric: str = ""
    relationship: str = ""


# ───────────────────────────────────────────────────────────────────────
# CascadePrediction — a predicted chain of effects
# ───────────────────────────────────────────────────────────────────────

@dataclass
class CascadePrediction:
    chain: list[str] = field(default_factory=list)
    probability: float = 0.0
    time_window_days: int = 0
    severity: float = 0.0
    relationship: str = "threatens"          # cascade type


# ───────────────────────────────────────────────────────────────────────
# HardRuleFire — a hard rule that was violated
# ───────────────────────────────────────────────────────────────────────

@dataclass
class HardRuleFire:
    rule_id: str
    severity: str
    message: str
    metric_key: str = ""
    current_value: float = 0.0


# ───────────────────────────────────────────────────────────────────────
# SynapseAssessment — the full output of one assessment cycle
# ───────────────────────────────────────────────────────────────────────

@dataclass
class SynapseAssessment:
    timestamp: str = ""
    mode: str = "stable"
    trajectory: str = "stable"

    # What fired
    activated_synapses: list[ActivatedSynapse] = field(default_factory=list)
    cascade_predictions: list[CascadePrediction] = field(default_factory=list)
    hard_rule_fires: list[HardRuleFire] = field(default_factory=list)

    # Positive signals
    thriving_metrics: list[dict] = field(default_factory=list)   # [{key, level, message}]
    active_streaks: list[dict] = field(default_factory=list)     # [{key, days, message}]
    momentum_signals: list[dict] = field(default_factory=list)   # [{key, direction, message}]
    reinforcement_cascades: list[CascadePrediction] = field(default_factory=list)

    # Merged assessment
    domain_severities: dict[str, float] = field(default_factory=dict)
    domain_strengths: dict[str, float] = field(default_factory=dict)
    overall_severity: float = 0.0
    overall_vitality: float = 0.0            # 0.0-1.0, the positive counterpart

    # Context
    coverage: CoverageReport = field(default_factory=CoverageReport)
    resilience: list[ResilienceVector] = field(default_factory=list)
    recovery: RecoveryState = field(default_factory=RecoveryState)
    data_requests: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = _now_iso()

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "mode": self.mode,
            "trajectory": self.trajectory,
            "activated_synapses": [
                {"synapse_id": a.synapse_id, "activation_level": a.activation_level,
                 "source_metric_value": a.source_metric_value,
                 "target_metric": a.target_metric, "relationship": a.relationship}
                for a in self.activated_synapses
            ],
            "cascade_predictions": [
                {"chain": c.chain, "probability": c.probability,
                 "time_window_days": c.time_window_days, "severity": c.severity,
                 "relationship": c.relationship}
                for c in self.cascade_predictions
            ],
            "hard_rule_fires": [
                {"rule_id": h.rule_id, "severity": h.severity, "message": h.message,
                 "metric_key": h.metric_key, "current_value": h.current_value}
                for h in self.hard_rule_fires
            ],
            "thriving_metrics": self.thriving_metrics,
            "active_streaks": self.active_streaks,
            "momentum_signals": self.momentum_signals,
            "reinforcement_cascades": [
                {"chain": c.chain, "probability": c.probability,
                 "time_window_days": c.time_window_days, "severity": c.severity,
                 "relationship": c.relationship}
                for c in self.reinforcement_cascades
            ],
            "domain_severities": self.domain_severities,
            "domain_strengths": self.domain_strengths,
            "overall_severity": self.overall_severity,
            "overall_vitality": self.overall_vitality,
            "coverage": {
                "total": self.coverage.total_synapses,
                "visible": self.coverage.visible_synapses,
                "blind": self.coverage.blind_synapses,
                "fraction": self.coverage.coverage_fraction,
                "per_domain": self.coverage.per_domain_coverage,
            },
            "resilience": [
                {"domain": r.domain, "buffer": r.buffer_score,
                 "net": r.net_resilience, "protective": len(r.protective_factors),
                 "vulnerable": len(r.vulnerability_factors)}
                for r in self.resilience
            ],
            "recovery": self.recovery.to_dict(),
            "data_requests": self.data_requests,
        }
