"""
Multi-model synapse architecture — objective substrate vs hypothesis ensemble.

The WeightLens pattern: one SynapseNetwork (the objective truth), multiple
read-only lenses that view it through different weight configurations.

Each hypothesis is a testable claim that produces a specific weight lens.
The ensemble blends all validated hypotheses proportional to their
earned confidence.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any

from .contracts import Synapse, _now_iso, _new_id

log = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────
# Hypothesis — a testable claim about how metrics relate
# ───────────────────────────────────────────────────────────────────────

@dataclass
class Hypothesis:
    hypothesis_id: str
    claim: str                              # "Stress above 12 increases spending within 1-3 days"
    origin: str                             # "narrative" | "research" | "observed" | "ai_discovered"
    source_detail: str = ""                 # e.g., "research_paper.pdf / Section 3"

    # What this hypothesis predicts
    metric_a: str = ""
    metric_b: str = ""
    predicted_direction: str = "negative"   # "positive" | "negative"
    predicted_weight: float = 0.5
    condition: str | None = None            # "metric_a > 12"
    activation_function: str = "linear"
    activation_params: dict[str, float] = field(default_factory=dict)

    # Weight lens: how this hypothesis modifies the synapse
    weight_modifier: float = 1.0            # >1.0 = this hypothesis says stronger
    threshold_override: float | None = None

    # Track record (earned from data)
    predictions_made: int = 0
    predictions_correct: int = 0
    predictions_wrong: int = 0
    confidence: float = 0.0
    status: str = "untested"                # "untested"|"testing"|"validated"|"falsified"|"conditional"

    created_at: str = ""
    last_tested: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = _now_iso()

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "claim": self.claim,
            "origin": self.origin,
            "source_detail": self.source_detail,
            "metric_a": self.metric_a,
            "metric_b": self.metric_b,
            "predicted_direction": self.predicted_direction,
            "predicted_weight": self.predicted_weight,
            "weight_modifier": self.weight_modifier,
            "activation_function": self.activation_function,
            "activation_params": self.activation_params,
            "predictions_made": self.predictions_made,
            "predictions_correct": self.predictions_correct,
            "predictions_wrong": self.predictions_wrong,
            "confidence": round(self.confidence, 4),
            "status": self.status,
            "created_at": self.created_at,
            "last_tested": self.last_tested,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Hypothesis:
        return cls(**{k: d[k] for k in d if k in cls.__dataclass_fields__})


# ───────────────────────────────────────────────────────────────────────
# DeltaEntry — an actionable disagreement between hypothesis and data
# ───────────────────────────────────────────────────────────────────────

@dataclass
class DeltaEntry:
    delta_id: str = ""
    hypothesis_id: str = ""
    delta_type: str = "untested"            # "blind_spot"|"growth"|"evolution"|"confirmed"|"untested"
    metric_key: str = ""
    domain: str = ""
    hypothesis_says: str = ""
    data_shows: str = ""
    magnitude: float = 0.0
    action_type: str = ""                   # "goal"|"probing_trigger"|"data_request"
    action_detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "delta_id": self.delta_id,
            "hypothesis_id": self.hypothesis_id,
            "delta_type": self.delta_type,
            "domain": self.domain,
            "hypothesis_says": self.hypothesis_says,
            "data_shows": self.data_shows,
            "magnitude": round(self.magnitude, 4),
            "action_type": self.action_type,
            "action_detail": self.action_detail,
        }


# ───────────────────────────────────────────────────────────────────────
# WeightLens — read-only view of network with modified weights
# ───────────────────────────────────────────────────────────────────────

class ObjectiveLens:
    """Returns network weights as-is. Filters narrative-only synapses."""

    def __init__(self, network):
        self._network = network

    def all(self):
        return [s for s in self._network.all() if s.source != "narrative_inferred"]

    def count(self):
        return len(self.all())

    def get_outgoing(self, metric_key):
        return [s for s in self._network.get_outgoing(metric_key)
                if s.source != "narrative_inferred"]

    def get_incoming(self, metric_key):
        return [s for s in self._network.get_incoming(metric_key)
                if s.source != "narrative_inferred"]

    def get_connected(self, metric_key):
        return [s for s in self._network.get_connected(metric_key)
                if s.source != "narrative_inferred"]

    def all_metrics(self):
        return self._network.all_metrics()

    def find_existing(self, a, b):
        return self._network.find_existing(a, b)


class EnsembleLens:
    """Applies confidence-weighted hypothesis modifiers to network weights.

    Each validated hypothesis contributes its weight_modifier proportional
    to its earned confidence. The ensemble is the sum of all evidence.
    """

    def __init__(self, network, hypotheses: list[Hypothesis], data_age_days: int = 0):
        self._network = network
        self._hypotheses = hypotheses
        self._data_age = data_age_days
        self._modifier_cache: dict[str, list[tuple[float, float]]] = {}
        self._build_cache()

    def _build_cache(self):
        """Pre-compute per-synapse modifier sets from active hypotheses."""
        for h in self._hypotheses:
            if h.status in ("falsified",):
                continue
            if h.weight_modifier == 1.0 and h.activation_function == "linear":
                continue  # No effect

            syn = self._network.find_existing(h.metric_a, h.metric_b)
            if syn is None:
                continue

            self._modifier_cache.setdefault(syn.synapse_id, []).append(
                (h.weight_modifier, h.confidence)
            )

    def _effective_weight(self, syn: Synapse) -> float:
        """Compute ensemble-blended weight for a synapse."""
        modifiers = self._modifier_cache.get(syn.synapse_id, [])
        if not modifiers:
            return syn.weight

        # Weighted average of all hypothesis modifiers
        total_confidence = sum(conf for _, conf in modifiers)
        if total_confidence <= 0:
            return syn.weight

        weighted_modifier = sum(mod * conf for mod, conf in modifiers) / total_confidence

        # Blend between objective weight and hypothesis-modified weight
        # based on data maturity
        alpha = min(1.0, 0.3 + 0.7 * min(1.0, self._data_age / 90.0))
        blended = alpha * syn.weight + (1 - alpha) * (syn.weight * weighted_modifier)

        return max(0.0, min(1.0, blended))

    def _apply_hypothesis_metadata(self, syn: Synapse) -> Synapse:
        """Apply non-linear activation from hypothesis to synapse metadata."""
        for h in self._hypotheses:
            if h.status == "falsified":
                continue
            if h.metric_a == syn.metric_a and h.metric_b == syn.metric_b:
                if h.activation_function != "linear":
                    modified = copy.copy(syn)
                    modified.weight = self._effective_weight(syn)
                    modified.metadata = dict(syn.metadata) if syn.metadata else {}
                    modified.metadata["activation_function"] = h.activation_function
                    modified.metadata["activation_params"] = h.activation_params
                    return modified
        # No non-linear hypothesis — just modify weight
        modified = copy.copy(syn)
        modified.weight = self._effective_weight(syn)
        return modified

    def all(self):
        return [self._apply_hypothesis_metadata(s) for s in self._network.all()]

    def count(self):
        return self._network.count()

    def get_outgoing(self, metric_key):
        return [self._apply_hypothesis_metadata(s) for s in self._network.get_outgoing(metric_key)]

    def get_incoming(self, metric_key):
        return [self._apply_hypothesis_metadata(s) for s in self._network.get_incoming(metric_key)]

    def get_connected(self, metric_key):
        return [self._apply_hypothesis_metadata(s) for s in self._network.get_connected(metric_key)]

    def all_metrics(self):
        return self._network.all_metrics()

    def find_existing(self, a, b):
        return self._network.find_existing(a, b)


# ───────────────────────────────────────────────────────────────────────
# HypothesisRegistry — stores and manages all hypotheses
# ───────────────────────────────────────────────────────────────────────

class HypothesisRegistry:
    """Stores, queries, and persists hypotheses."""

    def __init__(self, store):
        self._store = store
        self._hypotheses: dict[str, Hypothesis] = {}
        self._load()

    def _load(self):
        records = self._store._store.load("hypotheses")
        for r in records:
            try:
                h = Hypothesis.from_dict(r)
                self._hypotheses[h.hypothesis_id] = h
            except Exception:
                pass

    def save(self):
        self._store._store.save("hypotheses", [h.to_dict() for h in self._hypotheses.values()])

    def add(self, h: Hypothesis):
        self._hypotheses[h.hypothesis_id] = h

    def get(self, hypothesis_id: str) -> Hypothesis | None:
        return self._hypotheses.get(hypothesis_id)

    def all(self) -> list[Hypothesis]:
        return list(self._hypotheses.values())

    def by_status(self, status: str) -> list[Hypothesis]:
        return [h for h in self._hypotheses.values() if h.status == status]

    def by_metric(self, metric_key: str) -> list[Hypothesis]:
        return [h for h in self._hypotheses.values()
                if h.metric_a == metric_key or h.metric_b == metric_key]

    def active(self) -> list[Hypothesis]:
        """All hypotheses that should participate in the ensemble."""
        return [h for h in self._hypotheses.values()
                if h.status not in ("falsified",)]

    def count(self) -> int:
        return len(self._hypotheses)

    def stats(self) -> dict[str, int]:
        by_status = {}
        for h in self._hypotheses.values():
            by_status[h.status] = by_status.get(h.status, 0) + 1
        return by_status
