"""
Assessment engine — read the network, produce compound intelligence.

This is the main read path. Given current metric values, it determines:
- Which synapses are active (threats AND reinforcements)
- What cascades are predicted (downward AND upward spirals)
- Which hard rules fire
- What's thriving, what's streaking, what has momentum
- Per-domain severity AND strength
- Overall vitality alongside overall severity
"""

from __future__ import annotations

import logging
from typing import Any

from .contracts import (
    ActivatedSynapse, SynapseAssessment, CoverageReport,
)
from .network import SynapseNetwork, _domain_from_key
from .hard_rules import HardRuleEngine
from .information import InformationModel
from .resilience import ResilienceComputer
from .recovery import RecoveryStateMachine
from .cascade import CascadePredictor

log = logging.getLogger(__name__)


class AssessmentEngine:
    """The core intelligence read path."""

    def __init__(
        self,
        network: SynapseNetwork,
        hard_rules: HardRuleEngine,
        information: InformationModel,
        resilience: ResilienceComputer,
        recovery: RecoveryStateMachine,
        cascade: CascadePredictor,
    ):
        self._network = network
        self._hard_rules = hard_rules
        self._information = information
        self._resilience = resilience
        self._recovery = recovery
        self._cascade = cascade

    def assess(
        self,
        current_values: dict[str, float],
        trend_report: Any = None,
    ) -> SynapseAssessment:
        """Produce a full assessment from current metric values.

        This is the single call that produces everything downstream
        consumers need: the Desk, the Room, the Planner, the frontend.
        """
        # 1. Compute activations — which synapses are hot?
        activations = self._compute_activations(current_values)

        # 2. Predict cascades — both threat and reinforcement
        threat_cascades, reinforcement_cascades = self._cascade.predict(
            activations, current_values,
        )

        # 3. Hard rules — non-negotiable safety checks
        hard_fires = self._hard_rules.evaluate(current_values)

        # 4. Domain severities — merge synapse + hard rules
        synapse_severities = self._domain_severities_from_activations(activations)
        domain_severities = HardRuleEngine.corroborate(synapse_severities, hard_fires)

        # 5. Domain strengths — the positive side
        domain_strengths = self._domain_strengths_from_activations(activations)

        # 6. Positive signals — thriving, streaks, momentum
        thriving = self._detect_thriving(current_values)
        streaks = self._detect_streaks(current_values)
        momentum = self._detect_momentum(current_values, trend_report)

        # 7. Coverage — what can we see?
        coverage = self._information.compute_coverage(current_values)

        # 8. Resilience — per-domain protection vs vulnerability
        resilience = self._resilience.compute(current_values)

        # 9. Recovery state
        recovery = self._recovery.update(current_values, domain_severities)

        # 10. Data requests — what's missing that would help?
        data_requests = self._information.marginal_value_requests(current_values)

        # 11. Confidence degradation based on coverage
        if coverage.coverage_fraction < 0.8:
            for domain in domain_severities:
                if not any(f.rule_id for f in hard_fires
                           if HardRuleEngine.rule_to_domain(f.rule_id) == domain):
                    domain_severities[domain] *= coverage.coverage_fraction

        # 12. Overall scores
        severity_values = [v for v in domain_severities.values() if v > 0]
        overall_severity = max(severity_values) if severity_values else 0.0

        strength_values = [v for v in domain_strengths.values() if v > 0]
        overall_vitality = (sum(strength_values) / max(len(strength_values), 1)) if strength_values else 0.0

        return SynapseAssessment(
            mode=recovery.mode,
            trajectory=recovery.trajectory,
            activated_synapses=activations[:20],
            cascade_predictions=threat_cascades,
            hard_rule_fires=hard_fires,
            thriving_metrics=thriving,
            active_streaks=streaks,
            momentum_signals=momentum,
            reinforcement_cascades=reinforcement_cascades,
            domain_severities={k: round(v, 3) for k, v in domain_severities.items()},
            domain_strengths={k: round(v, 3) for k, v in domain_strengths.items()},
            overall_severity=round(overall_severity, 3),
            overall_vitality=round(overall_vitality, 3),
            coverage=coverage,
            resilience=resilience,
            recovery=recovery,
            data_requests=data_requests,
        )

    def _compute_activations(
        self, current_values: dict[str, float],
    ) -> list[ActivatedSynapse]:
        """Determine which synapses are currently active."""
        results = []
        for syn in self._network.all():
            val_a = current_values.get(syn.metric_a)
            if val_a is None:
                continue

            # Check threshold gate
            if syn.threshold_a is not None:
                # For "reinforces" synapses, threshold_a is the thriving threshold
                # The metric must exceed it (higher_is_better) or be below it
                spec = self._get_spec(syn.metric_a)
                if spec and spec.higher_is_better:
                    if val_a < syn.threshold_a:
                        continue
                elif spec and not spec.higher_is_better:
                    if val_a > syn.threshold_a:
                        continue
                else:
                    if val_a < syn.threshold_a:
                        continue

            # Compute activation level
            activation = self._activation_level(syn, val_a)
            if activation < 0.05:
                continue

            results.append(ActivatedSynapse(
                synapse_id=syn.synapse_id,
                activation_level=round(activation, 4),
                source_metric_value=val_a,
                target_metric=syn.metric_b,
                relationship=syn.relationship,
            ))

        results.sort(key=lambda a: a.activation_level, reverse=True)
        return results

    def _activation_level(self, syn, source_value: float) -> float:
        """Compute how strongly a synapse is firing."""
        spec = self._get_spec(syn.metric_a)
        if spec is None:
            return syn.weight * 0.3

        level = spec.evaluate(source_value)

        # Activation depends on how far the value is from neutral
        if syn.relationship == "threatens":
            # Threat synapses activate more when the source is degraded
            level_multiplier = {"critical": 1.0, "warning": 0.6, "healthy": 0.1, "thriving": 0.0}
        elif syn.relationship == "reinforces":
            # Reinforcement synapses activate when the source is thriving
            level_multiplier = {"thriving": 1.0, "healthy": 0.3, "warning": 0.0, "critical": 0.0}
        else:  # protects
            # Protective synapses: active when source is strong, vulnerability when weak
            level_multiplier = {"thriving": 0.8, "healthy": 0.4, "warning": -0.3, "critical": -0.6}

        multiplier = level_multiplier.get(level, 0.2)
        return max(0.0, syn.weight * abs(multiplier))

    def _domain_severities_from_activations(
        self, activations: list[ActivatedSynapse],
    ) -> dict[str, float]:
        """Compute per-domain severity from threatening activations."""
        severities: dict[str, float] = {}
        for act in activations:
            if act.relationship != "threatens":
                continue
            domain = _domain_from_key(act.target_metric)
            current = severities.get(domain, 0.0)
            severities[domain] = max(current, act.activation_level)
        return severities

    def _domain_strengths_from_activations(
        self, activations: list[ActivatedSynapse],
    ) -> dict[str, float]:
        """Compute per-domain strength from reinforcing/protective activations."""
        strengths: dict[str, float] = {}
        for act in activations:
            if act.relationship not in ("reinforces", "protects"):
                continue
            domain = _domain_from_key(act.target_metric)
            current = strengths.get(domain, 0.0)
            strengths[domain] = max(current, act.activation_level)
        return strengths

    def _detect_thriving(self, current_values: dict[str, float]) -> list[dict]:
        """Detect metrics that are in thriving range."""
        results = []
        try:
            from ..metric_catalog.catalog import ALL_METRICS
        except ImportError:
            return results

        for key, value in current_values.items():
            spec = ALL_METRICS.get(key)
            if spec and spec.evaluate(value) == "thriving":
                results.append({
                    "key": key,
                    "value": value,
                    "level": "thriving",
                    "message": spec.win_description or f"{spec.name} is thriving.",
                })
        return results

    def _detect_streaks(self, current_values: dict[str, float]) -> list[dict]:
        """Detect streak-worthy metrics. Full streak tracking needs observation
        history — for now, flag metrics that are currently thriving and have
        streak descriptions defined."""
        results = []
        try:
            from ..metric_catalog.catalog import STREAK_METRICS
        except ImportError:
            return results

        for key, spec in STREAK_METRICS.items():
            value = current_values.get(key)
            if value is not None and spec.evaluate(value) == "thriving":
                results.append({
                    "key": key,
                    "value": value,
                    "target_days": spec.thriving_streak_days,
                    "message": spec.streak_description or f"{spec.name} streak active.",
                })
        return results

    def _detect_momentum(
        self, current_values: dict[str, float], trend_report: Any,
    ) -> list[dict]:
        """Detect improving metrics (momentum signals)."""
        results = []
        try:
            from ..metric_catalog.catalog import MOMENTUM_METRICS
        except ImportError:
            return results

        if trend_report is None:
            return results

        for key, spec in MOMENTUM_METRICS.items():
            if key not in current_values:
                continue

            # Check if the trend engine has this metric improving
            # Try both catalog key and legacy key formats
            snapshot = getattr(trend_report, "metrics", {}).get(key)
            if snapshot is None:
                continue

            trend = getattr(snapshot, "trend", None)
            if trend is None:
                continue

            direction = getattr(trend, "direction", None)
            if direction is not None and str(direction).lower().endswith("improving"):
                results.append({
                    "key": key,
                    "value": current_values[key],
                    "direction": "improving",
                    "message": spec.momentum_description or f"{spec.name} is improving.",
                })

        return results

    def _get_spec(self, metric_key: str):
        try:
            from ..metric_catalog.catalog import ALL_METRICS
            return ALL_METRICS.get(metric_key)
        except ImportError:
            return None
