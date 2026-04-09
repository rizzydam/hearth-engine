"""
Resilience computer — per-domain protective vs vulnerability balance.

"If stress hits right now, what's protecting you? What's exposed?"
"""

from __future__ import annotations

from .contracts import ResilienceVector
from .network import SynapseNetwork, _domain_from_key


class ResilienceComputer:
    """Compute per-domain resilience from the synapse network and current values."""

    DOMAINS = ("fitness", "project", "business")

    def __init__(self, network: SynapseNetwork):
        self._network = network

    def compute(self, current_values: dict[str, float]) -> list[ResilienceVector]:
        """Compute resilience vectors for all domains.

        For each domain:
        - Protective factors: metrics in other domains that protect this one,
          currently in healthy/thriving range
        - Vulnerability factors: metrics that threaten this domain,
          currently in warning/critical range
        - Net resilience: buffer - vulnerability
        """
        vectors = []

        for domain in self.DOMAINS:
            protective: list[dict] = []
            vulnerable: list[dict] = []

            # Find all synapses targeting this domain
            for syn in self._network.all():
                target_domain = _domain_from_key(syn.metric_b)
                source_domain = _domain_from_key(syn.metric_a)

                if target_domain != domain:
                    continue
                if source_domain == domain:
                    continue  # Self-domain connections aren't cross-domain resilience

                source_value = current_values.get(syn.metric_a)
                if source_value is None:
                    continue

                # Evaluate the source metric
                level = self._evaluate_metric(syn.metric_a, source_value)

                if syn.relationship in ("protects", "reinforces"):
                    if level in ("thriving", "healthy"):
                        protective.append({
                            "factor": syn.metric_a,
                            "current_value": source_value,
                            "level": level,
                            "synapse_weight": syn.weight,
                            "relationship": syn.relationship,
                        })
                    elif level in ("warning", "critical"):
                        # A degraded protective factor IS a vulnerability
                        vulnerable.append({
                            "factor": syn.metric_a,
                            "current_value": source_value,
                            "level": level,
                            "synapse_weight": syn.weight,
                            "relationship": "degraded_protector",
                        })
                elif syn.relationship == "threatens":
                    if level in ("warning", "critical"):
                        vulnerable.append({
                            "factor": syn.metric_a,
                            "current_value": source_value,
                            "level": level,
                            "synapse_weight": syn.weight,
                            "relationship": syn.relationship,
                        })

            # Compute scores
            buffer_score = 0.0
            if protective:
                buffer_score = min(1.0, sum(p["synapse_weight"] for p in protective) / max(len(protective), 1))

            vuln_score = 0.0
            if vulnerable:
                vuln_score = min(1.0, sum(v["synapse_weight"] for v in vulnerable) / max(len(vulnerable), 1))

            net = max(0.0, min(1.0, buffer_score - vuln_score))

            # Scenario answers: what if a specific threat hits?
            scenarios = self._scenario_answers(domain, protective, vulnerable)

            vectors.append(ResilienceVector(
                domain=domain,
                buffer_score=round(buffer_score, 3),
                protective_factors=protective[:5],  # Top 5
                vulnerability_factors=vulnerable[:5],
                net_resilience=round(net, 3),
                scenario_answers=scenarios,
            ))

        return vectors

    def _evaluate_metric(self, metric_key: str, value: float) -> str:
        """Use the metric catalog to evaluate a value."""
        try:
            from ..metric_catalog.catalog import ALL_METRICS
            spec = ALL_METRICS.get(metric_key)
            if spec:
                return spec.evaluate(value)
        except ImportError:
            pass
        return "healthy"

    def _scenario_answers(
        self,
        domain: str,
        protective: list[dict],
        vulnerable: list[dict],
    ) -> dict[str, float]:
        """Compute "if X happens, how protected am I?" answers."""
        scenarios: dict[str, float] = {}

        # "If stress hits": how much protection from protective factors?
        if protective:
            stress_protection = sum(
                p["synapse_weight"] for p in protective
                if p.get("level") in ("thriving", "healthy")
            ) / max(len(protective), 1)
            scenarios["if_stress_hits"] = round(min(1.0, stress_protection), 3)

        # "If income drops": relevant for financial domain
        if domain == "financial":
            financial_buffers = [p for p in protective if "saving" in p.get("factor", "").lower()
                                 or "income" in p.get("factor", "").lower()]
            if financial_buffers:
                scenarios["if_income_drops"] = round(
                    min(1.0, sum(p["synapse_weight"] for p in financial_buffers)), 3
                )

        # Overall vulnerability count
        scenarios["active_vulnerabilities"] = float(len(vulnerable))
        scenarios["active_protections"] = float(len(protective))

        return scenarios
