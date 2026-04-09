"""
Delta computer — finds where hypotheses disagree with objective data
and converts every disagreement into an action.

The delta is never just informational. It always becomes:
- A goal (high magnitude)
- A probing trigger (medium magnitude)
- A data request (low magnitude)
"""

from __future__ import annotations

import logging

from .models import Hypothesis, DeltaEntry
from .contracts import _new_id

log = logging.getLogger(__name__)


class DeltaComputer:
    """Compare hypothesis predictions against objective synapse weights."""

    def __init__(self, network, registry):
        self._network = network
        self._registry = registry

    def compute(self, current_values: dict[str, float]) -> list[DeltaEntry]:
        """Produce actionable deltas for all active hypotheses.

        For each hypothesis that has been tested enough:
        - Compare its predicted weight/direction to what Hebbian learning found
        - Classify the disagreement
        - Generate an action
        """
        deltas = []

        for h in self._registry.active():
            if h.predictions_made < 5:
                # Not enough data to evaluate — generate data request
                if h.metric_a in current_values or h.metric_b in current_values:
                    missing = h.metric_b if h.metric_a in current_values else h.metric_a
                    deltas.append(DeltaEntry(
                        delta_id=_new_id("delta"),
                        hypothesis_id=h.hypothesis_id,
                        delta_type="untested",
                        domain=_domain_from_key(h.metric_b),
                        hypothesis_says=h.claim,
                        data_shows=f"Insufficient data ({h.predictions_made} observations)",
                        magnitude=0.1,
                        action_type="data_request",
                        action_detail=f"Log {missing} to test: {h.claim[:60]}",
                    ))
                continue

            # Find objective synapse weight
            syn = self._network.find_existing(h.metric_a, h.metric_b)
            obj_weight = syn.weight if syn else 0.0

            # Hypothesis-predicted effective weight
            nar_weight = obj_weight * h.weight_modifier

            delta_mag = abs(obj_weight - nar_weight)
            if delta_mag < 0.05:
                # Models agree
                deltas.append(DeltaEntry(
                    delta_id=_new_id("delta"),
                    hypothesis_id=h.hypothesis_id,
                    delta_type="confirmed",
                    domain=_domain_from_key(h.metric_b),
                    hypothesis_says=h.claim,
                    data_shows=f"Data confirms (objective={obj_weight:.2f}, hypothesis={nar_weight:.2f})",
                    magnitude=delta_mag,
                    action_type="",
                    action_detail="",
                ))
                continue

            # Classify the disagreement
            if h.confidence < 0.3 and h.predictions_made >= 10:
                # Hypothesis is failing — narrative overestimates
                delta_type = "growth"  # Person is more resilient than they think
                action_type = "probing_trigger" if delta_mag > 0.2 else "data_request"
                interpretation = (
                    f"Your profile says '{h.claim[:60]}' but data shows this connection "
                    f"is weaker than expected (obj={obj_weight:.2f}). "
                    f"This may be growth you haven't recognized."
                )
            elif obj_weight > nar_weight * 1.3:
                # Objective is stronger — data shows something narrative doesn't mention
                delta_type = "blind_spot"
                action_type = "goal" if delta_mag > 0.3 else "probing_trigger"
                interpretation = (
                    f"Data shows a strong {h.metric_a}→{h.metric_b} connection "
                    f"(obj={obj_weight:.2f}) that your profile underweights. Blind spot."
                )
            elif nar_weight > obj_weight * 1.3:
                # Narrative is stronger — person believes connection is stronger than data shows
                delta_type = "growth"
                action_type = "probing_trigger"
                interpretation = (
                    f"Your profile predicts stronger {h.metric_a}→{h.metric_b} "
                    f"(nar={nar_weight:.2f}) than data shows (obj={obj_weight:.2f}). "
                    f"Possible resilience or historical pattern."
                )
            else:
                delta_type = "evolution"
                action_type = "probing_trigger"
                interpretation = f"Pattern may be evolving: {h.claim[:60]}"

            # Convert to action detail
            domain = _domain_from_key(h.metric_b)
            if action_type == "goal":
                action_detail = f"Improve {domain}: {h.claim[:50]}"
            elif action_type == "probing_trigger":
                action_detail = interpretation
            else:
                action_detail = f"Log more data to clarify: {h.claim[:50]}"

            deltas.append(DeltaEntry(
                delta_id=_new_id("delta"),
                hypothesis_id=h.hypothesis_id,
                delta_type=delta_type,
                domain=domain,
                hypothesis_says=h.claim,
                data_shows=interpretation,
                magnitude=delta_mag,
                action_type=action_type,
                action_detail=action_detail,
            ))

        # Sort by magnitude (most significant first)
        deltas.sort(key=lambda d: d.magnitude, reverse=True)
        return deltas


def _domain_from_key(metric_key: str) -> str:
    try:
        from ..metric_catalog.catalog import ALL_METRICS, ALL_SENSES
        spec = ALL_METRICS.get(metric_key)
        if spec:
            return ALL_SENSES[spec.sense].domain
    except ImportError:
        pass
    return metric_key.split(".")[0].replace("sense", "")
