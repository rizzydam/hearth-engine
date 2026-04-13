"""
Cascade predictor — follow synapse chains to predict what happens next.

If sleep drops, stress rises, spending spikes — that's not three events,
it's one cascade. This module detects them before they complete.
"""

from __future__ import annotations

from .contracts import ActivatedSynapse, CascadePrediction
from .network import SynapseNetwork


class CascadePredictor:
    """Walk outgoing synapses from activated connections to predict chains."""

    def __init__(self, network: SynapseNetwork):
        self._network = network

    def predict(
        self,
        activations: list[ActivatedSynapse],
        current_values: dict[str, float],
        max_depth: int = 3,
    ) -> tuple[list[CascadePrediction], list[CascadePrediction]]:
        """Predict cascades from currently activated synapses.

        Returns two lists:
        - threat_cascades: downward spirals (threatens/negative chains)
        - reinforcement_cascades: upward spirals (reinforces/positive chains)
        """
        threat_cascades: list[CascadePrediction] = []
        reinforcement_cascades: list[CascadePrediction] = []

        for act in activations:
            syn = self._network.get(act.synapse_id)
            if not syn:
                continue

            chains = self._walk_chain(
                syn.metric_b, current_values,
                depth=0, max_depth=max_depth,
                visited={syn.synapse_id},
                relationship_filter=syn.relationship,
            )

            for chain_synapses in chains:
                full_chain = [syn.metric_a] + [s.metric_b for s in chain_synapses]
                probability = act.activation_level
                for cs in chain_synapses:
                    probability *= cs.weight

                time_window = sum(
                    s.lag_window[1] for s in chain_synapses
                ) + syn.lag_window[1]

                severity = self._chain_severity(chain_synapses, current_values)

                prediction = CascadePrediction(
                    chain=full_chain,
                    probability=round(probability, 4),
                    time_window_days=time_window,
                    severity=round(severity, 3),
                    relationship=syn.relationship,
                )

                if syn.relationship in ("threatens",):
                    threat_cascades.append(prediction)
                elif syn.relationship in ("reinforces", "protects"):
                    reinforcement_cascades.append(prediction)
                else:
                    threat_cascades.append(prediction)

        # Sort by impact (probability * severity)
        threat_cascades.sort(key=lambda c: c.probability * c.severity, reverse=True)
        reinforcement_cascades.sort(key=lambda c: c.probability * c.severity, reverse=True)

        return threat_cascades[:10], reinforcement_cascades[:10]

    def _walk_chain(
        self,
        metric: str,
        current_values: dict[str, float],
        depth: int,
        max_depth: int,
        visited: set[str],
        relationship_filter: str,
    ) -> list[list]:
        """Recursively follow outgoing synapses of the same relationship type."""
        if depth >= max_depth:
            return []

        chains = []
        outgoing = self._network.get_outgoing(metric)
        # Only follow synapses of the same relationship type (threats chain with threats)
        outgoing = [s for s in outgoing
                    if s.relationship == relationship_filter and s.synapse_id not in visited]
        outgoing.sort(key=lambda s: s.weight, reverse=True)

        for syn in outgoing[:3]:  # Limit branching factor
            visited.add(syn.synapse_id)
            chains.append([syn])

            # Continue the chain
            sub_chains = self._walk_chain(
                syn.metric_b, current_values,
                depth + 1, max_depth,
                visited, relationship_filter,
            )
            for sc in sub_chains:
                chains.append([syn] + sc)

            visited.discard(syn.synapse_id)

        return chains

    def _chain_severity(
        self,
        chain: list,
        current_values: dict[str, float],
    ) -> float:
        """Estimate severity of a cascade chain completing."""
        if not chain:
            return 0.0

        # Severity increases with chain length and weight
        avg_weight = sum(s.weight for s in chain) / len(chain)
        length_factor = min(len(chain) / 3.0, 1.0)  # Longer chains are more severe

        return avg_weight * length_factor
